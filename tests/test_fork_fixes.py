"""Regression tests for the fork's institutional-download fixes.

Each test pins one bug that was hit live (BUAA campus/WebVPN/CARSI setup);
all run offline with fakes — no browser, no network.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scansci_pdf import browser_backend as bb
from scansci_pdf import browser_pdf
from scansci_pdf import sources


# ---------------------------------------------------------------- browser setup

def test_direct_args_bypass_system_proxy():
    assert bb._direct_args(bb.BACKEND_CLOAKBROWSER, None, ["--x"]) == ["--x", "--no-proxy-server"]
    # an explicit proxy wins; Firefox (camoufox) takes no Chromium flags
    assert bb._direct_args(bb.BACKEND_PATCHRIGHT, {"server": "http://p"}, ["--x"]) == ["--x"]
    assert bb._direct_args(bb.BACKEND_CAMOUFOX, None, None) is None
    # never duplicated
    assert bb._direct_args(bb.BACKEND_PATCHRIGHT, None, ["--no-proxy-server"]) == ["--no-proxy-server"]


def test_launch_config_falls_back_to_saved_config(monkeypatch):
    import scansci_pdf.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "load_config", lambda: {"browser_backend": "cloakbrowser"})
    assert bb._resolve_launch_config(None) == {"browser_backend": "cloakbrowser"}
    assert bb._resolve_launch_config({"a": 1}) == {"a": 1}


@pytest.mark.skipif(os.name != "nt", reason="stable alias is Windows-only")
def test_stable_browser_alias_created_and_refreshed(tmp_path):
    exe = tmp_path / "chromium-1.2.3" / "chrome.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"v1")
    alias = Path(bb.stable_browser_alias(exe))
    assert alias.name == bb.STABLE_BROWSER_NAME and alias.read_bytes() == b"v1"
    # binary replaced by an upgrade -> alias follows
    exe.unlink()
    exe.write_bytes(b"v2-longer")
    assert Path(bb.stable_browser_alias(exe)).read_bytes() == b"v2-longer"
    # the alias itself is passed through untouched
    assert bb.stable_browser_alias(alias) == str(alias)


def test_close_browsers_since_kills_only_newer(monkeypatch):
    import scansci_pdf.browser_engine as be
    killed = []
    monkeypatch.setattr(be, "_tree_kill", lambda proc: killed.append(proc))

    def fake(proc):
        return SimpleNamespace(_impl_obj=SimpleNamespace(
            _connection=SimpleNamespace(_transport=SimpleNamespace(_proc=proc))))

    monkeypatch.setattr(bb, "_LAUNCHED", [])
    bb._remember(fake("old"))
    t0 = time.monotonic()
    bb._remember(fake("new"))
    assert bb.close_browsers_since(t0) == 1
    assert killed == ["new"]
    assert len(bb._LAUNCHED) == 1  # the older browser stays registered


def test_launch_guard_blocks_waived_sources(monkeypatch):
    """A source thread whose download already returned must not (re)launch."""
    monkeypatch.setattr(bb, "_launch_unregistered", lambda **kw: pytest.fail("launched"))
    sources._SOURCE_TLS.doi = "10.1/settled"
    sources._SETTLED.add("10.1/settled")
    try:
        with pytest.raises(RuntimeError, match="download already finished"):
            bb.launch(headless=True)
    finally:
        sources._SOURCE_TLS.doi = None
        sources._SETTLED.discard("10.1/settled")


# ---------------------------------------------------------------- racing hygiene

def test_late_finisher_drops_its_copy(tmp_path, monkeypatch):
    out = tmp_path / "10.1_late_SlowSource.pdf"
    sources._SETTLED.add("10.1/late")

    def slow_source(doi, output_path, config):
        output_path.write_bytes(b"%PDF-1.4 late copy")
        return {"success": True, "file": str(output_path)}

    try:
        assert sources._try_source(slow_source, "10.1/late", out, {}, "SlowSource") is None
        assert not out.exists()
    finally:
        sources._SETTLED.discard("10.1/late")


def test_rename_removes_duplicate_source_copies(tmp_path, monkeypatch):
    import scansci_pdf.citation as citation
    doi = "10.1038/s1"
    safe = sources.safe_filename(doi)
    winner = tmp_path / f"{safe}.pdf"
    winner.write_bytes(b"%PDF")
    for label in ("NatureDirect", "PublisherDirect"):
        (tmp_path / f"{safe}_{label}.pdf").write_bytes(b"%PDF")
    final = tmp_path / "Wu2024_Bridging.pdf"
    monkeypatch.setattr(citation, "fetch_metadata", lambda d, c: {"title": ["x"]})
    monkeypatch.setattr(sources, "rename_pdf", lambda p, m: Path(p).rename(final) or final)
    monkeypatch.setattr(sources, "_update_doi_index", lambda *a, **k: None)
    result = {"success": True, "file": str(winner)}
    sources._auto_rename(result, doi, {"auto_rename": True}, doi=doi, target_dir=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Wu2024_Bridging.pdf"]


# ---------------------------------------------------------------- publisher URLs

def test_elsevier_pii_keeps_check_char(monkeypatch):
    import requests
    from scansci_pdf import _publisher_strategies_core as core

    class _Sess:
        trust_env = True
        headers: dict = {}

        def get(self, url, **kw):
            return SimpleNamespace(url="https://linkinghub.elsevier.com/retrieve/pii/S030626192400494X",
                                   text="sciencedirect.com")

    monkeypatch.setattr(requests, "Session", _Sess)
    assert core._resolve_elsevier_pii("10.1016/x", {}).endswith("/pii/S030626192400494X")


# ---------------------------------------------------------------- session checks

def test_webvpn_session_probe_hits_gateway_root(monkeypatch):
    from scansci_pdf.sources import instsci
    seen = []

    class _Sess:
        trust_env = True
        cookies = SimpleNamespace(update=lambda jar: None)

        def get(self, url, **kw):
            seen.append(url)
            return SimpleNamespace(url="https://d.example.edu/", status_code=200)

    monkeypatch.setattr(instsci, "_load_cookies", lambda c: {"t": "1"})
    monkeypatch.setattr(instsci, "_get_webvpn_base", lambda c: "https://d.example.edu")
    monkeypatch.setattr(instsci.requests, "Session", _Sess)
    assert instsci.session_status({}) == "valid"
    assert seen == ["https://d.example.edu/"]  # not a proxied publisher


def test_carsi_cloudflare_probe_trusts_fresh_cookies(tmp_path, monkeypatch):
    from scansci_pdf.sources import carsi
    client = carsi.CARSIClient.__new__(carsi.CARSIClient)
    client._publisher_configs = {"sd": SimpleNamespace(domains=["www.sciencedirect.com"])}
    cookie = tmp_path / "sd.json"
    cookie.write_text("[]")
    client._cookie_path = lambda p: cookie
    resp = SimpleNamespace(url="https://www.sciencedirect.com/", status_code=403,
                           headers={"server": "cloudflare", "cf-mitigated": "challenge"})
    client._get_session = lambda p: SimpleNamespace(get=lambda *a, **k: resp)
    assert client._validate_session("sd") is True
    # a stale cookie file is still rejected before any probe
    old = time.time() - 30 * 3600
    os.utime(cookie, (old, old))
    assert client._validate_session("sd") is False


# ---------------------------------------------------------------- browser_pdf helpers

class _FakePage:
    def __init__(self, titles, href=None, body=""):
        self._titles = list(titles)
        self.url = "https://pub.example/article"
        self._href = href
        self._body = body
        self.waits = 0

    def wait_for_load_state(self, *a, **k):
        pass

    def title(self):
        return self._titles.pop(0) if len(self._titles) > 1 else self._titles[0]

    def wait_for_timeout(self, ms):
        self.waits += 1

    def evaluate(self, expr, *a):
        if "innerText" in expr:
            return self._body
        if "document.readyState" in expr:
            return "interactive"
        return self._href


def test_wait_page_settled_skips_interstitials():
    page = _FakePage(["Loading https://x", "请稍候…", "Article title"])
    assert browser_pdf.wait_page_settled(page, max_wait_s=10)[0] == "Article title"


def test_wait_page_settled_stops_on_cloudflare_verdict():
    page = _FakePage(["请稍候…"], body="pubs.aip.org 浏览器不支持")
    assert browser_pdf.wait_page_settled(page, max_wait_s=30)[0] == "请稍候…"
    assert page.waits == 0  # returned at once, not after 30 s


def test_find_page_pdf_link_polls_until_rendered():
    page = _FakePage(["t"])
    hrefs = [None, None, "https://pub.example/pdfft?md5=1"]
    page.evaluate = lambda expr, *a: hrefs.pop(0)
    assert browser_pdf.find_page_pdf_link(page, max_wait_s=5) == "https://pub.example/pdfft?md5=1"


def test_capture_pdf_via_cdp_reads_paused_document():
    import base64
    pdf = b"%PDF-1.7" + b"0" * 6000
    handlers = {}
    sent = []

    class _Cdp:
        def on(self, ev, fn):
            handlers[ev] = fn

        def send(self, method, params=None):
            sent.append(method)
            if method == "Fetch.getResponseBody":
                return {"body": base64.b64encode(pdf).decode(), "base64Encoded": True}
            return {}

        def detach(self):
            pass

    class _Page(_FakePage):
        def evaluate(self, expr, *a):  # the navigation: the browser answers with a PDF
            handlers["Fetch.requestPaused"]({
                "requestId": "1", "responseStatusCode": 200,
                "responseHeaders": [{"name": "Content-Type", "value": "application/pdf"}]})

    page = _Page(["t"])
    ctx = SimpleNamespace(new_cdp_session=lambda p: _Cdp())
    assert browser_pdf.capture_pdf_via_cdp(ctx, page, "https://pub.example/pdf") == pdf
    assert "Fetch.continueRequest" in sent  # the paused response was released


# ---------------------------------------------------------------- logging

def test_plugin_logger_does_not_duplicate_to_root():
    from scansci_pdf.log import get_logger
    assert get_logger().propagate is False
    assert isinstance(logging.getLogger("scansci_pdf"), logging.Logger)


# ---------------------------------------------------------------- cancellation

@pytest.fixture
def no_browser_kill(monkeypatch):
    monkeypatch.setattr(bb, "close_browsers_since", lambda t0: 0)


def test_cancel_marks_running_download_cancelled(monkeypatch, no_browser_kill):
    seen = {}

    def impl(identifier, *a, **k):
        seen["out"] = sources.cancel_downloads()  # user cancels mid-download
        return {"success": False, "reason": "all sources failed"}

    monkeypatch.setattr(sources, "_download_impl", impl)
    r = sources.download("10.1/cancel-me")
    assert r["error_type"] == "cancelled"
    assert "10.1/cancel-me" in seen["out"]["cancelled"]


def test_download_started_after_cancel_is_unaffected(monkeypatch, no_browser_kill):
    sources.cancel_downloads()
    monkeypatch.setattr(sources, "_download_impl", lambda *a, **k: {"success": False, "reason": "x"})
    assert sources.download("10.1/after-cancel").get("error_type") != "cancelled"


def test_wait_any_returns_early_once_cancelled(no_browser_kill):
    import threading
    sources._DL_TLS.gen = sources._CANCEL_GEN[0]
    try:
        sources.cancel_downloads()
        t = time.monotonic()
        assert sources._wait_any((threading.Event(), threading.Event()), 5) is False
        assert time.monotonic() - t < 1
        # and no new race starts for a cancelled download
        assert sources._run_tiers_parallel(
            [([(lambda *a: pytest.fail("raced"), "X")], "T", 3)],
            "10.1/x", Path("."), Path("o.pdf"), {}, False, 3) is None
    finally:
        sources._DL_TLS.gen = None
