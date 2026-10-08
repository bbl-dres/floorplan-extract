"""The page in a real browser against the stub bridge (no pipeline): it boots without a script error, the landing page
and its demo link render, and the Download menu stays hidden until there are results. Skipped when Playwright or its
Chromium is not installed (pip install playwright; python -m playwright install chromium)."""
import json

import pytest

playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture
def browser_page():
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as ex:                            # noqa: BLE001 - no browser binary: the test does not apply
            pytest.skip(f"Chromium for Playwright is not installed: {ex}")
        page = browser.new_context(viewport={"width": 1280, "height": 800}).new_page()
        yield page
        browser.close()


def test_landing_boots_without_errors(served, tmp_path, monkeypatch, browser_page):
    base, server, _ = served
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "house.png").write_bytes(b"\x89PNG fake")
    (demo / "demo.json").write_text(json.dumps({"file": "house.png", "title": "A house", "source": "test", "licence": "CC0"}), encoding="utf-8")
    monkeypatch.setattr(server, "DEMO", demo)
    errors, console = [], []
    browser_page.on("pageerror", lambda e: errors.append(str(e)))
    browser_page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
    browser_page.goto(base + "/")
    browser_page.wait_for_selector("text=Drop floor plans here", timeout=15000)
    assert browser_page.locator("text=or try a demo floor plan: A house").count() == 1
    assert browser_page.locator("#download-menu").is_hidden()
    assert browser_page.locator("#fatal").is_hidden()
    unnamed = browser_page.evaluate("""() => [...document.querySelectorAll('button, a[href], input')].filter((el) => el.getClientRects().length
      && !(el.getAttribute('aria-label') || el.textContent.trim() || el.getAttribute('title') || (el.labels && el.labels[0]))).length""")
    assert unnamed == 0
    assert errors == [] and console == []                   # no script error, no failed request on the landing page
    # the demo starts a job through the stub bridge and the page moves on to step 1 by itself (the stub has no preview
    # image, so a failed image request is expected there; a script error is not)
    browser_page.click("text=or try a demo floor plan")
    browser_page.wait_for_selector("text=Confirm the building area", timeout=15000)
    assert errors == []
