"""瀏覽器 e2e:以 tiny 模型跑真實 server + headless Chromium。

需要 playwright chromium(`uv run playwright install chromium`)。
"""

import threading
import time

import pytest
import uvicorn

import jspace.server as server
from tests.conftest import make_tiny_bundle

PORT = 7899


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    server.load_model = lambda mid, **kw: make_tiny_bundle()
    server.cache.DEFAULT_CACHE_DIR = tmp_path_factory.mktemp("cache")
    config = uvicorn.Config(server.app, host="127.0.0.1", port=PORT, log_level="warning")
    srv = uvicorn.Server(config)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(50):
        if srv.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{PORT}"
    srv.should_exit = True
    t.join(timeout=5)


def test_full_flow_in_browser(live_server, page):
    page.goto(live_server)
    # 模型清單載入
    page.wait_for_function("document.getElementById('model-select').options.length > 0")
    # 未預計算 → 顯示預計算橫幅
    assert page.locator("#precompute-banner").is_visible()

    # 輸入文字並分析
    page.fill("#text-input", "hello")
    page.click("#analyze-btn")
    page.wait_for_selector("table.grid", timeout=30000)

    # 格子:欄數 = token 數 + 層標籤欄
    header_cells = page.locator("table.grid tr").first.locator("th")
    assert header_cells.count() == len("hello") + 1

    # tiny 模型無 J-lens → 自動切到 logit lens
    active = page.locator("#lens-toggle button.active")
    assert active.get_attribute("data-lens") == "logit"

    # 點一個格子 → 側欄 top-k 長條出現
    page.locator("table.grid td").first.click()
    page.wait_for_selector("#sidebar:not(.hidden)")
    assert page.locator("#sidebar-bars .bar-row").count() > 0

    # 無 console 錯誤
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    assert errors == []
