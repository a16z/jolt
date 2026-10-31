"""Render synthetic publisher output through HTTP in bundled headless WebKit."""

import copy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from scripts.tests.test_benchmark_history import (
    REVISION, SCRIPT, collected_fixture, collected_push, history, synthetic_result,
)

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    sync_playwright = None


@unittest.skipUnless(sync_playwright, "Run with uv run --with playwright==1.61.0 for browser checks")
class BenchmarkHistoryBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.site = cls.directory / "site"
        for number in [*range(101, 113), 114, 115, 116, 117]:
            data = synthetic_result(number, tier="nightly" if number == 110 else "main", revision=f"{number:040x}")
            data["started_at"] = f"2026-01-{number - 100:02}T00:00:00Z"
            data["finished_at"] = f"2026-01-{number - 100:02}T00:01:00Z"
            case = data["cases"][0]
            if number == 101:
                for sample, proof_bytes, trace_rows in zip(case["samples"][1:], [123455, 123457], [4194300, 4194302]):
                    sample["metrics"].update(proof_bytes=proof_bytes, trace_rows=trace_rows, padded_trace_rows=4194304)
            if number in {102, 104}:
                for sample, value in zip(case["samples"][1:], [20, 24] if number == 102 else [30, 34]):
                    sample["metrics"]["e2e_s"] = value
            if number == 102:
                data["builds"][0].update(source_cargo_lock_sha256="b" * 64, cargo_lock_sha256="c" * 64)
            if number == 103:
                data["status"] = case["status"] = "failed"
                case["reason"] = '<img src="https://invalid.example/attack" onerror="window.__historyAttack=1">'
                case["samples"][-1].update(status="failed", exit_code=1)
                case["samples"][-1]["metrics"] = {metric: None for metric in history.METRICS}
            if number == 105:
                case["samples"][-1]["metrics"]["peak_rss_bytes"] = None
            if number == 106:
                data["builds"][0]["rustflags"] = "different flags"
            if number == 117:
                data["host"]["settings"]["boost"] = False
            if number in {107, 110}:
                case["scale_log2"] = 20 if number == 107 else 22
                case["input"]["parameters"]["iterations"] = 200
                case["input"]["sha256"] = "f" * 64
            if number in {108, 109}:
                case["physical_cores"] = 8 if number == 108 else 24
                case["cpu_ids"] = list(range(case["physical_cores"]))
            if number == 112:
                data.update(status="disabled", builds=[], cases=[], reason="SYNTHETIC runner disabled")
            if number == 114:
                data["builds"][0]["status"] = data["status"] = "failed"
            if number in {115, 116}:
                case["pcs"] = "akita"
                data["builds"][0]["akita_schedules"] = [{"path": "akita-schedules/synthetic.aks", "sha256": f"{number:064x}"}]
            record, raw = collected_fixture(data, missing=number == 111)
            history.append_site(cls.site, [(record, raw)])
        manual = synthetic_result(113, tier="pr")
        manual["kind"] = "null"
        manual["revisions"]["base"] = REVISION
        manual["provenance"] = {key: None for key in manual["provenance"]}
        manual["builds"].append(dict(copy.deepcopy(manual["builds"][0]), leg="base"))
        samples = manual["cases"][0]["samples"]
        samples.extend([dict(copy.deepcopy(sample), leg="base", order=sample["order"] + 3) for sample in samples])
        manual_file = cls.directory / "synthetic-preview.json"
        manual_file.write_text(json.dumps(manual))
        subprocess.run([
            sys.executable, str(SCRIPT), "preview", "--input", str(manual_file),
            "--site", str(cls.site), "--label", "SYNTHETIC null control",
        ], check=True, capture_output=True, text=True)

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(cls.site)))
        cls.server.daemon_threads = False
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.start()

        def close_server():
            cls.server.shutdown()
            cls.server.server_close()
            cls.thread.join()

        cls.addClassCleanup(close_server)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/dev/prover-history/"
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        cls.browser = cls.playwright.webkit.launch(headless=True)
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        self.page = self.browser.new_page(viewport={"width": 1280, "height": 900})
        self.errors = []
        self.requests = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.on("console", lambda message: self.errors.append(message.text) if message.type == "error" else None)
        self.page.on("requestfailed", lambda request: self.errors.append(request.url))
        self.page.on("request", lambda request: self.requests.append(request.url))
        self.addCleanup(self.page.close)
        self.page.goto(self.url)
        expect(self.page.locator("#load-status")).to_contain_text("archived run records")
        self.assertEqual(self.page.title(), "Jolt · CPU benchmark history")

    def select_point(self, *, cores=1, scale=16, leg="head", pcs="dory"):
        values = self.page.locator("#point-select option").evaluate_all("options => options.map(o => o.value)")
        selected = next(value for value in values if all([
            json.loads(value)["physical_cores"] == cores,
            json.loads(value)["scale_log2"] == scale,
            json.loads(value)["leg"] == leg,
            json.loads(value)["pcs"] == pcs,
        ]))
        self.page.select_option("#point-select", selected)
        regimes = self.page.locator("#regime-select option").evaluate_all("options => options.map(o => o.value)")
        baseline = next(value for value in regimes if json.loads(value)["host"]["settings"]["boost"]
                        and json.loads(value)["build_settings"]["rustflags"] == "-C target-cpu=synthetic")
        self.page.select_option("#regime-select", baseline)

    def assert_clean(self):
        self.assertEqual(self.errors, [])
        self.assertTrue(all(url.startswith(self.url) for url in self.requests), self.requests)

    def test_dimension_selectors_keep_revisions_comparable_without_pooling(self):
        self.select_point()
        with self.subTest(pcs="dory"):
            self.assertEqual(self.page.locator("#regime-select option").count(), 3)
            self.assertEqual(self.page.locator(".chart-point").count(), 4)
            self.assertEqual(self.page.locator("#history-rows tr").filter(has_text="Run 101, attempt 1").locator("td").nth(2).inner_text().splitlines()[0], "12")
            changed = self.page.locator("#history-rows tr").filter(has_text="Run 102, attempt 1")
            self.assertEqual(changed.locator("td").nth(2).inner_text().splitlines()[0], "22")
            build = json.loads(changed.locator("pre").text_content())["build"]
            self.assertEqual(build["source_cargo_lock_sha256"], "b" * 64)
            self.assertEqual(build["cargo_lock_sha256"], "c" * 64)
        self.select_point(cores=8)
        self.assertEqual(self.page.locator(".chart-point").count(), 1)
        self.select_point(cores=24)
        self.assertEqual(self.page.locator(".chart-point").count(), 1)
        self.select_point(scale=20)
        self.assertEqual(self.page.locator(".chart-point").count(), 1)
        self.select_point(pcs="akita")
        with self.subTest(pcs="akita"):
            self.assertEqual(self.page.locator("#regime-select option").count(), 1)
            self.assertEqual(self.page.locator(".chart-point").count(), 2)
            for number in (115, 116):
                changed = self.page.locator("#history-rows tr").filter(has_text=f"Run {number}, attempt 1")
                build = json.loads(changed.locator("pre").text_content())["build"]
                self.assertEqual(build["akita_schedules"][0]["sha256"], f"{number:064x}")
        self.page.get_by_role("button", name="Nightly", exact=True).click()
        expect(self.page.locator("#window-label")).to_have_text("Showing the newest 1 of 1 run records. Dates use UTC.")
        self.assertEqual(self.page.locator("#point-select option").count(), 1)
        self.assertIn("2^22 input", self.page.locator("#point-select").inner_text())
        self.assertEqual(self.page.locator("#history-rows tr").count(), 1)
        self.assert_clean()

    def test_failed_and_missing_values_break_lines_and_never_execute_data(self):
        self.select_point()
        failed = self.page.locator("#history-rows tr").filter(has_text="Run 103, attempt 1")
        self.assertEqual(failed.locator("td").nth(2).inner_text(), "—")
        self.assertIn("failed", failed.inner_text())
        self.assertIn("<img", failed.inner_text())
        self.assertEqual(self.page.locator("img").count(), 0)
        self.assertIsNone(self.page.evaluate("window.__historyAttack"))
        paths = self.page.locator(".chart-line").evaluate_all("paths => paths.map(p => p.getAttribute('points').trim().split(' ').length)")
        self.assertEqual(paths, [2, 2])
        self.page.select_option("#metric-select", "peak_rss_bytes")
        missing_metric = self.page.locator("#history-rows tr").filter(has_text="Run 105, attempt 1")
        self.assertEqual(missing_metric.locator("td").nth(2).inner_text(), "—")
        self.assertIn("missing", missing_metric.inner_text())
        self.assertEqual(self.page.locator("#history-rows tr").filter(has_text="Run 101, attempt 1").locator("td").nth(2).inner_text().splitlines()[0], "100")
        self.assertIn("missing", self.page.locator("#history-rows tr").filter(has_text="Run 111, attempt 1").inner_text())
        self.assertIn("disabled", self.page.locator("#history-rows tr").filter(has_text="Run 112, attempt 1").inner_text())
        self.assertEqual(self.page.locator("#history-rows tr").filter(has_text="Run 114, attempt 1").locator("td").nth(2).inner_text(), "—")
        self.page.select_option("#metric-select", "verify_s")
        self.assertEqual(self.page.locator("#history-rows tr").filter(has_text="Run 101, attempt 1").locator("td").nth(2).inner_text().splitlines()[0], "0.3")
        self.assert_clean()

    def test_integer_metrics_keep_all_digits_in_values_ranges_and_tooltips(self):
        self.select_point()
        row = self.page.locator("#history-rows tr").filter(has_text="Run 101, attempt 1")
        for metric, value, extent in (
            ("proof_bytes", "123,456", "123,455–123,457"),
            ("trace_rows", "4,194,301", "4,194,300–4,194,302"),
            ("padded_trace_rows", "4,194,304", "4,194,304–4,194,304"),
        ):
            with self.subTest(metric=metric):
                self.page.select_option("#metric-select", metric)
                self.assertEqual(row.locator("td").nth(2).inner_text().splitlines(), [value, extent])
                self.assertIn(f"{101:040x}"[:12] + f": {value}", self.page.locator(".chart-point title").all_text_contents())
        self.assert_clean()

    def test_multi_commit_records_and_range_limits_are_visible(self):
        records, raw_index = collected_push()
        history.append_site(self.site / "push-fixture", records, raw_index)
        self.assert_clean()
        self.requests.clear()
        self.url = self.url.replace("/dev/prover-history/", "/push-fixture/dev/prover-history/")
        self.page.goto(self.url)
        expect(self.page.locator("#load-status")).to_contain_text("archived run records")
        self.assertEqual(self.page.locator("#history-rows tr").count(), 3)
        self.assertEqual(self.page.locator(".chart-point").count(), 2)
        self.assertEqual(self.page.locator(".chart-gap").count(), 1)
        self.assertEqual(self.page.locator(".chart-line").count(), 0)
        self.assertIn("not independently verified", self.page.locator("#history-rows").inner_text())
        self.assertEqual(self.page.get_by_role("link", name="Push index", exact=True).count(), 3)
        self.assertIn("Main commit 3 of 3", self.page.locator("#history-rows tr").first.inner_text())
        self.assert_clean()

    def test_responsive_controls_raw_download_and_manual_purpose(self):
        self.select_point()
        for width in (390, 768, 1280):
            self.page.set_viewport_size({"width": width, "height": 900})
            chart_width = round(self.page.locator("#chart").bounding_box()["width"])
            expect(self.page.locator("#chart svg")).to_have_attribute("viewBox", f"0 0 {chart_width} 280")
            dimensions = self.page.evaluate("""() => {
                const main = document.querySelector('main').getBoundingClientRect();
                return {width: innerWidth, client: document.documentElement.clientWidth,
                    scroll: document.documentElement.scrollWidth, left: main.left, right: innerWidth - main.right};
            }""")
            self.assertLessEqual(dimensions["scroll"], dimensions["client"])
            self.assertAlmostEqual(dimensions["left"], dimensions["right"], delta=1)
            self.assertGreaterEqual(self.page.locator("#point-select").bounding_box()["height"], 44)
        self.page.locator("#configuration summary").click()
        self.assertTrue(self.page.locator("#configuration").evaluate("node => node.open"))
        self.assertIn('"physical_cores": 1', self.page.locator("#configuration-value").inner_text())
        self.page.locator("#configuration summary").click()
        download_link = self.page.get_by_role("link", name="Raw JSON", exact=True).first
        response = self.page.request.get(self.url + download_link.get_attribute("href"))
        self.assertEqual(response.status, 200)
        self.assertIn("application/json", response.headers["content-type"])
        self.assertEqual(response.json()["schema_version"], 1)
        with self.page.expect_download() as pending:
            download_link.click()
        download = pending.value
        self.assertTrue(download.suggested_filename.endswith(".json"))
        downloaded = self.directory / "downloaded.json"
        download.save_as(downloaded)
        self.assertEqual(json.loads(downloaded.read_text()), response.json())
        self.page.get_by_role("button", name="Manual previews", exact=True).click()
        expect(self.page.locator("#window-label")).to_have_text("Showing the newest 1 of 1 run records. Dates use UTC.")
        self.assertIn("Manual preview only", self.page.locator("#notice").inner_text())
        self.assertIn("SYNTHETIC null control", self.page.locator("#history-rows").inner_text())
        self.assertIn("pr / null", self.page.locator("#history-rows").inner_text())
        self.assertEqual(self.page.locator("#point-select option").count(), 2)
        self.select_point(leg="head")
        self.assertIn("12", self.page.locator("#history-rows tr td").nth(2).inner_text())
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()
