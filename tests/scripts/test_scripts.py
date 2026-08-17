"""Smoke-test the executable public-API scripts without requiring Wasmtime."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


SCRIPTS = {
    "wasm_examples.py": ((), "compiled"),
    "numeric_runtime.py": ((), "compiled"),
    "allocator_runtime.py": ((), "compiled"),
    "benchmark.py": (("--calls", "1", "--samples", "1", "--limit", "10"), "Wasm batched execution"),
    "limitations.py": ((), "expected compiler rejection"),
}
SCRIPT_DIRECTORY = Path(__file__).parent


class ScriptSmokeTests(unittest.TestCase):
    def test_scripts_compile_and_report_an_optional_runtime(self):
        for script, (arguments, expected_output) in SCRIPTS.items():
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, SCRIPT_DIRECTORY / script, *arguments],
                    cwd=SCRIPT_DIRECTORY.parents[1],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("Traceback", result.stdout + result.stderr)
                self.assertIn(expected_output, result.stdout)
