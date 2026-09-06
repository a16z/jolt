#!/usr/bin/env python3
"""Recover the pre-proof stale-lock failure with the unchanged observer."""
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))
import run as previous


if __name__ == "__main__":
    def stop(_signum, _frame):
        raise KeyboardInterrupt("BLAKE2b 80k observation interrupted")

    signal.signal(signal.SIGTERM, stop)
    previous.ROOT = ROOT
    (ROOT / "runs").mkdir(exist_ok=True)
    previous.main()
