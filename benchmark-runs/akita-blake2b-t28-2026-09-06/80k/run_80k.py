#!/usr/bin/env python3
"""Run the approved smaller input with the unchanged BLAKE2b observer."""
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import run as previous


if __name__ == "__main__":
    def stop(_signum, _frame):
        raise KeyboardInterrupt("BLAKE2b 80k observation interrupted")

    signal.signal(signal.SIGTERM, stop)
    previous.ROOT = ROOT
    (ROOT / "runs").mkdir(exist_ok=True)
    previous.main()
