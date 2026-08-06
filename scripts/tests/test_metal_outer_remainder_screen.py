from __future__ import annotations

import unittest
from unittest import mock

from scripts import metal_outer_remainder_screen as screen


class OuterRemainderScreenTests(unittest.TestCase):
    def test_configure_closes_the_generic_evaluator_at_log_25(self) -> None:
        with mock.patch.multiple(
            screen.evaluator,
            SCHEMA="unset",
            SCHEMA_VERSION=0,
            LOG_N=0,
            PAIRS=0,
            ROUNDS=0,
            STORAGE_BYTES=0,
            DENSE_STORAGE_BYTES=0,
            REMAINING_SEQUENCE_STORAGE_BYTES=0,
            MAXIMUM_STORAGE_BUFFER_BYTES=0,
            SOURCE_PATHS=("base",),
        ):
            screen.configure()
            screen.configure()

            self.assertEqual(screen.evaluator.SCHEMA, screen.SCHEMA)
            self.assertEqual(screen.evaluator.SCHEMA_VERSION, screen.SCHEMA_VERSION)
            self.assertEqual(screen.evaluator.LOG_N, 25)
            self.assertEqual(screen.evaluator.PAIRS, 3)
            self.assertEqual(screen.evaluator.ROUNDS, 26)
            self.assertEqual(screen.evaluator.STORAGE_BYTES, 2_152_596_208)
            self.assertEqual(
                screen.evaluator.REMAINING_SEQUENCE_STORAGE_BYTES,
                5_112_560,
            )
            self.assertEqual(screen.evaluator.MAXIMUM_STORAGE_BUFFER_BYTES, 1 << 30)
            self.assertEqual(
                screen.evaluator.SOURCE_PATHS.count(
                    "scripts/metal_outer_remainder_screen.py"
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
