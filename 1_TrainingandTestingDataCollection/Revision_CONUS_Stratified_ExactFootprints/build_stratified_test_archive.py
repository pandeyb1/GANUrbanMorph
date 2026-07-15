#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from stratified_archive_pipeline import DEFAULT_TEST_OUTPUT_DIR, main_for_role


def main() -> None:
    main_for_role("test", Path(DEFAULT_TEST_OUTPUT_DIR))


if __name__ == "__main__":
    main()
