#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from stratified_archive_pipeline import DEFAULT_TRAIN_OUTPUT_DIR, main_for_role


def main() -> None:
    main_for_role("train", Path(DEFAULT_TRAIN_OUTPUT_DIR))


if __name__ == "__main__":
    main()
