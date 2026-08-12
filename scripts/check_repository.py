#!/usr/bin/env python3
"""Check tracked files for credentials, private paths, caches, and size limits."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
FORBIDDEN_NAMES = {".env", "__pycache__", ".pytest_cache", "llm_audit"}
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".example",
    ".gitignore",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = {
    "OpenAI-style key": re.compile("s" + r"k-[A-Za-z0-9_-]{20,}"),
    "Hugging Face token": re.compile("h" + r"f_[A-Za-z0-9_-]{20,}"),
}
PRIVATE_PATH_PATTERNS = ("/Users/", "/home/")


def candidate_files():
    """Yield tracked files, falling back to a clean filesystem walk."""

    try:
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        for path in ROOT.rglob("*"):
            if ".git" not in path.parts and path.is_file():
                yield path
        return

    for relative in tracked.split(b"\0"):
        if not relative:
            continue
        path = ROOT / relative.decode("utf-8")
        if path.is_file():
            yield path


def main() -> None:
    problems: list[str] = []
    for path in candidate_files():
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_NAMES for part in relative.parts):
            problems.append(f"forbidden generated or secret path: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            problems.append(f"file exceeds 10 MiB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".env.example",
            ".gitignore",
            "LICENSE",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"possible {label}: {relative}")
        if relative != Path("scripts/check_repository.py"):
            for prefix in PRIVATE_PATH_PATTERNS:
                if prefix in text:
                    problems.append(f"machine-specific absolute path in {relative}")

    if problems:
        raise SystemExit("Repository check failed:\n- " + "\n- ".join(sorted(problems)))
    print("Repository check passed: no credentials, private paths, forbidden caches, or files over 10 MiB.")


if __name__ == "__main__":
    main()
