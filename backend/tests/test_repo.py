"""Repository hygiene.

A bare `data/` in .gitignore once matched the source package `app/data/` and
hid corpus.py, augment.py and dataset.py from git. Every test still passed,
because tests import from disk, not from the repository; a clean checkout
would simply not have had them. This asks git directly.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_no_source_file_is_gitignored():
    sources = [str(p) for p in (BACKEND / "app").rglob("*.py") if "__pycache__" not in p.parts]
    sources += [str(p) for p in (BACKEND / "tests").glob("*.py")]
    result = subprocess.run(["git", "check-ignore", *sources], cwd=BACKEND,
                            capture_output=True, text=True)
    if result.returncode == 128:
        pytest.skip("not inside a git checkout")
    assert result.stdout.strip() == "", f"ignored source files:\n{result.stdout}"
