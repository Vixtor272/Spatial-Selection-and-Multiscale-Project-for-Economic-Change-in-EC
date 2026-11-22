"""The results depend on one exact version of the engine of Kemp et al.
These tests fail if UrbanTraitSelection is missing, moved to another commit,
or if either of the two files we import has been edited."""
import subprocess

import pytest

from ecuador_price import paths

ENGINE_REPO = paths.ENGINE_DIR.parent
ENGINE_COMMIT = "045a994b2184aa4227d33aea11d6777745cfb6ab"
# git blob hashes of the two files at that commit
FILE_HASHES = {
    "runPrice.py": "1ba5a4dbf8a378a9361ee6b30bf6191820574abc",
    "procHelpers.py": "da9686b51a0c7decbde5dcb2fd9a39f480d5af72",
}


def git(*args):
    return subprocess.run(["git", "-C", str(ENGINE_REPO), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_engine_is_checked_out():
    assert (paths.ENGINE_DIR / "runPrice.py").exists(), \
        "engine missing: run `git submodule update --init`"


def test_engine_commit():
    assert git("rev-parse", "HEAD") == ENGINE_COMMIT


@pytest.mark.parametrize("name", sorted(FILE_HASHES))
def test_engine_files_are_unmodified(name):
    assert git("hash-object", f"calculation_scripts/{name}") == FILE_HASHES[name]
