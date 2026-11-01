"""Guards against the bundled suites becoming unreachable on a real install.

What broke
----------
`suites/` used to sit *beside* the package, located as
``Path(__file__).parent.parent / "suites"``. That resolves correctly from a
source checkout and from `pip install -e .`, so every local test passed. On a
non-editable install -- `pip install .`, or the `pip install git+...` a stranger
runs first -- the package is copied into site-packages and the same expression
resolves to ``site-packages/suites``, which does not exist. `package-data` was
empty, so the files were never shipped either way.

Result: a fresh install could not load a single bundled suite, and nothing in
the test suite noticed, because the tests only ever ran against a checkout.

These tests are cheap and they fail loudly if the layout regresses. The
packaging half (does the wheel actually contain the data?) cannot be asserted
from inside the test suite; it was verified by building a wheel and installing
it into a clean directory. See the comment on `_SUITES_DIR` in suites.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import quantcheck
from quantcheck.suites import BUNDLED_SUITES, list_bundled_suites, load_suite
from quantcheck.suites import _SUITES_DIR


def test_suites_live_inside_the_package():
    """The data directory must be a child of the package, not a sibling.

    This is the assertion that would have caught the original bug. A sibling
    path works from a checkout and silently breaks on install.
    """
    pkg_dir = Path(quantcheck.__file__).resolve().parent
    assert _SUITES_DIR == pkg_dir / "suites", (
        f"_SUITES_DIR is {_SUITES_DIR}, expected {pkg_dir / 'suites'}. A path "
        "outside the package resolves to site-packages/ on a non-editable "
        "install, where no data is shipped."
    )
    assert _SUITES_DIR.is_dir()


def test_declared_package_data_covers_the_suite_files():
    """Every file the loader needs must match a package-data glob.

    Read from pyproject.toml rather than hard-coded, so the test tracks the
    declaration instead of duplicating it.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        pytest.skip("tomllib requires Python 3.11+")

    root = Path(quantcheck.__file__).resolve().parent.parent
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("not running from a source checkout")

    cfg = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    globs = cfg["tool"]["setuptools"]["package-data"]["quantcheck"]
    assert globs, "package-data for 'quantcheck' is empty; suites will not ship"

    pkg_dir = Path(quantcheck.__file__).resolve().parent
    for f in sorted(_SUITES_DIR.iterdir()):
        rel = f.relative_to(pkg_dir).as_posix()
        assert any(Path(rel).match(g) for g in globs), (
            f"{rel} matches none of {globs}, so it would not be installed"
        )


@pytest.mark.parametrize("name", sorted(BUNDLED_SUITES))
def test_every_bundled_suite_loads(name):
    """Each advertised suite must actually load, with items in it."""
    suite = load_suite(name)
    assert len(suite) > 0
    assert suite.items[0].prompt
    assert suite.items[0].answer


def test_listed_suites_match_the_advertised_set():
    """`--list-suites` must not advertise a suite the loader cannot find."""
    assert set(list_bundled_suites()) == set(BUNDLED_SUITES)
