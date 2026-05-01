from setuptools import find_packages


EXPECTED_SUBPACKAGES = {
    "engram_peft",
    "engram_peft.utils",
}


def test_all_subpackages_are_discovered() -> None:
    pkgs = find_packages(where="src")
    for subpkg in EXPECTED_SUBPACKAGES:
        assert subpkg in pkgs, f"{subpkg} not found in setuptools discovery ({pkgs})"
