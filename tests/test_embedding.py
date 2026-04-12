import pytest
from engram_peft.embedding import dummy_embedding


def test_dummy_embedding() -> None:
    dummy_embedding()
    assert True
