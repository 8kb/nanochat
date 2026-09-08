"""Shared fixtures for datacore's test suite."""
import pytest

from datacore import CharTokenizer

CHARS = " abcdefghijklmnopqrstuvwxyz.,!?'\n0123456789"


@pytest.fixture
def char_tokenizer():
    return CharTokenizer(CHARS)
