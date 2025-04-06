import pathlib

import pytest
from pytest_factoryboy import register

from .factories import InboxEmailFactory

register(InboxEmailFactory)

TEST_PACKAGE_FOLDER = pathlib.Path(__file__).parent
FIXTURE_FOLDER = TEST_PACKAGE_FOLDER / "fixtures"


@pytest.fixture
def fixtures_folder() -> pathlib.Path:
    return FIXTURE_FOLDER
