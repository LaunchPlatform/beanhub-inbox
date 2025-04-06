import pytest

from .factories import InboxEmailFactory
from beanhub_inbox.data_types import InboxEmail
from beanhub_inbox.data_types import InboxMatch
from beanhub_inbox.processor import match_inbox_email


@pytest.mark.parametrize(
    "email, match, expected",
    [
        (
            InboxEmailFactory(
                subject="Mock subject",
            ),
            InboxMatch(subject="Mock .*"),
            True,
        ),
        (
            InboxEmailFactory(
                subject="Other subject",
            ),
            InboxMatch(subject="Mock .*"),
            False,
        ),
        (
            InboxEmailFactory(
                tags=["a", "b", "c"],
            ),
            InboxMatch(tags=["a"]),
            True,
        ),
        (
            InboxEmailFactory(
                tags=["a", "b", "c"],
            ),
            InboxMatch(tags=["b"]),
            True,
        ),
        (
            InboxEmailFactory(
                tags=["a", "b", "c"],
            ),
            InboxMatch(tags=["c"]),
            True,
        ),
        (
            InboxEmailFactory(
                tags=["a", "b", "c"],
            ),
            InboxMatch(tags=["a", "b"]),
            True,
        ),
        (
            InboxEmailFactory(
                tags=["a", "b", "c"],
            ),
            InboxMatch(tags=["a", "c"]),
            True,
        ),
        (
            InboxEmailFactory(
                tags=["a", "b", "c"],
            ),
            InboxMatch(tags=["b", "c"]),
            True,
        ),
        (
            InboxEmailFactory(
                tags=["a", "b", "c"],
            ),
            InboxMatch(tags=["a", "b", "c"]),
            True,
        ),
        (
            InboxEmailFactory(
                tags=["a", "b", "c"],
            ),
            InboxMatch(tags=["a", "b", "c", "d"]),
            False,
        ),
        (
            InboxEmailFactory(
                tags=["a", "b", "c"],
            ),
            InboxMatch(tags=["a", "other"]),
            False,
        ),
    ],
)
def test_match_inbox_email(email: InboxEmail, match: InboxMatch, expected: bool):
    assert match_inbox_email(email=email, match=match) == expected
