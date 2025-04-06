import pytest

from beanhub_inbox.data_types import InboxEmail
from beanhub_inbox.data_types import InboxMatch
from beanhub_inbox.processor import match_inbox_email


@pytest.mark.parametrize(
    "email, match, expected",
    [
        (
            InboxEmail(
                id="MOCK_ID",
                message_id="MOCK_MSG_ID",
                headers={"Mock-Header": "Mock-Value"},
                subject="Mock subject",
                from_address=["hello@fangpenlin.com"],
                recipients=["my-repo@inbox.beanhub.io"],
            ),
            InboxMatch(subject="Mock .*"),
            True,
        )
    ],
)
def test_match_inbox_email(email: InboxEmail, match: InboxMatch, expected: bool):
    assert match_inbox_email(email=email, match=match) == expected
