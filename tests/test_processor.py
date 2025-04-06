import pytest
from jinja2.sandbox import SandboxedEnvironment

from .factories import InboxEmailFactory
from beanhub_inbox.data_types import ArchiveInboxAction
from beanhub_inbox.data_types import InboxAction
from beanhub_inbox.data_types import InboxConfig
from beanhub_inbox.data_types import InboxEmail
from beanhub_inbox.data_types import InboxMatch
from beanhub_inbox.processor import match_inbox_email
from beanhub_inbox.processor import process_inbox_email


@pytest.fixture
def template_env() -> SandboxedEnvironment:
    return SandboxedEnvironment()


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
        (
            InboxEmailFactory(headers=dict(key="value")),
            InboxMatch(headers=dict(key="value")),
            True,
        ),
        (
            InboxEmailFactory(headers=dict(key="value")),
            InboxMatch(headers=dict(key="val.+")),
            True,
        ),
        (
            InboxEmailFactory(headers=dict(key="value")),
            InboxMatch(headers=dict(key="value", eggs="spam")),
            False,
        ),
        (
            InboxEmailFactory(headers=dict(key="value")),
            InboxMatch(headers=dict(key="other")),
            False,
        ),
        (
            InboxEmailFactory(
                from_addresses=["fangpen@launchplatform.com", "hello@fangpenlin.com"]
            ),
            InboxMatch(from_address="fangpen@launchplatform.com"),
            True,
        ),
        (
            InboxEmailFactory(
                from_addresses=["fangpen@launchplatform.com", "hello@fangpenlin.com"]
            ),
            InboxMatch(from_address="hello@fangpenlin.com"),
            True,
        ),
        (
            InboxEmailFactory(
                from_addresses=["fangpen@launchplatform.com", "hello@fangpenlin.com"]
            ),
            InboxMatch(from_address="fangpen@.+"),
            True,
        ),
        (
            InboxEmailFactory(
                from_addresses=["fangpen@launchplatform.com", "hello@fangpenlin.com"]
            ),
            InboxMatch(from_address=".*fangpen.*"),
            True,
        ),
        (
            InboxEmailFactory(
                from_addresses=["fangpen@launchplatform.com", "hello@fangpenlin.com"]
            ),
            InboxMatch(from_address="other"),
            False,
        ),
    ],
)
def test_match_inbox_email(email: InboxEmail, match: InboxMatch, expected: bool):
    assert match_inbox_email(email=email, match=match) == expected


@pytest.mark.parametrize(
    "email, inbox_configs, expected",
    [
        (
            InboxEmailFactory(
                id="mock-id",
                subject="foo",
            ),
            [
                InboxConfig(
                    match=InboxMatch(subject="foo"),
                    action=ArchiveInboxAction(
                        output_file="path/to/{{ id }}",
                    ),
                )
            ],
            ArchiveInboxAction(output_file="path/to/mock-id"),
        ),
    ],
)
def test_process_inbox_email(
    template_env: SandboxedEnvironment,
    email: InboxEmail,
    inbox_configs: list[InboxConfig],
    expected: InboxAction | None,
):
    assert (
        process_inbox_email(
            template_env=template_env, email=email, inbox_configs=inbox_configs
        )
        == expected
    )
