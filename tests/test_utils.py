import typing

import pytest

from beanhub_inbox.utils import parse_tags


@pytest.mark.parametrize(
    "email, domains, expected",
    [
        ("fangpen@launchplatform.com", ["inbox.beanhub.io"], None),
        ("fangpen+mybook@inbox.beanhub.io", ["inbox.beanhub.io"], None),
        ("fangpen@inbox.beanhub.io", ["inbox.beanhub.io"], None),
        (
            "fangpen+mybook+tag0+tag1@inbox.beanhub.io",
            ["inbox.beanhub.io"],
            ["tag0", "tag1"],
        ),
        ("fangpen+mybook+tag0+tag1@inbox.beanhub.io", [], None),
        (
            "fangpen+mybook+tag0+tag1@domain1.com",
            ["domain1.com", "domain2.com"],
            ["tag0", "tag1"],
        ),
        ("fangpen+mybook+tag0+tag1@domain3.com", ["domain1.com", "domain2.com"], None),
        (
            "fangpen+mybook+tag0+tag1@InBoX.BeAnHuB.Io",
            ["inbox.beanhub.io"],
            ["tag0", "tag1"],
        ),
        ("fangpen+mybook+tag0@inbox.beanhub.io", ["inbox.beanhub.io"], ["tag0"]),
        (
            "fangpen+mybook+tag0+tag1+tag2@inbox.beanhub.io",
            ["inbox.beanhub.io"],
            ["tag0", "tag1", "tag2"],
        ),
        (
            "fangpen+mybook+tag-0+tag_1@inbox.beanhub.io",
            ["inbox.beanhub.io"],
            ["tag-0", "tag_1"],
        ),
        ("fangpen+mybook++tag1@inbox.beanhub.io", ["inbox.beanhub.io"], ["", "tag1"]),
        (
            "fangpen+mybook+++tag1@inbox.beanhub.io",
            ["inbox.beanhub.io"],
            ["", "", "tag1"],
        ),
    ],
)
def test_parse_tags(
    email: str, domains: typing.Sequence[str], expected: list[str] | None
):
    assert parse_tags(email_address=email, domains=domains) == expected
