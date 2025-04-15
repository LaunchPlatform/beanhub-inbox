import pathlib
import textwrap

import pytest
import yaml
from jinja2.sandbox import SandboxedEnvironment

from .factories import InboxEmailFactory
from beanhub_inbox.data_types import ArchiveInboxAction
from beanhub_inbox.data_types import IgnoreInboxAction
from beanhub_inbox.data_types import InboxAction
from beanhub_inbox.data_types import InboxActionType
from beanhub_inbox.data_types import InboxConfig
from beanhub_inbox.data_types import InboxDoc
from beanhub_inbox.data_types import InboxEmail
from beanhub_inbox.data_types import InboxMatch
from beanhub_inbox.data_types import SimpleFileMatch
from beanhub_inbox.data_types import StrExactMatch
from beanhub_inbox.data_types import StrRegexMatch
from beanhub_inbox.processor import extract_html_text
from beanhub_inbox.processor import extract_json_block
from beanhub_inbox.processor import extract_received_for_email
from beanhub_inbox.processor import match_file
from beanhub_inbox.processor import match_inbox_email
from beanhub_inbox.processor import process_imports
from beanhub_inbox.processor import process_inbox_email
from beanhub_inbox.processor import render_input_config_match


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
        pytest.param(
            InboxEmailFactory(
                id="mock-id",
                subject="foo",
            ),
            [
                InboxConfig(
                    match=InboxMatch(subject="eggs"),
                    action=ArchiveInboxAction(
                        output_file="path/to/other/{{ id }}.eml",
                    ),
                ),
                InboxConfig(
                    match=InboxMatch(subject="foo"),
                    action=ArchiveInboxAction(
                        output_file="path/to/{{ id }}.eml",
                    ),
                ),
            ],
            ArchiveInboxAction(output_file="path/to/mock-id.eml"),
            id="order",
        ),
        pytest.param(
            InboxEmailFactory(
                id="mock-id",
                subject="foo",
            ),
            [
                InboxConfig(
                    match=InboxMatch(subject="eggs"),
                    action=ArchiveInboxAction(
                        output_file="path/to/other/{{ id }}.eml",
                    ),
                ),
                InboxConfig(
                    action=ArchiveInboxAction(
                        output_file="path/to/{{ id }}.eml",
                    ),
                ),
            ],
            ArchiveInboxAction(output_file="path/to/mock-id.eml"),
            id="match-none",
        ),
        pytest.param(
            InboxEmailFactory(
                id="mock-id",
                message_id="mock-msg-id",
                subject="foo",
                headers=dict(key="value"),
            ),
            [
                InboxConfig(
                    match=InboxMatch(subject="foo"),
                    action=ArchiveInboxAction(
                        output_file="{{ message_id }}/{{ subject }}/{{ headers['key'] }}.eml",
                    ),
                ),
            ],
            ArchiveInboxAction(output_file="mock-msg-id/foo/value.eml"),
            id="render",
        ),
        pytest.param(
            InboxEmailFactory(
                subject="spam",
            ),
            [
                InboxConfig(
                    match=InboxMatch(subject="eggs"),
                    action=ArchiveInboxAction(
                        output_file="path/to/other/{{ id }}.eml",
                    ),
                ),
                InboxConfig(
                    match=InboxMatch(subject="spam"),
                    action=IgnoreInboxAction(type=InboxActionType.ignore),
                ),
                InboxConfig(
                    action=ArchiveInboxAction(
                        output_file="path/to/{{ id }}.eml",
                    ),
                ),
            ],
            IgnoreInboxAction(type=InboxActionType.ignore),
            id="ignore",
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


@pytest.mark.parametrize(
    "pattern, path, expected",
    [
        ("/path/to/*/foo*.csv", "/path/to/bar/foo.csv", True),
        ("/path/to/*/foo*.csv", "/path/to/bar/foo-1234.csv", True),
        ("/path/to/*/foo*.csv", "/path/to/eggs/foo-1234.csv", True),
        ("/path/to/*/foo*.csv", "/path/to/eggs/foo.csv", True),
        ("/path/to/*/foo*.csv", "/path/from/eggs/foo.csv", False),
        ("/path/to/*/foo*.csv", "foo.csv", False),
        (StrRegexMatch(regex=r"^/path/to/([0-9]+)"), "/path/to/0", True),
        (StrRegexMatch(regex=r"^/path/to/([0-9]+)"), "/path/to/0123", True),
        (StrRegexMatch(regex=r"^/path/to/([0-9]+)"), "/path/to/a0123", False),
        (StrExactMatch(equals="foo.csv"), "foo.csv", True),
        (StrExactMatch(equals="foo.csv"), "xfoo.csv", False),
    ],
)
def test_match_file(pattern: SimpleFileMatch, path: str, expected: bool):
    assert match_file(pattern, pathlib.PurePosixPath(path)) == expected


@pytest.mark.parametrize(
    "match, values, expected",
    [
        (
            "inbox-data/connect/{{ foo }}",
            dict(foo="bar.csv"),
            "inbox-data/connect/bar.csv",
        ),
        (
            "inbox-data/connect/eggs.csv",
            dict(foo="bar.csv"),
            "inbox-data/connect/eggs.csv",
        ),
        (
            StrExactMatch(equals="inbox-data/connect/{{ foo }}"),
            dict(foo="bar.csv"),
            StrExactMatch(equals="inbox-data/connect/bar.csv"),
        ),
        (
            StrRegexMatch(regex="inbox-data/connect/{{ foo }}"),
            dict(foo="bar.csv"),
            StrRegexMatch(regex="inbox-data/connect/bar.csv"),
        ),
    ],
)
def test_render_input_config_match(
    template_env: SandboxedEnvironment,
    match: SimpleFileMatch,
    values: dict,
    expected: SimpleFileMatch,
):
    render_str = lambda value: template_env.from_string(value).render(values)
    assert render_input_config_match(render_str=render_str, match=match) == expected


@pytest.mark.parametrize(
    "filename",
    [
        "sample.yaml",
    ],
)
def test_parse_yaml(fixtures_folder: pathlib.Path, filename: str):
    yaml_file = fixtures_folder / filename
    with yaml_file.open("rb") as fo:
        payload = yaml.safe_load(fo)
    doc = InboxDoc.model_validate(payload)
    assert doc


@pytest.mark.parametrize(
    "header_value, expected",
    [
        (
            (
                "from mail-4317.protonmail.ch (mail-4317.protonmail.ch [185.70.43.17])"
                " by inbound-smtp.us-west-2.amazonaws.com with SMTP id n9dtgvp7tq2eoggselt8sr53kd1eglmau0kbn181"
                " for fangpenlin+mybook+tag0+tag1@dev-inbox.beanhub.io;"
                " Sun, 13 Apr 2025 22:48:38 +0000 (UTC)"
            ),
            "fangpenlin+mybook+tag0+tag1@dev-inbox.beanhub.io",
        ),
        (
            "not-relative stuff",
            None,
        ),
        ("", None),
    ],
)
def test_extract_received_for_email(header_value: str, expected: str | None):
    assert extract_received_for_email(header_value) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param(
            textwrap.dedent("""\
            ```json
            {}
            ```
            """),
            [{}],
            id="json-lang-emtpy-dict",
        ),
        pytest.param(
            textwrap.dedent("""\
            ```
            {}
            ```
            """),
            [{}],
            id="no-lang-emtpy-dict",
        ),
        pytest.param(
            textwrap.dedent("""\
            ```json
            {"key": "value"}
            ```
            """),
            [{"key": "value"}],
            id="json-lang-dict",
        ),
        pytest.param(
            textwrap.dedent("""\
            # Section 1
            
            ```json
            {"key0": "value0"}
            ```
            
            # Section 2
            
            ```
            {"key1": "value1"}
            ```
            
            # Section 3
            
            ```{}```
            
            """),
            [{"key0": "value0"}, {"key1": "value1"}, {}],
            id="multiple",
        ),
    ],
)
def test_extract_json_block(text: str, expected: list[dict]):
    assert list(extract_json_block(text)) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("basic", []),
    ],
)
def test_process_imports(
    fixtures_folder: pathlib.Path,
    folder: str,
    expected: list,
):
    folder_path = fixtures_folder / "processor" / folder
    with open(folder_path / "inbox.yaml", "rt") as fo:
        payload = yaml.safe_load(fo)
    doc = InboxDoc.model_validate(payload)
    assert list(process_imports(inbox_doc=doc, input_dir=folder_path)) == expected


@pytest.mark.parametrize(
    "html, expected",
    [
        pytest.param(
            textwrap.dedent("""\
            <div style="font-family: Arial, sans-serif; font-size: 14px;"><br></div><div class="protonmail_quote">
                    ------- Forwarded Message -------<br>
                    From: DigitalOcean Support &lt;support@digitalocean.com&gt;<br>
                    Date: On Sunday, September 1st, 2024 at 12:17 AM<br>
                    Subject: [DigitalOcean] Your 2024-08 invoice is available<br>
                    To: Fang-Pen Lin &lt;fangpen@launchplatform.com&gt;<br>
                    <br>
                    <blockquote class="protonmail_quote" type="cite">
                        <div>Usage charges for 2024-08</div>
                    </blockquote>
            </div>
            """),
            textwrap.dedent("""\
            ------- Forwarded Message -------
            From: DigitalOcean Support <support@digitalocean.com>
            Date: On Sunday, September 1st, 2024 at 12:17 AM
            Subject: [DigitalOcean] Your 2024-08 invoice is available
            To: Fang-Pen Lin <fangpen@launchplatform.com>
            Usage charges for 2024-08"""),
            id="basic",
        ),
        pytest.param(
            textwrap.dedent("""\
            first line
            <style>
            h1 { color: red; }
            </style>
            second line
            <div>
                third line
            </div>
            """),
            "first line\nsecond line\nthird line",
            id="style",
        ),
        pytest.param(
            textwrap.dedent("""\
            first line
            <script>
            console.log('hi')
            </script>
            second line
            <div>
                third line
            </div>
            """),
            "first line\nsecond line\nthird line",
            id="script",
        ),
    ],
)
def test_extract_html_text(html: str, expected: str):
    assert extract_html_text(html) == expected
