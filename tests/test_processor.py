import json
import pathlib
import re
import textwrap

import ollama
import pytest
import yaml
from jinja2.sandbox import SandboxedEnvironment
from pytest_mock import MockerFixture

from .factories import EmailFileFactory
from .factories import InboxEmailFactory
from .factories import MockEmail
from .factories import MockEmailFactory
from beanhub_inbox.data_types import ArchiveInboxAction
from beanhub_inbox.data_types import EmailFileMatchRule
from beanhub_inbox.data_types import ExtractConfig
from beanhub_inbox.data_types import ExtractImportAction
from beanhub_inbox.data_types import IgnoreInboxAction
from beanhub_inbox.data_types import ImportConfig
from beanhub_inbox.data_types import InboxAction
from beanhub_inbox.data_types import InboxActionType
from beanhub_inbox.data_types import InboxConfig
from beanhub_inbox.data_types import InboxDoc
from beanhub_inbox.data_types import InboxEmail
from beanhub_inbox.data_types import InboxMatch
from beanhub_inbox.data_types import InputConfig
from beanhub_inbox.data_types import SimpleFileMatch
from beanhub_inbox.data_types import StrContainsMatch
from beanhub_inbox.data_types import StrExactMatch
from beanhub_inbox.data_types import StrOneOfMatch
from beanhub_inbox.data_types import StrPrefixMatch
from beanhub_inbox.data_types import StrRegexMatch
from beanhub_inbox.data_types import StrSuffixMatch
from beanhub_inbox.processor import EmailFile
from beanhub_inbox.processor import extract_html_text
from beanhub_inbox.processor import extract_json_block
from beanhub_inbox.processor import extract_received_for_email
from beanhub_inbox.processor import match_email_file
from beanhub_inbox.processor import match_file
from beanhub_inbox.processor import match_inbox_email
from beanhub_inbox.processor import match_str
from beanhub_inbox.processor import process_imports
from beanhub_inbox.processor import process_inbox_email
from beanhub_inbox.processor import render_input_config_match


@pytest.fixture
def template_env() -> SandboxedEnvironment:
    return SandboxedEnvironment()


@pytest.mark.parametrize(
    "pattern, value, expected",
    [
        ("^Foo([0-9]+)", "Foo0", (True, {})),
        ("^Foo([0-9]+)", "Foo", (False, {})),
        ("^Foo([0-9]+)", "foo0", (False, {})),
        ("^Foo([0-9]+)", "", (False, {})),
        ("^Foo([0-9]+)", None, (False, {})),
        (
            r"(?P<first_name>\w+) (?P<last_name>\w+)",
            "Malcolm Reynolds",
            (True, {"first_name": "Malcolm", "last_name": "Reynolds"}),
        ),
        (StrPrefixMatch(prefix="Foo"), "Foo", (True, {})),
        (StrPrefixMatch(prefix="Foo"), "Foobar", (True, {})),
        (StrPrefixMatch(prefix="Foo"), "FooBAR", (True, {})),
        (StrPrefixMatch(prefix="Foo"), "xFooBAR", (False, {})),
        (StrPrefixMatch(prefix="Foo"), "", (False, {})),
        (StrPrefixMatch(prefix="Foo"), None, (False, {})),
        (StrSuffixMatch(suffix="Bar"), "Bar", (True, {})),
        (StrSuffixMatch(suffix="Bar"), "fooBar", (True, {})),
        (StrSuffixMatch(suffix="Bar"), "FooBar", (True, {})),
        (StrSuffixMatch(suffix="Bar"), "Foobar", (False, {})),
        (StrSuffixMatch(suffix="Bar"), "FooBarx", (False, {})),
        (StrSuffixMatch(suffix="Bar"), "", (False, {})),
        (StrSuffixMatch(suffix="Bar"), None, (False, {})),
        (StrContainsMatch(contains="Foo"), "Foo", (True, {})),
        (StrContainsMatch(contains="Foo"), "prefix-Foo", (True, {})),
        (StrContainsMatch(contains="Foo"), "Foo-suffix", (True, {})),
        (StrContainsMatch(contains="Foo"), "prefix-Foo-suffix", (True, {})),
        (StrContainsMatch(contains="Foo"), "prefix-Fo-suffix", (False, {})),
        (StrContainsMatch(contains="Foo"), "", (False, {})),
        (StrContainsMatch(contains="Foo"), None, (False, {})),
        (StrOneOfMatch(one_of=["Foo", "Bar"]), "Foo", (True, {})),
        (StrOneOfMatch(one_of=["Foo", "Bar"]), "Bar", (True, {})),
        (StrOneOfMatch(one_of=["Foo", "Bar"]), "Eggs", (False, {})),
        (StrOneOfMatch(one_of=["Foo", "Bar"]), "boo", (False, {})),
        (StrOneOfMatch(one_of=["Foo", "Bar"], ignore_case=True), "bar", (True, {})),
        (
            StrOneOfMatch(one_of=["Foo(.+)", "Bar(.+)"], regex=True),
            "FooBar",
            (True, {}),
        ),
        (StrOneOfMatch(one_of=["Foo(.+)", "Bar(.+)"], regex=True), "Foo", (False, {})),
        (StrOneOfMatch(one_of=["Foo(.+)", "Bar(.+)"], regex=True), "foo", (False, {})),
        (
            StrOneOfMatch(one_of=["Foo(.+)", "Bar(.+)"], regex=True, ignore_case=True),
            "foobar",
            (True, {}),
        ),
        (
            StrOneOfMatch(
                one_of=["Foo(.+)", "Bar(?P<val>.+)", "bar(.+)"],
                regex=True,
                ignore_case=True,
            ),
            "bar1234",
            (True, dict(val="1234")),
        ),
    ],
)
def test_match_str(
    pattern: SimpleFileMatch, value: str | None, expected: tuple[bool, dict]
):
    assert match_str(pattern, value) == expected


@pytest.mark.parametrize(
    "email_file, rule, extra_attrs, expected",
    [
        pytest.param(
            EmailFileFactory(subject="MOCK_SUBJECT"),
            EmailFileMatchRule(subject=StrExactMatch(equals="MOCK_SUBJECT")),
            None,
            (True, {}),
            id="match-subject",
        ),
        pytest.param(
            EmailFileFactory(subject="MOCK_SUBJECT"),
            EmailFileMatchRule(subject="MOCK_(?P<val>.+)"),
            None,
            (True, dict(val="SUBJECT")),
            id="match-subject-regex-capture",
        ),
        pytest.param(
            EmailFileFactory(subject="MOCK_SUBJECT"),
            EmailFileMatchRule(subject=StrExactMatch(equals="OTHER_SUBJECT")),
            None,
            (False, {}),
            id="not-match-subject",
        ),
        pytest.param(
            EmailFileFactory(filepath="/path/to/mock.eml"),
            EmailFileMatchRule(filepath=StrExactMatch(equals="/path/to/mock.eml")),
            None,
            (True, {}),
            id="match-filepath",
        ),
        pytest.param(
            EmailFileFactory(filepath="/path/to/mock.eml"),
            EmailFileMatchRule(filepath=StrExactMatch(equals="/path/to/other.eml")),
            None,
            (False, {}),
            id="not-match-filepath",
        ),
    ],
)
def test_match_email_file(
    email_file: EmailFile,
    rule: EmailFileMatchRule,
    extra_attrs: dict,
    expected: tuple[bool, dict],
):
    assert match_email_file(email_file, rule, extra_attrs=extra_attrs) == expected


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
    assert match_inbox_email(inbox_email=email, match=match) == expected


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
            template_env=template_env, inbox_email=email, inbox_configs=inbox_configs
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
    "inbox_doc, email_files, think_results, expected",
    [
        pytest.param(
            InboxDoc(
                inputs=[
                    InputConfig(match="*.eml"),
                ],
                imports=[
                    ImportConfig(
                        actions=[
                            ExtractImportAction(
                                extract=ExtractConfig(output_csv="output.csv")
                            )
                        ]
                    )
                ],
            ),
            {
                "mock.eml": MockEmailFactory(),
            },
            dict(
                valid=False,
            ),
            [
                ("StartProcessingEmail", lambda e: e.email_file.id == "mock"),
                ("MatchImportRule", lambda e: e.email_file.id == "mock"),
                ("StartExtractingColumn", lambda e: e.email_file.id == "mock"),
                ("StartThinking", lambda e: e.email_file.id == "mock"),
                ("UpdateThinking", lambda e: e.email_file.id == "mock"),
                ("FinishThinking", lambda e: e.email_file.id == "mock"),
                (
                    "FinishExtractingColumn",
                    lambda e: e.email_file.id == "mock"
                    and e.column.name == "valid"
                    and e.value == False,
                ),
                (
                    "FinishExtractingRow",
                    lambda e: e.email_file.id == "mock" and e.row == dict(valid=False),
                ),
            ],
            id="unconditional-match",
        ),
        pytest.param(
            InboxDoc(
                inputs=[
                    InputConfig(match="*.eml"),
                ],
                imports=[
                    ImportConfig(
                        match=EmailFileMatchRule(
                            subject=StrExactMatch(equals="MOCK_SUBJECT")
                        ),
                        actions=[
                            ExtractImportAction(
                                extract=ExtractConfig(output_csv="output.csv")
                            )
                        ],
                    )
                ],
            ),
            {
                "mock.eml": MockEmailFactory(subject="MOCK_SUBJECT"),
            },
            dict(
                valid=False,
            ),
            [
                ("StartProcessingEmail", lambda e: e.email_file.id == "mock"),
                ("MatchImportRule", lambda e: e.email_file.id == "mock"),
                ("StartExtractingColumn", lambda e: e.email_file.id == "mock"),
                ("StartThinking", lambda e: e.email_file.id == "mock"),
                ("UpdateThinking", lambda e: e.email_file.id == "mock"),
                ("FinishThinking", lambda e: e.email_file.id == "mock"),
                (
                    "FinishExtractingColumn",
                    lambda e: e.email_file.id == "mock"
                    and e.column.name == "valid"
                    and e.value == False,
                ),
                (
                    "FinishExtractingRow",
                    lambda e: e.email_file.id == "mock" and e.row == dict(valid=False),
                ),
            ],
            id="rule-match",
        ),
        pytest.param(
            InboxDoc(
                inputs=[
                    InputConfig(match="*.eml"),
                ],
                imports=[
                    ImportConfig(
                        match=EmailFileMatchRule(subject=StrExactMatch(equals="OTHER")),
                        actions=[
                            ExtractImportAction(
                                extract=ExtractConfig(output_csv="output.csv")
                            )
                        ],
                    )
                ],
            ),
            {
                "mock.eml": MockEmailFactory(subject="MOCK_SUBJECT"),
            },
            dict(
                valid=False,
            ),
            [
                ("StartProcessingEmail", lambda e: e.email_file.id == "mock"),
                ("NoMatch", lambda e: e.email_file.id == "mock"),
            ],
            id="not-match",
        ),
    ],
)
def test_process_imports(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
    inbox_doc: InboxDoc,
    email_files: dict[str, MockEmail],
    think_results: dict,
    expected: list,
):
    mock_chat = mocker.patch.object(ollama, "chat")

    def chat_side_effect(messages, **kwargs):
        msg = messages[0]
        match = re.search("with only one field `(.+?)`", msg.content)
        key = match.group(1)
        yield ollama.ChatResponse(
            message=ollama.Message(
                role="assistant", content=json.dumps({key: think_results[key]})
            )
        )

    mock_chat.side_effect = chat_side_effect

    for email_path, email_file in email_files.items():
        (tmp_path / email_path).write_text(str(email_file.make_msg()))

    events = list(
        process_imports(
            inbox_doc=inbox_doc,
            input_dir=tmp_path,
            llm_model="deepcoder",
            workdir_path=tmp_path,
        )
    )
    event_types = list(map(lambda event: event.__class__.__name__, events))
    assert event_types == list(map(lambda item: item[0], expected))
    for event, expected_item in zip(events, expected):
        _, validator = expected_item
        if validator is None:
            continue
        assert validator(event)


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
