import contextlib
import csv
import dataclasses
import json
import logging
import os
import pathlib
import re
import typing
import uuid

import fast_mail_parser
import ollama
from fast_mail_parser import parse_email
from jinja2.sandbox import SandboxedEnvironment
from lxml import etree

from .data_types import ArchiveInboxAction
from .data_types import EmailMatchRule
from .data_types import ExtractImportAction
from .data_types import IgnoreImportAction
from .data_types import IgnoreInboxAction
from .data_types import ImportConfig
from .data_types import InboxAction
from .data_types import InboxActionType
from .data_types import InboxConfig
from .data_types import InboxDoc
from .data_types import InboxEmail
from .data_types import InboxMatch
from .data_types import InputConfig
from .data_types import OutputColumn
from .data_types import SimpleFileMatch
from .data_types import StrExactMatch
from .data_types import StrRegexMatch
from .llm import build_row_model
from .llm import DEFAULT_COLUMNS
from .llm import extract
from .llm import think
from .templates import make_environment
from .utils import GeneratorResult
from .utils import parse_tags

logger = logging.getLogger(__name__)
BEANHUB_INBOX_DOMAINS = frozenset(
    ["inbox.beanhub.io", "stage-inbox.beanhub.io", "dev-inbox.beanhub.io"]
)
DEFAULT_PROMPT_TEMPLATE = """\
# Instruction

Extract value from the following email content and output to an object with only one field `{{ column.name }}` in JSON.
Think step by step.

# JSON value definition

{{ column.description }}.
{%- if not column.required %}
Output null value if the value is not available.
{%- endif %}
{%- if column.pattern %}
Ensure the value match regular expression `{{ column.pattern }}`
{%- endif %}

# Email content

```
{{ content }}
```
"""


@dataclasses.dataclass(frozen=True)
class RenderedInputConfig:
    input_config: InputConfig


@dataclasses.dataclass(frozen=True)
class EmailFile:
    id: str
    filepath: str
    subject: str
    from_addresses: list[str]
    recipients: list[str]
    headers: dict[str, str]
    tags: list[str]


@dataclasses.dataclass(frozen=True)
class ProcessImportEvent:
    email_file: EmailFile


@dataclasses.dataclass(frozen=True)
class StartProcessingEmail(ProcessImportEvent):
    pass


@dataclasses.dataclass(frozen=True)
class NoMatch(ProcessImportEvent):
    pass


@dataclasses.dataclass(frozen=True)
class MatchImportRule(ProcessImportEvent):
    import_rule_index: int
    import_config: ImportConfig


@dataclasses.dataclass(frozen=True)
class IgnoreEmail(ProcessImportEvent):
    pass


@dataclasses.dataclass(frozen=True)
class CSVRowExists(ProcessImportEvent):
    pass


@dataclasses.dataclass(frozen=True)
class StartExtractingColumn(ProcessImportEvent):
    column: OutputColumn


@dataclasses.dataclass(frozen=True)
class StartThinking(ProcessImportEvent):
    column: OutputColumn
    prompt: str


@dataclasses.dataclass(frozen=True)
class UpdateThinking(ProcessImportEvent):
    column: OutputColumn
    piece: str


@dataclasses.dataclass(frozen=True)
class FinishThinking(ProcessImportEvent):
    column: OutputColumn
    thinking: str


@dataclasses.dataclass(frozen=True)
class FinishExtractingColumn(ProcessImportEvent):
    column: OutputColumn
    value: typing.Any


def match_inbox_email(email: InboxEmail, match: InboxMatch) -> bool:
    if match.tags is not None:
        if email.tags is None:
            return False
        email_tags = frozenset(email.tags)
        matching_tags = frozenset(match.tags)
        if matching_tags.intersection(email_tags) != matching_tags:
            return False
    if match.subject is not None:
        if re.match(match.subject, email.subject) is None:
            return False
    if match.headers is not None:
        for key, value in match.headers.items():
            if key not in email.headers:
                return False
            email_header_value = email.headers[key]
            if re.match(value, email_header_value) is None:
                return False
    if match.from_address is not None:
        if not any(
            re.match(match.from_address, address, flags=re.IGNORECASE)
            for address in email.from_addresses
        ):
            return False
    return True


def process_inbox_email(
    template_env: SandboxedEnvironment,
    email: InboxEmail,
    inbox_configs: list[InboxConfig],
) -> InboxAction | None:
    for config in inbox_configs:
        if config.match is None or match_inbox_email(email=email, match=config.match):
            if isinstance(config.action, ArchiveInboxAction):
                template_ctx = email.model_dump(mode="json")
                output_file = template_env.from_string(
                    config.action.output_file
                ).render(**template_ctx)
                return ArchiveInboxAction(
                    type=InboxActionType.archive, output_file=output_file
                )
            elif isinstance(config.action, IgnoreInboxAction):
                return config.action


def walk_dir_files(
    target_dir: pathlib.Path,
) -> typing.Generator[pathlib.Path, None, None]:
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            yield pathlib.Path(root) / file


def render_input_config_match(
    render_str: typing.Callable, match: SimpleFileMatch
) -> SimpleFileMatch:
    if isinstance(match, str):
        return render_str(match)
    elif isinstance(match, StrExactMatch):
        return StrExactMatch(equals=render_str(match.equals))
    elif isinstance(match, StrRegexMatch):
        return StrRegexMatch(regex=render_str(match.regex))
    else:
        raise ValueError(f"Unexpected match type {type(match)}")


def expand_input_loops(
    template_env: SandboxedEnvironment,
    inputs: list[InputConfig],
    omit_token: str,
) -> typing.Generator[RenderedInputConfig, None, None]:
    for input_config in inputs:
        if input_config.loop is not None:
            if not input_config.loop:
                raise ValueError("Loop content cannot be empty")
            loop = input_config.loop
        else:
            loop = [None]

        for values in loop:
            render_str = lambda value: template_env.from_string(value).render(
                **(dict(omit=omit_token) | (values if values is not None else {}))
            )
            rendered_match = render_input_config_match(
                render_str=render_str,
                match=input_config.match,
            )
            yield RenderedInputConfig(
                input_config=InputConfig(
                    match=rendered_match,
                ),
            )


def match_file(
    pattern: SimpleFileMatch, filepath: pathlib.Path | pathlib.PurePath
) -> bool:
    if isinstance(pattern, str):
        return filepath.match(pattern)
    if isinstance(pattern, StrRegexMatch):
        return re.match(pattern.regex, str(filepath)) is not None
    elif isinstance(pattern, StrExactMatch):
        return str(filepath) == pattern.equals
    else:
        raise ValueError(f"Unexpected file match type {type(pattern)}")


def extract_html_text(html: str) -> str:
    parser = etree.HTMLParser()
    tree = etree.fromstring(html, parser)
    # remove unwanted tags such as style
    etree.strip_elements(tree, "style", "script", with_tail=False)
    content = etree.tostring(tree, method="text", encoding="utf8").decode("utf8")
    return "\n".join(
        filter(lambda line: line, (line.strip() for line in content.splitlines()))
    )


def extract_received_for_email(header_value: str) -> str | None:
    match = re.match("from (.+) by (.+) for (.+);", header_value)
    if match is None:
        return None
    return match.group(3)


def split_emails(email_text: str) -> list[str]:
    return list(map(lambda item: item.strip(), email_text.split(",")))


def build_email_file(
    filepath: pathlib.Path, email: fast_mail_parser.PyMail
) -> EmailFile:
    received = email.headers.get("Received")
    tags = None
    if received is not None:
        email_address = extract_received_for_email(received)
        # TODO: make it possible for tags to work for email collected outside of BeanHub
        tags = parse_tags(email_address, domains=BEANHUB_INBOX_DOMAINS)
    from_addresses = split_emails(email.headers["From"])
    recipients = split_emails(email.headers["To"])
    return EmailFile(
        id=filepath.stem,
        filepath=str(filepath),
        subject=email.subject,
        headers=email.headers,
        from_addresses=from_addresses,
        recipients=recipients,
        tags=tags,
    )


def match_email_file(match_rule: EmailMatchRule, email_file: EmailFile) -> bool:
    # TODO: FIXME
    return True


def extract_json_block(text: str) -> typing.Generator[dict, None, None]:
    for match in re.finditer("```(json\n)?([^`]*)```", text, flags=re.IGNORECASE):
        try:
            yield json.loads(match.group(2))
        except ValueError:
            continue


def perform_extract_action(
    template_env: SandboxedEnvironment,
    email_file: EmailFile,
    parsed_email: fast_mail_parser.PyMail,
    action: ExtractImportAction,
    llm_model: str,
    progress_output_folder: pathlib.Path | None = None,
    think_progress_factory: typing.Callable[
        [], typing.ContextManager[typing.Callable[[str], None]]
    ]
    | None = None,
) -> typing.Generator[ProcessImportEvent, None, None]:
    output_csv = pathlib.Path(action.extract.output_csv)
    if output_csv.exists():
        # TODO: extract this
        with output_csv.open("rt") as fo:
            reader = csv.DictReader(fo)
            if "id" not in reader.fieldnames:
                raise ValueError(
                    f"No id column found in the existing output csv file at {output_csv}"
                )
            for index, row in enumerate(reader):
                email_id = row["id"]
                if email_id == email_file.id:
                    logger.info(
                        "Found email %s row %s in output CSV file %s, skip",
                        email_file.id,
                        index + 1,
                        output_csv,
                    )
                    yield CSVRowExists(email_file=email_file)
                    return

    if parsed_email.text_html:
        text = extract_html_text(parsed_email.text_html[0])
    elif parsed_email.text_plain:
        text = parsed_email.text_plain[0]
    else:
        raise ValueError(
            f"The email {email_file.id} has no no content available for processing"
        )

    # TODO: get template from action or default value
    template = DEFAULT_PROMPT_TEMPLATE
    if action.extract.template is not None:
        template = action.extract.template

    row = {}
    columns = DEFAULT_COLUMNS
    # TODO: we can run all columns at once to speed up if we need to
    for column in columns:
        logger.info(
            'Extracting "%s" (%s type) column value ...',
            column.name,
            column.type.value,
        )
        yield StartExtractingColumn(email_file=email_file, column=column)
        response_model_cls = build_row_model(
            output_columns=[column],
        )

        prompt = template_env.from_string(template).render(
            json_schema=response_model_cls.model_json_schema(),
            content=text,
            column=column,
        )
        logger.debug(
            "Thinking about extracting data for email %s with prompt:\n%s",
            email_file.id,
            prompt,
        )
        messages = [ollama.Message(role="user", content=prompt)]
        yield StartThinking(email_file=email_file, column=column, prompt=prompt)
        think_generator = GeneratorResult(
            think(model=llm_model, messages=messages, stream=True)
        )
        for part in think_generator:
            yield UpdateThinking(
                email_file=email_file, column=column, piece=part.message.content
            )
        yield FinishThinking(
            email_file=email_file, column=column, thinking=think_generator.value.content
        )

        result = extract(
            model=llm_model,
            messages=messages,
            response_model_cls=response_model_cls,
        )

        json_obj = result.model_dump(mode="json")
        logger.info('Extracted "%s" value %r', column.name, json_obj.get(column.name))

        row.update(json_obj)
        if column.name == "valid" and not json_obj["valid"]:
            # TODO: find a way to make it possible to define which column is the "valid"
            logger.info(
                "Email %s is not a valid one, skip all other columns",
                email_file.id,
            )
            break

    logger.info(
        "Write email %s row data %s to CSV file %s",
        email_file.id,
        row,
        output_csv,
    )
    if output_csv.exists():
        # TODO: lock file
        with output_csv.open("at+", newline="") as fo:
            writer = csv.DictWriter(
                fo, fieldnames=["id", *(column.name for column in columns)]
            )
            # TODO: sort by id column?
            writer.writerow(dict(id=email_file.id) | row)
    else:
        # TODO: lock file
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("wt", newline="") as fo:
            writer = csv.DictWriter(
                fo, fieldnames=["id", *(column.name for column in columns)]
            )
            writer.writeheader()
            writer.writerow(dict(id=email_file.id) | row)


def process_imports(
    inbox_doc: InboxDoc,
    input_dir: pathlib.Path,
    llm_model: str,
    progress_output_folder: pathlib.Path | None = None,
    think_progress_factory: typing.Callable[
        [], typing.ContextManager[typing.Callable[[str], None]]
    ]
    | None = None,
) -> typing.Generator[ProcessImportEvent, None, None]:
    template_env = make_environment()
    omit_token = uuid.uuid4().hex

    expanded_input_configs = list(
        expand_input_loops(
            template_env=template_env, inputs=inbox_doc.inputs, omit_token=omit_token
        ),
    )

    # sort filepaths for deterministic behavior across platforms
    # TODO: this might be a bit slow if the input dir has a tons of files...
    filepaths = sorted(walk_dir_files(input_dir))
    for filepath in filepaths:
        matched_input_config = None
        for input_config_index, rendered_input_config in enumerate(
            expanded_input_configs
        ):
            input_config = rendered_input_config.input_config
            if match_file(input_config.match, filepath):
                matched_input_config = input_config
                logger.info("Matched input config %s", input_config_index)
                break
        if matched_input_config is None:
            # Not interested in this file, skip
            continue

        rel_filepath = filepath.relative_to(input_dir)
        parsed_email = parse_email(filepath.read_bytes())
        email_file = build_email_file(filepath=rel_filepath, email=parsed_email)
        yield StartProcessingEmail(email_file=email_file)

        matched_import_config = None
        matched_import_config_index = None
        for index, import_config in enumerate(inbox_doc.imports):
            if import_config.match is None or match_email_file(
                import_config.match, email_file
            ):
                matched_import_config = import_config
                matched_import_config_index = index
                break
        if matched_import_config is None:
            logger.info(
                "No import rule match for email %s at %s, skip",
                email_file.id,
                email_file.filepath,
            )
            yield NoMatch(email_file=email_file)
            continue

        logger.info(
            "Match email %s at %s with import rule %s",
            email_file.id,
            email_file.filepath,
            matched_import_config.name
            if matched_import_config.name is not None
            else matched_import_config_index,
        )
        yield MatchImportRule(
            email_file=email_file,
            import_rule_index=matched_import_config_index,
            import_config=matched_import_config,
        )
        for action in matched_import_config.actions:
            if isinstance(action, ExtractImportAction):
                yield from perform_extract_action(
                    template_env=template_env,
                    email_file=email_file,
                    parsed_email=parsed_email,
                    action=action,
                    llm_model=llm_model,
                    progress_output_folder=progress_output_folder,
                    think_progress_factory=think_progress_factory,
                )
            elif isinstance(action, IgnoreImportAction):
                logger.info("Ignore email %s", email_file.id)
                yield IgnoreEmail(email_file=email_file)
            else:
                raise ValueError(f"Unexpected action type {type(action)}")
