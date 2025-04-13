import dataclasses
import logging
import os
import pathlib
import re
import textwrap
import typing
import uuid

from fast_mail_parser import parse_email
from jinja2.sandbox import SandboxedEnvironment
from lxml import etree

from .data_types import ArchiveInboxAction
from .data_types import IgnoreInboxAction
from .data_types import InboxAction
from .data_types import InboxActionType
from .data_types import InboxConfig
from .data_types import InboxDoc
from .data_types import InboxEmail
from .data_types import InboxMatch
from .data_types import InputConfig
from .data_types import SimpleFileMatch
from .data_types import StrExactMatch
from .data_types import StrRegexMatch
from .llm import build_response_model
from .llm import DEFAULT_COLUMNS
from .llm import extract
from .llm import think
from .templates import make_environment


@dataclasses.dataclass(frozen=True)
class RenderedInputConfig:
    input_config: InputConfig


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
    content = etree.tostring(tree, method="text").decode("utf8")
    return "\n".join(
        filter(lambda line: line, (line.strip() for line in content.splitlines()))
    )


def process_imports(
    inbox_doc: InboxDoc,
    input_dir: pathlib.Path,
):
    logger = logging.getLogger(__name__)
    template_env = make_environment()
    omit_token = uuid.uuid4().hex

    expanded_input_configs = list(
        expand_input_loops(
            template_env=template_env, inputs=inbox_doc.inputs, omit_token=omit_token
        ),
    )

    # sort filepaths for deterministic behavior across platforms
    filepaths = sorted(walk_dir_files(input_dir))
    for filepath in filepaths:
        for rendered_input_config in expanded_input_configs:
            input_config = rendered_input_config.input_config
            if not match_file(input_config.match, filepath):
                continue
            rel_filepath = filepath.relative_to(input_dir)
            email = parse_email(filepath.read_bytes())

            # TODO: match import rules

            if email.text_html:
                text = extract_html_text(email.text_html[0])
            elif email.text_plain:
                text = email.text_plain[0]
            else:
                raise ValueError("The email has no no content available for processing")

            columns = DEFAULT_COLUMNS
            response_model_cls = build_response_model(
                output_columns=columns, output_folders=[], attachment_count=0
            )

            template = textwrap.dedent("""\
            # Instruction

            Extract the following email content and output JSON with the provided JSON schema.

            # JSON Schema

            {{ json_schema | tojson }}

            # Email content

            {{ content }}
            """)
            prompt = template_env.from_string(template).render(
                json_schema=response_model_cls.model_json_schema(), content=text
            )
            model_name = "deepcoder"
            messages = think(model=model_name, prompt=prompt)
            print(f"Prompt:\n{prompt}")
            print("Reasoning:", messages[1].content)
            result = extract(
                model=model_name,
                messages=messages,
                response_model_cls=response_model_cls,
            )
            import pprint

            pprint.pprint(result.model_dump(mode="json"))

            # XXX:
            yield None
