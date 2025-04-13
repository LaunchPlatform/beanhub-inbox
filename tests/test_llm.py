import datetime
import enum
import json
import logging
import textwrap
import typing

import ollama
import pydantic
import pytest

from beanhub_inbox.data_types import OutputColumn
from beanhub_inbox.data_types import OutputColumnType
from beanhub_inbox.llm import build_archive_attachment_model
from beanhub_inbox.llm import build_column_field
from beanhub_inbox.llm import build_response_model
from beanhub_inbox.llm import build_row_model
from beanhub_inbox.llm import DECIMAL_REGEX
from beanhub_inbox.llm import DEDUCTION_DEFAULT_OPTIONS
from beanhub_inbox.llm import DEFAULT_COLUMNS
from beanhub_inbox.llm import extract
from beanhub_inbox.llm import LLMResponseBaseModel
from beanhub_inbox.llm import think
from beanhub_inbox.utils import GeneratorResult

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "output_column, expected",
    [
        pytest.param(
            OutputColumn(
                name="desc",
                type=OutputColumnType.str,
                description="summary of the transaction from the invoice or receipt",
            ),
            (
                "desc",
                typing.Annotated[
                    str,
                    pydantic.Field(
                        description="summary of the transaction from the invoice or receipt"
                    ),
                ],
            ),
            id="str",
        ),
        pytest.param(
            OutputColumn(
                name="txn_id",
                type=OutputColumnType.str,
                description="Id of transaction",
                pattern="[0-9]{10}",
            ),
            (
                "txn_id",
                typing.Annotated[
                    str,
                    pydantic.Field(
                        description="Id of transaction", pattern="[0-9]{10}"
                    ),
                ],
            ),
            id="str-with-regex",
        ),
        pytest.param(
            OutputColumn(
                name="amount",
                type=OutputColumnType.decimal,
                description="transaction amount",
            ),
            (
                "amount",
                typing.Annotated[
                    str,
                    pydantic.Field(
                        description="transaction amount", pattern=DECIMAL_REGEX
                    ),
                ],
            ),
            id="decimal",
        ),
        pytest.param(
            OutputColumn(
                name="year",
                type=OutputColumnType.int,
                description="transaction year",
            ),
            (
                "year",
                typing.Annotated[
                    int,
                    pydantic.Field(description="transaction year"),
                ],
            ),
            id="int",
        ),
        pytest.param(
            OutputColumn(
                name="valid",
                type=OutputColumnType.bool,
                description="is this a invoice or something else",
            ),
            (
                "valid",
                typing.Annotated[
                    bool,
                    pydantic.Field(description="is this a invoice or something else"),
                ],
            ),
            id="bool",
        ),
        pytest.param(
            OutputColumn(
                name="date",
                type=OutputColumnType.date,
                description="date of transaction",
            ),
            (
                "date",
                typing.Annotated[
                    datetime.date,
                    pydantic.Field(description="date of transaction"),
                ],
            ),
            id="date",
        ),
        pytest.param(
            OutputColumn(
                name="timestamp",
                type=OutputColumnType.datetime,
                description="timestamp of transaction",
            ),
            (
                "timestamp",
                typing.Annotated[
                    datetime.datetime,
                    pydantic.Field(description="timestamp of transaction"),
                ],
            ),
            id="datetime",
        ),
    ],
)
def test_build_column_field(
    output_column: OutputColumn, expected: tuple[str, typing.Type]
):
    model = pydantic.create_model(
        "TestModel", **dict([build_column_field(output_column)])
    )
    expected_model = pydantic.create_model("TestModel", **dict([expected]))
    assert model.model_json_schema() == expected_model.model_json_schema()


@pytest.mark.parametrize(
    "output_columns, expected",
    [
        (
            [
                OutputColumn(
                    name="desc",
                    type=OutputColumnType.str,
                    description="summary of the transaction from the invoice or receipt",
                ),
                OutputColumn(
                    name="year",
                    type=OutputColumnType.int,
                    description="transaction year",
                ),
            ],
            pydantic.create_model(
                "CsvRow",
                desc=typing.Annotated[
                    str,
                    pydantic.Field(
                        description="summary of the transaction from the invoice or receipt"
                    ),
                ],
                year=typing.Annotated[
                    int, pydantic.Field(description="transaction year")
                ],
            ),
        ),
    ],
)
def test_build_row_model(
    output_columns: list[OutputColumn], expected: typing.Type[LLMResponseBaseModel]
):
    model = build_row_model(output_columns=output_columns)
    assert model.model_json_schema() == expected.model_json_schema()


@pytest.mark.parametrize(
    "output_folders, attachment_count, expected",
    [
        (
            ["a", "b", "c"],
            3,
            pydantic.create_model(
                "ArchiveAttachment",
                attachment_index=typing.Annotated[
                    int,
                    pydantic.Field(
                        ge=0, lt=3, description="The index of email attachment file"
                    ),
                ],
                outout_folder=typing.Annotated[
                    enum.Enum(
                        "OutputFolder",
                        {k: k for k in ["a", "b", "c"]},
                    ),
                    pydantic.Field(
                        description="which folder to archive the email attachment file to"
                    ),
                ],
                filename=typing.Annotated[
                    str,
                    pydantic.Field(
                        description="The output filename of email attachment file to write to the output folder"
                    ),
                ],
            ),
        ),
    ],
)
def test_build_archive_attachment_model(
    output_folders: list[str],
    attachment_count: int,
    expected: typing.Type[LLMResponseBaseModel],
):
    model = build_archive_attachment_model(
        output_folders=output_folders, attachment_count=attachment_count
    )
    assert model.model_json_schema() == expected.model_json_schema()


@pytest.mark.parametrize(
    "model, prompt, end_token",
    [
        ("deepcoder", "What is the result of 1 + 1?", "</think>"),
    ],
)
def test_think(model: str, prompt: str, end_token: str):
    think_msg = think(
        model=model,
        messages=[ollama.Message(role="user", content=prompt)],
        end_token=end_token,
    )
    assert think_msg.role == "assistant"
    assert think_msg.content.startswith("<think>")
    assert think_msg.content.endswith("</think>")
    logger.info("Think content:\n%s", think_msg.content)
    assert "2" in think_msg.content


@pytest.mark.parametrize(
    "model, prompt, end_token",
    [
        ("deepcoder", "What is the result of 1 + 1?", "</think>"),
    ],
)
def test_think_stream(model: str, prompt: str, end_token: str):
    chunks: list[str] = []
    think_generator = GeneratorResult(
        think(
            model=model,
            messages=[ollama.Message(role="user", content=prompt)],
            end_token=end_token,
            stream=True,
        )
    )
    for part in think_generator:
        assert part.message.role == "assistant"
        chunks.append(part.message.content)
    content = "".join(chunks)
    assert content.startswith("<think>")
    assert content.endswith("</think>")
    logger.info("Think content:\n%s", content)
    assert "2" in content
    assert think_generator.value.content == content
    assert think_generator.value.role == "assistant"


@pytest.mark.parametrize(
    "model, prompt, end_token, expected",
    [
        pytest.param(
            "deepcoder",
            "What is the result of 1 + 1?",
            "</think>",
            2,
            id="deepcoder-simple-math",
        ),
    ],
)
def test_extract(model: str, prompt: str, end_token: str, expected: int):
    messages = [ollama.Message(role="user", content=prompt)]
    think_message = think(model=model, messages=messages, end_token=end_token)
    messages.append(think_message)

    class CalculationResult(LLMResponseBaseModel):
        value: int

    result = extract(
        model=model, messages=messages, response_model_cls=CalculationResult
    )
    assert result.value == 2


@pytest.mark.parametrize(
    "columns, output_folders, attachment_count, prompt, expected",
    [
        # pytest.param(
        #     [
        #         OutputColumn(
        #             name="amount",
        #             type=OutputColumnType.int,
        #             description="transaction amount",
        #         ),
        #     ],
        #     [],
        #     0,
        #     textwrap.dedent("""\
        #     Extract the following email content and output JSON
        #
        #     # Email content
        #
        #     Thank you for purchase BeanHub.io, the total amount is $30.00 USD
        #
        #     """),
        #     {"csv_row": {"amount": 30}},
        #     id="minimal",
        # ),
        pytest.param(
            DEFAULT_COLUMNS,
            [],
            0,
            textwrap.dedent("""\
            # Instruction
            
            Extract the following email content and output JSON with the provided JSON schema.
            
            # JSON Schema
            
            {json_schema}

            # Email content
            From: BeanHub <noreply@beanhub.io>
            To: user+repo@inbox.beanhub.io
            Subject: Your BeanHub Pro subscription receipt

            Thank you for purchase BeanHub Pro, the total amount is $29.99 USD.
            The transaction id is: 7ffa4dbf-3f51-4c23-a85b-edd837db29ee
            The tax amount is $1.23 USD.
            The payment is processed on April 12, 2025.
            
            BeanHub Team.

            """),
            {
                "csv_row": {
                    "amount": "29.99",
                    "tax": "1.23",
                    "desc": "BeanHub Pro Subscription",
                    "merchant": "BeanHub",
                    "txn_id": "7ffa4dbf-3f51-4c23-a85b-edd837db29ee",
                    "txn_date": "2025-04-12",
                }
            },
            id="default-columns",
        ),
    ],
)
def test_extract_email_values(
    prompt: str,
    columns: list[OutputColumn],
    output_folders: list[str],
    attachment_count: int,
    expected: dict,
):
    model_name = "deepcoder"
    response_model_cls = build_response_model(
        output_columns=columns,
        output_folders=output_folders,
        attachment_count=attachment_count,
    )
    messages = think(
        model=model_name,
        prompt=prompt.format(json_schema=response_model_cls.model_json_schema()),
    )
    print("@" * 20, messages[1].content)
    logger.info("Think content:\n%s", messages[1].content)

    result = extract(
        model=model_name,
        messages=messages,
        response_model_cls=response_model_cls,
    )
    assert result.model_dump(mode="json") == expected
