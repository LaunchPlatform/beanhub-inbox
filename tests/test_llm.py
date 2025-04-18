import datetime
import json
import logging
import typing

import ollama
import pydantic
import pytest
from pytest_mock import MockFixture

from beanhub_inbox.data_types import OutputColumn
from beanhub_inbox.data_types import OutputColumnType
from beanhub_inbox.llm import build_column_field
from beanhub_inbox.llm import build_row_model
from beanhub_inbox.llm import DECIMAL_REGEX
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
                "Row",
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
    "model, prompt, end_token",
    [
        ("deepcoder", "What is the result of 1 + 1?", "</think>"),
    ],
)
def test_think(mocker: MockFixture, model: str, prompt: str, end_token: str):
    mock_chat = mocker.patch.object(ollama, "chat")
    mock_chat.return_value = ollama.ChatResponse(
        message=ollama.Message(
            role="assistant",
            content="<think>The result of 1 + 1 is 2</think> The result is 2",
        )
    )

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
def test_think_stream(mocker: MockFixture, model: str, prompt: str, end_token: str):
    def generate_result():
        for chunk in [
            "<think>",
            "The",
            " result",
            " of",
            " 1",
            " +",
            " 1",
            " is",
            " 2",
            "</think>",
            " The",
            " result",
            " is",
            " 2",
        ]:
            yield ollama.ChatResponse(
                message=ollama.Message(role="assistant", content=chunk)
            )

    mock_chat = mocker.patch.object(ollama, "chat")
    mock_chat.return_value = generate_result()

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
def test_extract(
    mocker: MockFixture, model: str, prompt: str, end_token: str, expected: int
):
    def generate_result():
        for chunk in json.dumps(dict(value=2)):
            yield ollama.ChatResponse(
                message=ollama.Message(role="assistant", content=chunk)
            )

    mock_chat = mocker.patch.object(ollama, "chat")
    mock_chat.side_effect = [
        ollama.ChatResponse(
            message=ollama.Message(
                role="assistant",
                content="<think>The result of 1 + 1 is 2</think> The result is 2",
            )
        ),
        generate_result(),
    ]

    messages = [ollama.Message(role="user", content=prompt)]
    think_message = think(model=model, messages=messages, end_token=end_token)
    messages.append(think_message)

    class CalculationResult(LLMResponseBaseModel):
        value: int

    result = extract(
        model=model, messages=messages, response_model_cls=CalculationResult
    )
    assert result.value == 2
