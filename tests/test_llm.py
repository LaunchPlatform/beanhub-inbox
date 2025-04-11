import datetime
import enum
import textwrap
import typing

import pydantic
import pytest

from beanhub_inbox.data_types import OutputColumn
from beanhub_inbox.data_types import OutputColumnType
from beanhub_inbox.llm import build_archive_attachment_model
from beanhub_inbox.llm import build_column_field
from beanhub_inbox.llm import build_response_model
from beanhub_inbox.llm import build_row_model
from beanhub_inbox.llm import DECIMAL_REGEX
from beanhub_inbox.llm import LLMResponseBaseModel


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
                "ArchiveAttachmentAction",
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
