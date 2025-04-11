import typing

import pydantic
import pytest

from beanhub_inbox.data_types import OutputColumn
from beanhub_inbox.data_types import OutputColumnType
from beanhub_inbox.llm import build_field


@pytest.mark.parametrize(
    "output_column, expected",
    [
        (
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
        )
    ],
)
def test_build_field(output_column: OutputColumn, expected: tuple[str, typing.Type]):
    model = pydantic.create_model("TestModel", **dict([build_field(output_column)]))
    expected_model = pydantic.create_model("TestModel", **dict([expected]))
    assert model.model_json_schema() == expected_model.model_json_schema()
