import decimal
import typing

import pydantic

from .data_types import OutputColumn
from .data_types import OutputColumnType


def build_field(output_column: OutputColumn) -> (str, typing.Type):
    kwargs = dict(description=output_column.description)
    annotated_type: typing.Type
    if output_column.type == OutputColumnType.str:
        if output_column.pattern is not None:
            kwargs["pattern"] = output_column.pattern
        annotated_type = typing.Annotated[str, pydantic.Field(**kwargs)]
    elif output_column.type == OutputColumnType.int:
        annotated_type = typing.Annotated[int, pydantic.Field(**kwargs)]
    elif output_column.type == OutputColumnType.decimal:
        annotated_type = typing.Annotated[decimal.Decimal, pydantic.Field(**kwargs)]
    elif output_column.type == OutputColumnType.bool:
        annotated_type = typing.Annotated[bool, pydantic.Field(**kwargs)]
    else:
        raise ValueError(f"Unexpected type {output_column.type}")
    return output_column.name, annotated_type


def build_response_model(
    output_columns: list[OutputColumn],
) -> typing.Type[pydantic.BaseModel]:
    fields = map(build_field, output_columns)
    return pydantic.create_model("LLMResponse", **dict(fields))
