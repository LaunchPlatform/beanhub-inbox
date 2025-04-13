import datetime
import enum
import typing

import ollama
import pydantic
from ollama import chat

from .data_types import OutputColumn
from .data_types import OutputColumnType


DECIMAL_REGEX = "^-?(0|[1-9][0-9]*)(\\.[0-9]+)?$"
DEDUCTION_DEFAULT_OPTIONS = dict(temperature=0)
DEDUCTION_DEFAULT_END_TOKEN = "</think>"
DEFAULT_COLUMNS: list[OutputColumn] = [
    OutputColumn(
        name="desc",
        type=OutputColumnType.str,
        description="The summary of the transaction",
    ),
    OutputColumn(
        name="merchant",
        type=OutputColumnType.str,
        description="Name of the merchant if available",
        required=False,
    ),
    OutputColumn(
        name="amount",
        type=OutputColumnType.decimal,
        description="Transaction amount, do not include dollar sign and please follow the regex format",
    ),
    OutputColumn(
        name="txn_id",
        type=OutputColumnType.str,
        description="Id of transaction if available",
        required=False,
    ),
    OutputColumn(
        name="txn_date",
        type=OutputColumnType.date,
        description="The date of transaction if available",
        required=False,
    ),
]


class LLMResponseBaseModel(pydantic.BaseModel):
    pass


def build_column_field(output_column: OutputColumn) -> (str, typing.Type):
    kwargs = dict(
        description=output_column.description,
        default=... if output_column.required else None,
    )
    annotated_type: typing.Type
    if output_column.type == OutputColumnType.str:
        if output_column.pattern is not None:
            kwargs["pattern"] = output_column.pattern
        annotated_type = typing.Annotated[str, pydantic.Field(**kwargs)]
    elif output_column.type == OutputColumnType.int:
        annotated_type = typing.Annotated[int, pydantic.Field(**kwargs)]
    elif output_column.type == OutputColumnType.decimal:
        annotated_type = typing.Annotated[
            str, pydantic.Field(pattern=DECIMAL_REGEX, **kwargs)
        ]
    elif output_column.type == OutputColumnType.date:
        annotated_type = typing.Annotated[datetime.date, pydantic.Field(**kwargs)]
    elif output_column.type == OutputColumnType.datetime:
        annotated_type = typing.Annotated[datetime.datetime, pydantic.Field(**kwargs)]
    elif output_column.type == OutputColumnType.bool:
        annotated_type = typing.Annotated[bool, pydantic.Field(**kwargs)]
    else:
        raise ValueError(f"Unexpected type {output_column.type}")
    return output_column.name, annotated_type


def build_row_model(
    output_columns: list[OutputColumn],
) -> typing.Type[pydantic.BaseModel]:
    fields = map(build_column_field, output_columns)
    return pydantic.create_model(
        "CsvRow", **dict(fields), __base__=LLMResponseBaseModel
    )


def build_archive_attachment_model(output_folders: list[str], attachment_count: int):
    if attachment_count <= 0:
        raise ValueError(f"Invalid attachment count {attachment_count}")
    if not output_folders:
        raise ValueError("The output_folders value cannot be empty")
    OutputFolder = enum.Enum(
        "OutputFolder",
        {output_folder: output_folder for output_folder in output_folders},
    )
    return pydantic.create_model(
        "ArchiveAttachment",
        attachment_index=typing.Annotated[
            int,
            pydantic.Field(
                ge=1,
                le=attachment_count,
                description="The index of email attachment file, starting from 1",
            ),
        ],
        outout_folder=typing.Annotated[
            OutputFolder,
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
        __base__=LLMResponseBaseModel,
    )


def build_response_model(
    output_columns: list[OutputColumn],
    output_folders: list[str],
    attachment_count: int,
) -> typing.Type[LLMResponseBaseModel]:
    kwargs = {}
    if attachment_count > 0:
        kwargs["archive_attachments"] = typing.Annotated[
            list[
                build_archive_attachment_model(
                    output_folders=output_folders,
                    attachment_count=attachment_count,
                )
                | None
            ],
            pydantic.Field(None),
        ]
    return pydantic.create_model(
        "LLMResponse",
        csv_row=build_row_model(output_columns=output_columns),
        **kwargs,
        __base__=LLMResponseBaseModel,
    )


def think(
    model: str,
    prompt: str,
    end_token: str = DEDUCTION_DEFAULT_END_TOKEN,
    options: dict | None = None,
) -> list[ollama.Message]:
    if options is None:
        options = DEDUCTION_DEFAULT_OPTIONS
    chunks = []
    messages = [ollama.Message(role="user", content=prompt)]
    for part in chat(model=model, messages=messages, options=options, stream=True):
        msg_content = part["message"]["content"]
        chunks.append(msg_content)
        if msg_content == end_token:
            break
    messages.append(ollama.Message(role="assistant", content="".join(chunks)))
    return messages


T = typing.TypeVar("T", bound=LLMResponseBaseModel)


def extract(
    model: str,
    messages: list[ollama.Message],
    response_model_cls: typing.Type[T],
    options: dict | None = None,
) -> T:
    if options is None:
        options = DEDUCTION_DEFAULT_OPTIONS
    response = chat(
        model=model,
        messages=messages,
        options=options,
        format=response_model_cls.model_json_schema(),
    )
    return response_model_cls.model_validate_json(response["message"]["content"])
