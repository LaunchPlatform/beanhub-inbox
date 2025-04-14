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
DEFAULT_COLUMNS: list[OutputColumn] = [
    OutputColumn(
        name="valid",
        type=OutputColumnType.bool,
        description="True if this email is for a transaction such as an invoice or receipt, otherwise False",
    ),
    OutputColumn(
        name="desc",
        type=OutputColumnType.str,
        description="The summary of the transaction in a short sentence",
        required=False,
    ),
    OutputColumn(
        name="merchant",
        type=OutputColumnType.str,
        description="Name of the merchant who sent the email if available",
        required=False,
    ),
    OutputColumn(
        name="amount",
        type=OutputColumnType.decimal,
        description="Transaction amount as a decimal string value, do not include dollar sign and please follow the regex format",
        required=False,
    ),
    OutputColumn(
        name="tax",
        type=OutputColumnType.decimal,
        description="Tax amount as a decimal string value, do not include dollar sign and please follow the regex format",
        required=False,
    ),
    OutputColumn(
        name="txn_id",
        type=OutputColumnType.str,
        description="Id of transaction, such as invoice number or receipt number",
        required=False,
    ),
    OutputColumn(
        name="txn_date",
        type=OutputColumnType.date,
        description="The date of transaction if available, in YYYY-MM-DD format",
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
        value_type = str
    elif output_column.type == OutputColumnType.int:
        value_type = int
    elif output_column.type == OutputColumnType.decimal:
        kwargs["pattern"] = DECIMAL_REGEX
        value_type = str
    elif output_column.type == OutputColumnType.date:
        value_type = datetime.date
    elif output_column.type == OutputColumnType.datetime:
        value_type = datetime.datetime
    elif output_column.type == OutputColumnType.bool:
        value_type = bool
    else:
        raise ValueError(f"Unexpected type {output_column.type}")
    if not output_column.required:
        value_type = typing.Optional[value_type]
    annotated_type = typing.Annotated[value_type, pydantic.Field(**kwargs)]
    return output_column.name, annotated_type


def build_row_model(
    output_columns: list[OutputColumn],
) -> typing.Type[LLMResponseBaseModel]:
    fields = map(build_column_field, output_columns)
    return pydantic.create_model(
        "FieldValue", **dict(fields), __base__=LLMResponseBaseModel
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
        # reasoning_steps=typing.Annotated[list[str], pydantic.Field(description="Reasoning steps")],
        csv_row=build_row_model(output_columns=output_columns),
        **kwargs,
        __base__=LLMResponseBaseModel,
    )


def _stream_think(
    model: str,
    messages: list[ollama.Message],
    end_token: str | None = None,
    options: dict | None = None,
) -> typing.Generator[ollama.ChatResponse, None, ollama.Message]:
    chunks: list[str] = []
    for part in chat(model=model, messages=messages, options=options, stream=True):
        msg_content = part["message"]["content"]
        yield part
        chunks.append(msg_content)
        if end_token is not None and msg_content == end_token:
            break
    return ollama.Message(role="assistant", content="".join(chunks))


def think(
    model: str,
    messages: list[ollama.Message],
    end_token: str | None = None,
    options: dict | None = None,
    stream: bool = False,
) -> typing.Generator[ollama.ChatResponse, None, ollama.Message] | ollama.Message:
    if options is None:
        options = DEDUCTION_DEFAULT_OPTIONS
    if stream:
        return _stream_think(
            model=model, messages=messages, options=options, end_token=end_token
        )
    resp = chat(model=model, messages=messages, options=options)
    if end_token is not None:
        resp.message.content = resp.message.content.split(end_token, 1)[0] + end_token
    return resp.message


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
        stream=True,
    )

    chunks = []
    for part in response:
        msg_content = part["message"]["content"]
        chunks.append(msg_content)
        # XXX:
        print(msg_content, end="", flush=True)

    return response_model_cls.model_validate_json("".join(chunks))
