import enum
import typing

import pydantic
from pydantic import BaseModel


@enum.unique
class InboxActionType(str, enum.Enum):
    archive = "archive"
    ignore = "ignore"


@enum.unique
class ImportActionType(str, enum.Enum):
    extract = "extract"
    ignore = "ignore"


class InboxBaseModel(BaseModel):
    pass


class InboxMatch(InboxBaseModel):
    tags: list[str] | None = None
    headers: dict[str, str] | None = None
    subject: str | None = None
    from_address: str | None = None


class ArchiveInboxAction(InboxBaseModel):
    output_file: str
    type: typing.Literal[InboxActionType.archive] = pydantic.Field(
        InboxActionType.archive
    )


class IgnoreInboxAction(InboxBaseModel):
    type: typing.Literal[InboxActionType.ignore]


InboxAction = ArchiveInboxAction | IgnoreInboxAction


class InboxConfig(InboxBaseModel):
    action: InboxAction
    match: InboxMatch | None = None


class StrRegexMatch(InboxBaseModel):
    regex: str


class StrExactMatch(InboxBaseModel):
    equals: str


class StrOneOfMatch(InboxBaseModel):
    one_of: list[str]
    regex: bool = False
    ignore_case: bool = False


class StrPrefixMatch(InboxBaseModel):
    prefix: str


class StrSuffixMatch(InboxBaseModel):
    suffix: str


class StrContainsMatch(InboxBaseModel):
    contains: str


SimpleFileMatch = str | StrExactMatch | StrRegexMatch


class InputConfig(InboxBaseModel):
    match: SimpleFileMatch
    loop: list[dict] | None = None


@enum.unique
class OutputColumnType(enum.Enum):
    str = "str"
    int = "int"
    decimal = "decimal"
    bool = "bool"
    date = "date"
    datetime = "datetime"


class OutputColumn(InboxBaseModel):
    name: str
    type: OutputColumnType
    description: str
    pattern: str | None = None
    required: bool = True


class ExtractConfig(InboxBaseModel):
    output_csv: str
    template: str | None = None


class ExtractImportAction(InboxBaseModel):
    type: typing.Literal[ImportActionType.extract] = pydantic.Field(
        ImportActionType.extract
    )
    extract: ExtractConfig


class IgnoreImportAction(InboxBaseModel):
    type: typing.Literal[ImportActionType.ignore]


ImportAction = ExtractImportAction | IgnoreImportAction


StrMatch = (
    str
    | StrPrefixMatch
    | StrSuffixMatch
    | StrExactMatch
    | StrContainsMatch
    | StrOneOfMatch
)


class EmailFileMatchRule(InboxBaseModel):
    filepath: StrMatch | None = None
    subject: StrMatch | None = None


class ImportConfig(InboxBaseModel):
    # Name of import rule, for users to read only
    name: str | None = None
    match: EmailFileMatchRule | None = None
    actions: list[ImportAction]


class InboxDoc(InboxBaseModel):
    inbox: list[InboxConfig] | None = None
    inputs: list[InputConfig] | None = None
    imports: list[ImportConfig] | None = None


class InboxEmail(InboxBaseModel):
    id: str
    message_id: str
    headers: dict[str, str]
    subject: str
    from_addresses: list[str]
    recipients: list[str]
    tags: list[str] | None = None
