import enum
import typing

import pydantic
from pydantic import BaseModel


@enum.unique
class InboxActionType(str, enum.Enum):
    archive = "archive"
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


SimpleFileMatch = str | StrExactMatch | StrRegexMatch


class InputConfig(InboxBaseModel):
    match: SimpleFileMatch
    loop: list[dict] | None = None


class ImportConfig(InboxBaseModel):
    # Name of import rule, for users to read only
    name: str | None = None
    # match: TxnMatchRule | list[TxnMatchVars]
    # actions: list[Action]


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
