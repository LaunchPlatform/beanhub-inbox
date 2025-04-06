import enum
import typing

import pydantic
from pydantic import BaseModel


@enum.unique
class ActionType(str, enum.Enum):
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
    type: typing.Literal[ActionType.archive] = pydantic.Field(ActionType.archive)


class IgnoreInboxAction(InboxBaseModel):
    type: typing.Literal[ActionType.ignore]


InboxAction = ArchiveInboxAction | IgnoreInboxAction


class InboxConfig(InboxBaseModel):
    action: InboxAction
    match: InboxMatch | None = None


class InboxDoc(InboxBaseModel):
    inbox: list[InboxConfig] | None = None


class InboxEmail(InboxBaseModel):
    id: str
    message_id: str
    headers: dict[str, str]
    subject: str
    from_addresses: list[str]
    recipients: list[str]
    tags: list[str] | None = None
