from pydantic import BaseModel


class InboxBaseModel(BaseModel):
    pass


class InboxMatch(InboxBaseModel):
    tags: list[str] | None = None
    headers: dict[str, str] | None = None
    subject: str | None = None
    from_address: str | None = None


class InboxAction(InboxBaseModel):
    output_file: str


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
    from_address: list[str]
    recipients: list[str]
    tags: list[str] | None = None
