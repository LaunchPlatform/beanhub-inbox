from pydantic import BaseModel


class InboxBaseModel(BaseModel):
    pass


class InboxMatch(BaseModel):
    tags: list[str] | None = None
    headers: dict[str, str] | None = None
    subject: str | None = None


class InboxAction(BaseModel):
    output_file: str


class InboxConfig(BaseModel):
    match: InboxMatch
    action: InboxAction


class InboxDoc(BaseModel):
    inbox: list[InboxConfig] | None = None
