from pydantic import BaseModel


class InboxBaseModel(BaseModel):
    pass


class InboxMatch(BaseModel):
    pass


class InboxAction(BaseModel):
    pass


class InboxConfig(BaseModel):
    match: InboxMatch
    action: InboxAction


class InboxDoc(BaseModel):
    inbox: list[InboxConfig] | None = None
