from pydantic import BaseModel


class InboxBaseModel(BaseModel):
    pass


class InboxMatch(BaseModel):
    pass


class InboxDoc(BaseModel):
    inbox: list[InboxMatch] | None = None
