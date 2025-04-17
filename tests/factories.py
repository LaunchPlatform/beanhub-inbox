import dataclasses

from factory import Dict
from factory import Factory
from factory import Faker
from factory import LazyFunction
from factory import List
from factory import SubFactory
from faker import Faker as OriginalFaker

fake = OriginalFaker()

from beanhub_inbox.data_types import InboxEmail


@dataclasses.dataclass(frozen=True)
class EmailAttachment:
    content: bytes
    mime_type: str = "application/octet-stream"
    filename: str | None = None


@dataclasses.dataclass(frozen=True)
class EmailFile:
    headers: dict[str, str]
    subject: str
    from_addresses: list[str]
    recipients: list[str]
    tags: list[str] | None = None
    attachments: list[EmailAttachment] | None = None


class InboxEmailFactory(Factory):
    id = Faker("uuid4")
    mime = Faker("slug")

    class Meta:
        model = InboxEmail


class EmailAttachmentFactory(Factory):
    content = Faker("paragraph")
    mime_type = Faker("mime_type")
    filename = None

    class Meta:
        model = EmailAttachment


class EmailFileFactory(Factory):
    subject = Faker("sentence")
    from_addresses = LazyFunction(lambda: [fake.email()])
    recipients = LazyFunction(lambda: [fake.email()])
    headers = Dict(
        {
            "Date": Faker("past_datetime"),
        }
    )
    attachments = List([SubFactory(EmailAttachmentFactory, mime_type="text/plain")])
    tags = None

    class Meta:
        model = EmailFile
