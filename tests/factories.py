import dataclasses
from email.message import EmailMessage

from factory import Dict
from factory import Factory
from factory import Faker
from factory import LazyFunction
from factory import List
from factory import SubFactory
from faker import Faker as OriginalFaker

from beanhub_inbox.data_types import InboxEmail
from beanhub_inbox.processor import EmailFile

fake = OriginalFaker()


@dataclasses.dataclass(frozen=True)
class EmailAttachment:
    content: bytes
    mime_type: str = "application/octet-stream"
    filename: str | None = None


@dataclasses.dataclass(frozen=True)
class MockEmail:
    headers: dict[str, str]
    subject: str
    from_addresses: list[str]
    recipients: list[str]
    tags: list[str] | None = None
    text: EmailAttachment | None = None
    html: EmailAttachment | None = None
    attachments: list[EmailAttachment] | None = None

    def make_msg(self) -> EmailMessage:
        msg = EmailMessage()
        msg["From"] = ", ".join(self.from_addresses)
        msg["To"] = ", ".join(self.recipients)
        msg["Subject"] = self.subject
        date = self.headers.pop("Date")
        if date is not None:
            msg["Date"] = date.strftime("%a, %d %b %Y %H:%M:%S %z")

        content_parts = []
        if self.text is not None:
            content_parts.append(self.text)
        if self.html is not None:
            content_parts.append(self.html)
        if not content_parts:
            raise ValueError("Need to set at least one of text or html")

        for i, part in enumerate(content_parts):
            if i == 0:
                method = msg.set_content
            else:
                method = msg.add_alternative
            main_type, sub_type = part.mime_type.split("/")
            method(part.content, maintype=main_type, subtype=sub_type)

        if self.attachments is not None:
            for attachment in self.attachments:
                main_type, sub_type = attachment.mime_type.split("/")
                msg.add_attachment(
                    attachment.content,
                    maintype=main_type,
                    subtype=sub_type,
                    filename=attachment.filename,
                )
        return msg


class InboxEmailFactory(Factory):
    id = Faker("uuid4")
    message_id = Faker("slug")
    headers = Dict(
        {
            "mock-header": Faker("slug"),
        }
    )
    subject = Faker("sentence")
    from_addresses = List([Faker("email")])
    recipients = List([Faker("email")])
    tags = List([Faker("slug")])

    class Meta:
        model = InboxEmail


class EmailAttachmentFactory(Factory):
    content = LazyFunction(lambda: fake.paragraph().encode("utf8"))
    mime_type = Faker("mime_type")
    filename = None

    class Meta:
        model = EmailAttachment


class MockEmailFactory(Factory):
    subject = Faker("sentence")
    from_addresses = LazyFunction(lambda: [fake.email()])
    recipients = LazyFunction(lambda: [fake.email()])
    headers = Dict(
        {
            "Date": Faker("past_datetime"),
        }
    )
    text = SubFactory(EmailAttachmentFactory, mime_type="text/plain")
    html = SubFactory(EmailAttachmentFactory, mime_type="text/html")
    attachments = None
    tags = None

    class Meta:
        model = MockEmail


class EmailFileFactory(Factory):
    id = Faker("uuid4")
    subject = Faker("sentence")
    filepath = Faker("file_path", extension="eml")
    from_addresses = LazyFunction(lambda: [fake.email()])
    recipients = LazyFunction(lambda: [fake.email()])
    headers = Dict(
        {
            "Mock-Key": Faker("slug"),
        }
    )
    tags = None

    class Meta:
        model = EmailFile
