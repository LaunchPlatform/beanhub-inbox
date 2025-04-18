import dataclasses
from email.message import EmailMessage

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
    text: EmailAttachment | None = None
    html: EmailAttachment | None = None
    attachments: list[EmailAttachment] | None = None


class InboxEmailFactory(Factory):
    id = Faker("uuid4")
    mime = Faker("slug")

    class Meta:
        model = InboxEmail


class EmailAttachmentFactory(Factory):
    content = LazyFunction(lambda: fake.paragraph().encode("utf8"))
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
    text = SubFactory(EmailAttachmentFactory, mime_type="text/plain")
    html = SubFactory(EmailAttachmentFactory, mime_type="text/html")
    attachments = None
    tags = None

    class Meta:
        model = EmailFile


def make_email_msg(email_file: EmailFile) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = ", ".join(email_file.from_addresses)
    msg["To"] = ", ".join(email_file.recipients)
    msg["Subject"] = email_file.subject

    content_parts = []
    if email_file.text is not None:
        content_parts.append(email_file.text)
    if email_file.html is not None:
        content_parts.append(email_file.html)
    if not content_parts:
        raise ValueError("Need to set at least one of text or html")

    for i, part in enumerate(content_parts):
        if i == 0:
            method = msg.set_content
        else:
            method = msg.add_alternative
        main_type, sub_type = part.mime_type.split("/")
        method(part.content, maintype=main_type, subtype=sub_type)

    if email_file.attachments is not None:
        for attachment in email_file.attachments:
            main_type, sub_type = attachment.mime_type.split("/")
            msg.add_attachment(
                b"xxx", maintype="application", subtype="pdf", filename="foobar"
            )
    # for attachment in email_file.attachments:
    #     msg.add_attachment(
    #         file_data, maintype=attachment.mime_type, subtype=subtype, filename=file_name
    #     )
    return msg
