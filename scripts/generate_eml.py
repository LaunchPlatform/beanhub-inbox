import mimetypes
import os
import typing
from email.message import EmailMessage

import click


@click.command(
    help="Generate an EML file with given headers, content, and attachments."
)
@click.option("--header", multiple=True, help='Email headers in "Key:Value" format')
@click.option(
    "--content-file",
    type=click.Path(exists=True),
    required=True,
    help="File containing the email body",
)
@click.option(
    "--attachment", multiple=True, type=click.Path(exists=True), help="Files to attach"
)
@click.argument("output", type=click.Path())
def generate_eml(
    header: typing.Tuple[str, ...],
    content_file: str,
    attachment: typing.Tuple[str, ...],
    output: str,
) -> None:
    # Read content from the specified file
    with open(content_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Create an EmailMessage object
    msg = EmailMessage()

    # Set headers from the provided list
    for h in header:
        if ":" not in h:
            raise click.BadParameter(f'Invalid header format: {h}. Must be "Key:Value"')
        key, value = h.split(":", 1)
        msg[key.strip()] = value.strip()

    # Set the content of the email
    msg.set_content(content)

    # Attach specified files
    for attachment_file in attachment:
        with open(attachment_file, "rb") as f:
            file_data = f.read()
            file_name = os.path.basename(attachment_file)
            mime_type, _ = mimetypes.guess_type(attachment_file)
            if mime_type is None:
                mime_type = "application/octet-stream"
            maintype, subtype = mime_type.split("/", 1)
            msg.add_attachment(
                file_data, maintype=maintype, subtype=subtype, filename=file_name
            )

    # Save the email message to the output file
    with open(output, "wb") as f:
        f.write(msg.as_bytes())


if __name__ == "__main__":
    generate_eml()
