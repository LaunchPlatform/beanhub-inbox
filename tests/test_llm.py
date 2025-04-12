import datetime
import enum
import textwrap
import typing

import pydantic
import pytest

from beanhub_inbox.data_types import OutputColumn
from beanhub_inbox.data_types import OutputColumnType
from beanhub_inbox.llm import build_archive_attachment_model
from beanhub_inbox.llm import build_column_field
from beanhub_inbox.llm import build_response_model
from beanhub_inbox.llm import build_row_model
from beanhub_inbox.llm import DECIMAL_REGEX
from beanhub_inbox.llm import LLMResponseBaseModel


@pytest.mark.parametrize(
    "output_column, expected",
    [
        pytest.param(
            OutputColumn(
                name="desc",
                type=OutputColumnType.str,
                description="summary of the transaction from the invoice or receipt",
            ),
            (
                "desc",
                typing.Annotated[
                    str,
                    pydantic.Field(
                        description="summary of the transaction from the invoice or receipt"
                    ),
                ],
            ),
            id="str",
        ),
        pytest.param(
            OutputColumn(
                name="txn_id",
                type=OutputColumnType.str,
                description="Id of transaction",
                pattern="[0-9]{10}",
            ),
            (
                "txn_id",
                typing.Annotated[
                    str,
                    pydantic.Field(
                        description="Id of transaction", pattern="[0-9]{10}"
                    ),
                ],
            ),
            id="str-with-regex",
        ),
        pytest.param(
            OutputColumn(
                name="amount",
                type=OutputColumnType.decimal,
                description="transaction amount",
            ),
            (
                "amount",
                typing.Annotated[
                    str,
                    pydantic.Field(
                        description="transaction amount", pattern=DECIMAL_REGEX
                    ),
                ],
            ),
            id="decimal",
        ),
        pytest.param(
            OutputColumn(
                name="year",
                type=OutputColumnType.int,
                description="transaction year",
            ),
            (
                "year",
                typing.Annotated[
                    int,
                    pydantic.Field(description="transaction year"),
                ],
            ),
            id="int",
        ),
        pytest.param(
            OutputColumn(
                name="valid",
                type=OutputColumnType.bool,
                description="is this a invoice or something else",
            ),
            (
                "valid",
                typing.Annotated[
                    bool,
                    pydantic.Field(description="is this a invoice or something else"),
                ],
            ),
            id="bool",
        ),
        pytest.param(
            OutputColumn(
                name="date",
                type=OutputColumnType.date,
                description="date of transaction",
            ),
            (
                "date",
                typing.Annotated[
                    datetime.date,
                    pydantic.Field(description="date of transaction"),
                ],
            ),
            id="date",
        ),
        pytest.param(
            OutputColumn(
                name="timestamp",
                type=OutputColumnType.datetime,
                description="timestamp of transaction",
            ),
            (
                "timestamp",
                typing.Annotated[
                    datetime.datetime,
                    pydantic.Field(description="timestamp of transaction"),
                ],
            ),
            id="datetime",
        ),
    ],
)
def test_build_column_field(
    output_column: OutputColumn, expected: tuple[str, typing.Type]
):
    model = pydantic.create_model(
        "TestModel", **dict([build_column_field(output_column)])
    )
    expected_model = pydantic.create_model("TestModel", **dict([expected]))
    assert model.model_json_schema() == expected_model.model_json_schema()


@pytest.mark.parametrize(
    "output_columns, expected",
    [
        (
            [
                OutputColumn(
                    name="desc",
                    type=OutputColumnType.str,
                    description="summary of the transaction from the invoice or receipt",
                ),
                OutputColumn(
                    name="year",
                    type=OutputColumnType.int,
                    description="transaction year",
                ),
            ],
            pydantic.create_model(
                "CsvRow",
                desc=typing.Annotated[
                    str,
                    pydantic.Field(
                        description="summary of the transaction from the invoice or receipt"
                    ),
                ],
                year=typing.Annotated[
                    int, pydantic.Field(description="transaction year")
                ],
            ),
        ),
    ],
)
def test_build_row_model(
    output_columns: list[OutputColumn], expected: typing.Type[LLMResponseBaseModel]
):
    model = build_row_model(output_columns=output_columns)
    assert model.model_json_schema() == expected.model_json_schema()


@pytest.mark.parametrize(
    "output_folders, attachment_count, expected",
    [
        (
            ["a", "b", "c"],
            3,
            pydantic.create_model(
                "ArchiveAttachment",
                attachment_index=typing.Annotated[
                    int,
                    pydantic.Field(
                        ge=0, lt=3, description="The index of email attachment file"
                    ),
                ],
                outout_folder=typing.Annotated[
                    enum.Enum(
                        "OutputFolder",
                        {k: k for k in ["a", "b", "c"]},
                    ),
                    pydantic.Field(
                        description="which folder to archive the email attachment file to"
                    ),
                ],
                filename=typing.Annotated[
                    str,
                    pydantic.Field(
                        description="The output filename of email attachment file to write to the output folder"
                    ),
                ],
            ),
        ),
    ],
)
def test_build_archive_attachment_model(
    output_folders: list[str],
    attachment_count: int,
    expected: typing.Type[LLMResponseBaseModel],
):
    model = build_archive_attachment_model(
        output_folders=output_folders, attachment_count=attachment_count
    )
    assert model.model_json_schema() == expected.model_json_schema()


# TODO: temporary
def test_extract():
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen2.5-14B-Instruct-1M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    resp_model_cls = build_response_model(
        output_columns=[
            OutputColumn(
                name="desc",
                type=OutputColumnType.str,
                description="summary of the transaction",
            ),
            OutputColumn(
                name="amount",
                type=OutputColumnType.decimal,
                description="transaction amount",
            ),
            OutputColumn(
                name="txn_id",
                type=OutputColumnType.str,
                description="Id of transaction",
            ),
            OutputColumn(
                name="date",
                type=OutputColumnType.date,
                description="transaction date",
            ),
        ],
        output_folders=["receipts", "invoices"],
        attachment_count=2,
    )
    import json

    print(json.dumps(resp_model_cls.model_json_schema()))
    # return
    prompt = textwrap.dedent(f"""\
    Extract data from the following email into JSON payload.
    Output ArchiveAttachment in the JSON for attachment files that look like invoices to the "invoices" folder.
    Output ArchiveAttachment in the JSON for attachment files that look like receipts to the "receipts" folder.
    
    # JSON schema
    
    ```
    {json.dumps(resp_model_cls.model_json_schema())}
    ```
    
    # Output folders
    
    - receipts
    - invoices
    
    # Attachments
    
    1. filename="GitHub-invoice.pdf", mime_type="application/pdf"
    2. filename="GitHub-receipt.pdf", mime_type="application/pdf"
    
    # Email content
    
    We received payment for your GitHub.com subscription. Thanks for your business!

    Questions? Visit https://github.com/contact

    ------------------------------------
    GITHUB RECEIPT - PERSONAL SUBSCRIPTION - fangpenlin


    GitHub Developer Plan - Month: $4.00 USD
    Apr 1, 2025 - Apr 30, 2025


    Tax: $0.00 USD
    Total: $4.00 USD*

    Charged to: Visa (4*** **** **** 1234)
    Transaction ID: 6cd4f0a6-0686-45bf-a077-775e95206da6
    Date: 01 Apr 2025 09:55AM PDT


    GitHub, Inc.
    88 Colin P. Kelly Jr. Street
    San Francisco, CA 94107
    ------------------------------------

    * VAT/GST paid directly by GitHub, where applicable
    """)

    # print(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True))
    from outlines import models, generate

    model = models.transformers(
        model_name,
        device="auto",
        model_kwargs=dict(
            torch_dtype="auto",
            device_map="auto",
        ),
    )
    generator = generate.json(model, resp_model_cls)
    result = generator(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    )
    print(result)
