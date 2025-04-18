import typing

from email_validator import validate_email


T = typing.TypeVar("T")
S = typing.TypeVar("S")
R = typing.TypeVar("R")


class GeneratorResult(typing.Generic[T, S, R]):
    def __init__(self, generator: typing.Generator[T, S, R]):
        self.generator: typing.Generator[T, S, R] = generator
        self.value: R | None = None

    def __iter__(self) -> typing.Generator[T, S, R]:
        self.value = yield from self.generator


def parse_tags(email_address: str, domains: typing.Collection[str]) -> list[str] | None:
    email_info = validate_email(email_address, check_deliverability=False)
    domain = email_info.domain.lower()
    if domain not in domains:
        return None
    parts = email_info.local_part.split("+")
    if len(parts) <= 2:
        return None
    return parts[2:]
