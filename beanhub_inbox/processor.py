import re

from .data_types import InboxConfig
from .data_types import InboxEmail
from .data_types import InboxMatch


def match_inbox_email(email: InboxEmail, match: InboxMatch) -> bool:
    if match.tags is not None:
        if email.tags is None:
            return False
        email_tags = frozenset(email.tags)
        matching_tags = frozenset(match.tags)
        if matching_tags.intersection(email_tags) != matching_tags:
            return False
    if match.subject is not None:
        if re.match(match.subject, match.subject) is None:
            return False

    return True


def process_inbox_email(email: InboxEmail, inbox_configs: list[InboxConfig]):
    pass
