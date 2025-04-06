from factory import Factory
from factory import Faker
from factory import LazyFunction
from faker import Faker as OriginalFaker

fake = OriginalFaker()

from beanhub_inbox.data_types import InboxEmail


class InboxEmailFactory(Factory):
    id = Faker("uuid4")
    message_id = Faker("slug")
    subject = Faker("sentence")
    from_addresses = LazyFunction(lambda: [fake.email()])
    recipients = LazyFunction(lambda: [fake.email()])
    headers = LazyFunction(lambda: {fake.slug(): fake.slug()})

    class Meta:
        model = InboxEmail
