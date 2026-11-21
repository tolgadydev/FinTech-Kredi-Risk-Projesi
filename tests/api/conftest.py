import pytest
from django.contrib.auth.models import User

from credit.models import Application


@pytest.fixture
def user(db):
    return User.objects.create_user("kurum", password="test-sifre-123")


@pytest.fixture
def application(user):
    return Application.objects.create(
        sk_id_curr=100001, form={"AMT_CREDIT": 450000}, created_by=user
    )