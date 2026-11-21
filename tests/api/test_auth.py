# pyright: reportAttributeAccessIssue=false
import pytest
from django.contrib.auth.models import User
from rest_framework.test import APIClient


@pytest.fixture
def user(db):
    return User.objects.create_user("kurum", password="test-sifre-123")


def test_token_issued_for_valid_user(user):
    response = APIClient().post(
        "/api/v1/token/", {"username": "kurum", "password": "test-sifre-123"}, format="json"
    )
    assert response.status_code == 200
    assert {"access", "refresh"} <= set(response.json())


def test_wrong_password_gets_401(user):
    response = APIClient().post(
        "/api/v1/token/", {"username": "kurum", "password": "yanlis"}, format="json"
    )
    assert response.status_code == 401