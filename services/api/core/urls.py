from django.contrib import admin
from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/token/", TokenObtainPairView.as_view(), name="token"),
    path("api/v1/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
]