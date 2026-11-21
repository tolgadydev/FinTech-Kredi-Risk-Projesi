from django.urls import path

from credit.views import (
    ApplicationCreateView,
    ApplicationDetailView,
    ApplicationScoreView,
    BatchCreateView,
    BatchDetailView,
)

urlpatterns = [
    path("applications", ApplicationCreateView.as_view(), name="application-create"),
    path("applications/<int:pk>", ApplicationDetailView.as_view(), name="application-detail"),
    path("applications/<int:pk>/score", ApplicationScoreView.as_view(), name="application-score"),
    path("batches", BatchCreateView.as_view(), name="batch-create"),
    path("batches/<int:pk>", BatchDetailView.as_view(), name="batch-detail"),
]