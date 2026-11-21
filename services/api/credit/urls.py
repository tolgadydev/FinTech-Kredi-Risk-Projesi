from django.urls import path

from credit.views import ApplicationCreateView, ApplicationDetailView, ApplicationScoreView

urlpatterns = [
    path("applications", ApplicationCreateView.as_view(), name="application-create"),
    path("applications/<int:pk>", ApplicationDetailView.as_view(), name="application-detail"),
    path("applications/<int:pk>/score", ApplicationScoreView.as_view(), name="application-score"),
]