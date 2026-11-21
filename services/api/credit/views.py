from typing import cast
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from credit.models import Application
from credit.scoring_client import ScoringUnavailable
from credit.serializers import ApplicationCreateSerializer, application_data
from credit.services import AlreadyDecided, score_application


def _own_application(request, pk: int) -> Application:
    return get_object_or_404(
        Application.objects.select_related("decision"), pk=pk, created_by=request.user
    )


def _fresh(application: Application) -> Application:
    return Application.objects.select_related("decision").get(pk=application.pk)


def _score_and_respond(application: Application, success_status: int) -> Response:
    try:
        score_application(application)
    except ScoringUnavailable:
        return Response(
            {
                **application_data(_fresh(application)),
                "detail": "Skorlama servisine şu an ulaşılamıyor, başvuru beklemede.",
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except AlreadyDecided:
        return Response(
            {
                **application_data(_fresh(application)),
                "detail": "Bu başvuru için zaten karar verilmiş.",
            },
            status=status.HTTP_409_CONFLICT,
        )
    return Response(application_data(_fresh(application)), status=success_status)


class ApplicationCreateView(APIView):
    def post(self, request):
        serializer = ApplicationCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = cast(dict, serializer.validated_data)
        application = Application.objects.create(
            sk_id_curr=data["sk_id_curr"],
            form=dict(data.get("application", {})),
            model_key=data["model"],
            created_by=request.user,
        )
        return _score_and_respond(application, status.HTTP_201_CREATED)


class ApplicationDetailView(APIView):
    def get(self, request, pk: int):
        return Response(application_data(_own_application(request, pk)))


class ApplicationScoreView(APIView):
    def post(self, request, pk: int):
        return _score_and_respond(_own_application(request, pk), status.HTTP_200_OK)