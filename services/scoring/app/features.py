import pandas as pd
from credit_features.application import build_g1
from credit_features.frame import FeatureSpec, align_frame

from app.schemas import ScoreRequest


def request_to_frame(request: ScoreRequest, spec: FeatureSpec) -> pd.DataFrame:
    frame = build_g1(pd.DataFrame([request.application.model_dump()]))
    for name, value in request.behaviour.items():
        frame[name] = value
    return align_frame(frame, spec)