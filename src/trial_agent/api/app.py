from typing import Any, Dict, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - optional dependency
    FastAPI = None
    BaseModel = object
    HTTPException = Exception
    _import_error = exc
else:
    _import_error = None

from trial_agent.agent.orchestrator import run_pipeline
from trial_agent.config import settings


class TrialRequest(BaseModel):
    trial: Dict[str, Any]
    top_k: Optional[int] = settings.default_top_k


if FastAPI:
    app = FastAPI(title="Trial Agent MVP")

    @app.post("/run")
    async def run_trial(req: TrialRequest):
        try:
            result = run_pipeline(req.trial, top_k=req.top_k or settings.default_top_k)
            return result
        except FileNotFoundError as err:
            raise HTTPException(status_code=500, detail=str(err))  # type: ignore[misc]
else:
    app = None


def get_app():
    """
    Utility to get the FastAPI app or raise a helpful error.
    """
    if app is None:
        raise RuntimeError(
            f"FastAPI not installed: {_import_error}. Install dependencies or use CLI orchestrator."
        )
    return app

