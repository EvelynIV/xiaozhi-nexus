from __future__ import annotations

from fastapi import FastAPI

from xiaozhi_nexus.api.ws import router as ws_router


def create_app() -> FastAPI:
    app = FastAPI(title="xiaozhi-nexus")
    app.include_router(ws_router)
    return app

