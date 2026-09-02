from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from .routers import products

app = FastAPI(title="Sambalpuri AI Catalog API")

app.include_router(products.router)


@app.exception_handler(OperationalError)
def db_unavailable_handler(request: Request, exc: OperationalError):
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


@app.get("/health")
def health():
    return {"status": "ok"}
