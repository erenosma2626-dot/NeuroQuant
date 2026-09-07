"""
Quantile Forecast Router
"""

from fastapi import APIRouter, HTTPException
from backend.services.model_service import generate_forecast

router = APIRouter(prefix="/api/forecast", tags=["Forecast"])

@router.get("/{ticker}")
async def get_forecast(ticker: str):
    try:
        return generate_forecast(ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
