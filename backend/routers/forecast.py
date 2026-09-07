"""
Quantile & Foundation Forecast Router
"""

from fastapi import APIRouter, HTTPException, Query
from backend.services.model_service import generate_forecast

router = APIRouter(prefix="/api/forecast", tags=["Forecast"])

@router.get("/{ticker}")
async def get_forecast(
    ticker: str,
    engine: str = Query("timesfm", description="Model motoru: timesfm, lightgbm, hybrid")
):
    try:
        return generate_forecast(ticker, engine=engine)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
