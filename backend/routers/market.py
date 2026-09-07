"""
Market Data Router
"""

from fastapi import APIRouter, HTTPException, Query
from backend.services.market_service import fetch_market_data, fetch_screener_data

router = APIRouter(prefix="/api/market", tags=["Market Data"])

@router.get("/screener/all")
async def get_screener():
    try:
        return fetch_screener_data()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{ticker}")
async def get_market(ticker: str, period: str = Query("3y", description="Geçmiş veri periyodu (1y, 2y, 3y, 5y)")):
    try:
        return fetch_market_data(ticker, period=period)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

