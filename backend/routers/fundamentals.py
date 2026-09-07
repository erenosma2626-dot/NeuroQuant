"""
Fundamental Valuation & Earnings Router
"""

from fastapi import APIRouter, HTTPException
from backend.services.fundamental_service import fetch_fundamentals

router = APIRouter(prefix="/api/fundamentals", tags=["Fundamentals"])

@router.get("/{ticker}")
async def get_fundamentals(ticker: str):
    try:
        return fetch_fundamentals(ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
