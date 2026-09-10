"""
News & Sentiment Router (Stock-specific & Global Macro Wall Street)
"""

from fastapi import APIRouter, HTTPException, Query
from backend.services.sentiment_service import get_news_and_sentiment, get_global_macro_news

router = APIRouter(prefix="/api/news", tags=["News & Sentiment"])

@router.get("/global")
async def get_global_news(max_results: int = Query(10, ge=1, le=25)):
    try:
        return get_global_macro_news(max_results=max_results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{ticker}")
async def get_news(ticker: str, max_results: int = Query(10, ge=1, le=25)):
    try:
        clean_t = ticker.strip().upper()
        if clean_t in ("GLOBAL", "MACRO"):
            return get_global_macro_news(max_results=max_results)
        return get_news_and_sentiment(clean_t, max_results=max_results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
