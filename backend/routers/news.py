"""
News & Sentiment Router
"""

from fastapi import APIRouter, HTTPException, Query
from backend.services.sentiment_service import get_news_and_sentiment

router = APIRouter(prefix="/api/news", tags=["News & Sentiment"])

@router.get("/{ticker}")
async def get_news(ticker: str, max_results: int = Query(10, ge=1, le=25)):
    try:
        return get_news_and_sentiment(ticker, max_results=max_results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
