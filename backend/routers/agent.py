"""
AI Strategist Report Router
"""

from fastapi import APIRouter, HTTPException
from backend.services.market_service import fetch_market_data
from backend.services.model_service import generate_forecast
from backend.services.fundamental_service import fetch_fundamentals
from backend.services.sentiment_service import get_news_and_sentiment
from backend.services.gemini_service import generate_strategist_report

router = APIRouter(prefix="/api/agent", tags=["AI Strategist"])

@router.get("/comment/{ticker}")
async def get_agent_comment(ticker: str):
    try:
        clean_t = ticker.strip().upper()
        m_data = fetch_market_data(clean_t)
        f_data = generate_forecast(clean_t)
        fund_data = fetch_fundamentals(clean_t)
        s_data = get_news_and_sentiment(clean_t)
        
        return generate_strategist_report(
            clean_t,
            market_data=m_data,
            forecast_data=f_data,
            fundamental_data=fund_data,
            sentiment_data=s_data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
