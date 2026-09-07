"""
Historical Backtest Router
"""

from fastapi import APIRouter, HTTPException
from backend.services.simulation_service import run_simulation

router = APIRouter(prefix="/api/backtest", tags=["Backtest"])

@router.get("/{ticker}")
async def get_backtest(ticker: str):
    try:
        sim = run_simulation(ticker)
        return {
            "ticker": sim["ticker"],
            "benchmark": sim["benchmark"],
            "start_date": sim["start_date"],
            "end_date": sim["end_date"],
            "test_period_days": sim["test_period_days"],
            "performance": sim["performance"],
            "total_trades": len(sim["trades"])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
