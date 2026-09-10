from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from backend.services.simulation_service import run_simulation
from backend.services.portfolio_sim_service import run_portfolio_simulation

router = APIRouter(prefix="/api/simulation", tags=["Simulation"])

class BasketItem(BaseModel):
    ticker: str
    weight_pct: float = Field(..., ge=0.0, le=100.0)

class PortfolioSimRequest(BaseModel):
    basket: List[BasketItem]
    cash_weight_pct: float = Field(20.0, ge=0.0, le=100.0)
    max_cash_pct: float = Field(80.0, ge=10.0, le=100.0)
    initial_capital: float = Field(100000.0, gt=0.0)
    rebalance_interval_days: int = Field(7, ge=1, le=30)

@router.get("/{ticker}")
async def get_simulation(ticker: str):
    try:
        return run_simulation(ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/portfolio")
async def simulate_portfolio(req: PortfolioSimRequest):
    try:
        basket_dicts = [{"ticker": item.ticker, "weight_pct": item.weight_pct} for item in req.basket]
        return run_portfolio_simulation(
            basket=basket_dicts,
            cash_weight_pct=req.cash_weight_pct,
            max_cash_pct=req.max_cash_pct,
            initial_capital=req.initial_capital,
            rebalance_interval_days=req.rebalance_interval_days
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

