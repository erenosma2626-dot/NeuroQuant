"""
10,000 Capital Replay Simulation Router
"""

from fastapi import APIRouter, HTTPException
from backend.services.simulation_service import run_simulation

router = APIRouter(prefix="/api/simulation", tags=["Simulation"])

@router.get("/{ticker}")
async def get_simulation(ticker: str):
    try:
        return run_simulation(ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
