"""
Macro Barometers & RORO Regime Router
"""

from fastapi import APIRouter, HTTPException
from backend.services.macro_service import get_macro_regime_and_barometers

router = APIRouter(prefix="/api/macro", tags=["Macro & RORO Regime"])

@router.get("/barometers")
async def get_macro_barometers():
    try:
        return get_macro_regime_and_barometers()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
