"""
NeuroQuant 3.0: Backend Configuration
"""

import os
from pydantic import BaseModel
from typing import List, Dict

class Settings(BaseModel):
    APP_NAME: str = "NeuroQuant 3.0 Sovereign Quant Engine"
    VERSION: str = "3.0.0"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    CACHE_TTL_SECONDS: int = 900  # 15 dakika akıllı önbellekleme
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ]
    
    # Sektörel / Pazar Referans Haritası (Wall Street Sektör ETF'leri)
    BENCHMARK_MAP: Dict[str, str] = {
        "NVDA": "SMH",
        "AMD": "SMH",
        "AVGO": "SMH",
        "TSM": "SMH",
        "AAPL": "QQQ",
        "MSFT": "QQQ",
        "GOOGL": "QQQ",
        "META": "QQQ",
        "TSLA": "QQQ",
        "AMZN": "XLY",
        "JPM": "XLF",
        "GS": "XLF",
        "V": "XLF",
        "LLY": "XLV",
        "BTC-USD": "BTC-USD",
        "ETH-USD": "BTC-USD",
        "SOL-USD": "BTC-USD"
    }

    def get_benchmark(self, ticker: str) -> str:
        clean_t = ticker.upper()
        if clean_t in self.BENCHMARK_MAP:
            return self.BENCHMARK_MAP[clean_t]
        if "-USD" in clean_t:
            return "BTC-USD"
        return "SPY"

settings = Settings()
