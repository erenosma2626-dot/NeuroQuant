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
    
    # Sektörel / Pazar Referans Haritası
    BENCHMARK_MAP: Dict[str, str] = {
        "NVDA": "SMH",
        "AAPL": "QQQ",
        "MSFT": "QQQ",
        "AMD": "SMH",
        "GOOGL": "QQQ",
        "TSLA": "QQQ",
        "BTC-USD": "BTC-USD",
        "ETH-USD": "BTC-USD",
        "SOL-USD": "BTC-USD",
        "THYAO.IS": "XU100.IS",
        "EREGL.IS": "XU100.IS",
        "ASELS.IS": "XU100.IS",
        "TUPRS.IS": "XU100.IS",
        "BIMAS.IS": "XU100.IS"
    }

    def get_benchmark(self, ticker: str) -> str:
        clean_t = ticker.upper()
        if clean_t in self.BENCHMARK_MAP:
            return self.BENCHMARK_MAP[clean_t]
        if clean_t.endswith(".IS"):
            return "XU100.IS"
        if "-USD" in clean_t:
            return "BTC-USD"
        return "SPY"

settings = Settings()
