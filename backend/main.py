"""
NeuroQuant 3.0: High-Performance Asynchronous FastAPI Backend
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.routers import (
    market,
    forecast,
    fundamentals,
    simulation,
    news,
    agent,
    backtest
)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Sovereign Quant: Institutional Quantitative Intelligence, Forecasting & Multi-Factor Simulation Platform",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware (React / Vite frontend bağlantısı için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router'ları Sisteme Bağla
app.include_router(market.router)
app.include_router(forecast.router)
app.include_router(fundamentals.router)
app.include_router(simulation.router)
app.include_router(news.router)
app.include_router(agent.router)
app.include_router(backtest.router)

@app.get("/")
async def root():
    return {
        "status": "online",
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "docs": "/docs"
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": "active",
        "engine": "Quantile LightGBM + TimesFM 3.0 + Fundamental Event Engine"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
