"""
Market Data & Quantitative Feature Service
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Any, List
from backend.config import settings
from backend.services.cache_service import cache

def fetch_market_data(ticker: str, period: str = "3y") -> Dict[str, Any]:
    """
    yfinance üzerinden OHLCV ve teknik/sektörel göstergeleri çeker.
    TradingView Lightweight Charts formatına uygun liste ve son durum metrikleri döndürür.
    """
    clean_ticker = ticker.strip().upper()
    cache_key = f"market_data_{clean_ticker}_{period}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    benchmark = settings.get_benchmark(clean_ticker)
    
    # 1. Veri İndirme
    df_asset = yf.download(clean_ticker, period=period, progress=False)
    if df_asset.empty:
        raise ValueError(f"'{clean_ticker}' için piyasa verisi bulunamadı.")
        
    df_bench = yf.download(benchmark, period=period, progress=False)
    
    # MultiIndex Sütun Düzleştirme
    for d, s in [(df_asset, clean_ticker), (df_bench, benchmark)]:
        if isinstance(d.columns, pd.MultiIndex):
            try:
                d = d.xs(s, axis=1, level=1)
            except Exception:
                pass

    # Kapanış ve Temel Kolonlar
    asset_close = df_asset['Close'].iloc[:, 0] if isinstance(df_asset['Close'], pd.DataFrame) else df_asset['Close']
    bench_close = df_bench['Close'].iloc[:, 0] if isinstance(df_bench['Close'], pd.DataFrame) else df_bench['Close']
    
    df = pd.DataFrame({
        'Open': df_asset['Open'].squeeze(),
        'High': df_asset['High'].squeeze(),
        'Low': df_asset['Low'].squeeze(),
        'Close': asset_close,
        'Volume': df_asset['Volume'].squeeze(),
        'Bench_Close': bench_close
    }).dropna()

    # 2. Teknik ve Trend İndikatörleri
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['dist_sma50'] = (df['Close'] - df['SMA_50']) / (df['SMA_50'] + 1e-8)
    df['dist_sma200'] = (df['Close'] - df['SMA_200']) / (df['SMA_200'] + 1e-8)
    df['sma50_200_ratio'] = (df['SMA_50'] - df['SMA_200']) / (df['SMA_200'] + 1e-8)
    
    # Bollinger Bantları (20)
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    
    # MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = macd_hist

    # Sektörel Göreceli Getiri (Alpha) ve Beta
    ret_asset = df['Close'].pct_change()
    ret_bench = df['Bench_Close'].pct_change()
    df['alpha_1d'] = ret_asset - ret_bench
    df['alpha_20d_cum'] = df['alpha_1d'].rolling(window=20).sum()
    
    cov = ret_asset.rolling(window=20).cov(ret_bench)
    var = ret_bench.rolling(window=20).var()
    df['beta_20d'] = (cov / (var + 1e-8)).clip(-3.0, 3.0)

    # Hacim Oranı (20 günlük hacim ortalamasına oran)
    vol_sma20 = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / (vol_sma20 + 1e-8)

    # 3. TradingView Lightweight Charts Formatı
    candles: List[Dict[str, Any]] = []
    sma50_series: List[Dict[str, Any]] = []
    sma200_series: List[Dict[str, Any]] = []
    bb_upper_series: List[Dict[str, Any]] = []
    bb_lower_series: List[Dict[str, Any]] = []
    macd_series: List[Dict[str, Any]] = []
    volume_series: List[Dict[str, Any]] = []

    for date_idx, row in df.iterrows():
        t_str = date_idx.strftime("%Y-%m-%d")
        o, h, l, c, v = float(row['Open']), float(row['High']), float(row['Low']), float(row['Close']), float(row['Volume'])
        
        candles.append({
            "time": t_str,
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2)
        })
        
        volume_series.append({
            "time": t_str,
            "value": round(v, 2),
            "color": "#10B981" if c >= o else "#EF4444"
        })

        if not np.isnan(row['SMA_50']):
            sma50_series.append({"time": t_str, "value": round(float(row['SMA_50']), 2)})
        if not np.isnan(row['SMA_200']):
            sma200_series.append({"time": t_str, "value": round(float(row['SMA_200']), 2)})
        if not np.isnan(row['BB_Upper']):
            bb_upper_series.append({"time": t_str, "value": round(float(row['BB_Upper']), 2)})
        if not np.isnan(row['BB_Lower']):
            bb_lower_series.append({"time": t_str, "value": round(float(row['BB_Lower']), 2)})
        if not np.isnan(row['MACD']):
            macd_series.append({
                "time": t_str,
                "macd": round(float(row['MACD']), 2),
                "signal": round(float(row['MACD_Signal']), 2),
                "hist": round(float(row['MACD_Hist']), 2)
            })

    # Son Durum Metrikleri
    last_row = df.iloc[-1]
    prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else float(last_row['Close'])
    last_close = float(last_row['Close'])
    change_pct = round(((last_close - prev_close) / prev_close) * 100, 2)
    
    is_golden_cross = bool(last_row['sma50_200_ratio'] > 0)
    is_above_sma200 = bool(last_row['dist_sma200'] > 0)

    result = {
        "ticker": clean_ticker,
        "benchmark": benchmark,
        "current_price": round(last_close, 2),
        "change_pct": change_pct,
        "volume": float(last_row['Volume']),
        "volume_ratio": round(float(last_row['volume_ratio']), 2) if not np.isnan(last_row['volume_ratio']) else 1.0,
        "beta": round(float(last_row['beta_20d']), 2) if not np.isnan(last_row['beta_20d']) else 1.0,
        "alpha_20d_cum": round(float(last_row['alpha_20d_cum']) * 100, 2) if not np.isnan(last_row['alpha_20d_cum']) else 0.0,
        "dist_sma50_pct": round(float(last_row['dist_sma50']) * 100, 2) if not np.isnan(last_row['dist_sma50']) else 0.0,
        "dist_sma200_pct": round(float(last_row['dist_sma200']) * 100, 2) if not np.isnan(last_row['dist_sma200']) else 0.0,
        "is_golden_cross": is_golden_cross,
        "is_above_sma200": is_above_sma200,
        "candles": candles,
        "volume_series": volume_series,
        "sma50": sma50_series,
        "sma200": sma200_series,
        "bb_upper": bb_upper_series,
        "bb_lower": bb_lower_series,
        "macd": macd_series
    }

    cache.set(cache_key, result)
    return result


TRACKED_UNIVERSE = [
    {"ticker": "NVDA", "name": "Nvidia Corporation", "sector": "Yarı İletken & AI", "category": "Tech", "last_close": 230.36, "change_pct": 0.84, "dist_sma200_pct": 17.22, "is_golden_cross": True, "alpha_20d_cum": 5.94, "beta": 1.21, "ai_signal": "GÜÇLÜ AL", "confidence_score": 88.5, "volume_ratio": 1.05},
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Tüketici Elektroniği", "category": "Tech", "last_close": 224.50, "change_pct": -0.32, "dist_sma200_pct": 6.80, "is_golden_cross": True, "alpha_20d_cum": 1.45, "beta": 0.95, "ai_signal": "AL", "confidence_score": 72.0, "volume_ratio": 0.92},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Bulut & Kurumsal AI", "category": "Tech", "last_close": 418.20, "change_pct": 1.15, "dist_sma200_pct": 9.40, "is_golden_cross": True, "alpha_20d_cum": 3.80, "beta": 1.02, "ai_signal": "AL", "confidence_score": 79.5, "volume_ratio": 1.12},
    {"ticker": "GOOGL", "name": "Alphabet Inc. (Google)", "sector": "Arama & Bulut / AI", "category": "Tech", "last_close": 178.40, "change_pct": 0.65, "dist_sma200_pct": 8.10, "is_golden_cross": True, "alpha_20d_cum": 2.90, "beta": 1.05, "ai_signal": "AL", "confidence_score": 75.0, "volume_ratio": 1.02},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "E-Ticaret & AWS Bulut", "category": "Tech", "last_close": 186.20, "change_pct": 1.40, "dist_sma200_pct": 11.30, "is_golden_cross": True, "alpha_20d_cum": 4.10, "beta": 1.14, "ai_signal": "GÜÇLÜ AL", "confidence_score": 82.0, "volume_ratio": 1.15},
    {"ticker": "META", "name": "Meta Platforms Inc.", "sector": "Sosyal Medya & Llama AI", "category": "Tech", "last_close": 515.80, "change_pct": 2.10, "dist_sma200_pct": 19.50, "is_golden_cross": True, "alpha_20d_cum": 8.70, "beta": 1.25, "ai_signal": "GÜÇLÜ AL", "confidence_score": 86.0, "volume_ratio": 1.22},
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Otomotiv & Robotik / FSD", "category": "Tech", "last_close": 242.80, "change_pct": -2.40, "dist_sma200_pct": -4.20, "is_golden_cross": False, "alpha_20d_cum": -6.10, "beta": 1.65, "ai_signal": "AZALT", "confidence_score": 64.0, "volume_ratio": 1.35},
    {"ticker": "AMD", "name": "Advanced Micro Devices", "sector": "Yarı İletken & Sunucu", "category": "Semis", "last_close": 154.60, "change_pct": 1.85, "dist_sma200_pct": 12.40, "is_golden_cross": True, "alpha_20d_cum": 6.20, "beta": 1.45, "ai_signal": "AL", "confidence_score": 78.0, "volume_ratio": 1.10},
    {"ticker": "AVGO", "name": "Broadcom Inc.", "sector": "Özel AI Çipleri & Ağ", "category": "Semis", "last_close": 162.30, "change_pct": 2.45, "dist_sma200_pct": 21.00, "is_golden_cross": True, "alpha_20d_cum": 9.10, "beta": 1.30, "ai_signal": "GÜÇLÜ AL", "confidence_score": 87.5, "volume_ratio": 1.18},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Yatırım Bankacılığı & Finans", "category": "Finance", "last_close": 212.50, "change_pct": 0.75, "dist_sma200_pct": 14.80, "is_golden_cross": True, "alpha_20d_cum": 4.80, "beta": 0.88, "ai_signal": "GÜÇLÜ AL", "confidence_score": 83.0, "volume_ratio": 0.95},
    {"ticker": "LLY", "name": "Eli Lilly and Company", "sector": "Biyofarma & Sağlık", "category": "Healthcare", "last_close": 948.00, "change_pct": 1.10, "dist_sma200_pct": 24.50, "is_golden_cross": True, "alpha_20d_cum": 10.50, "beta": 0.72, "ai_signal": "GÜÇLÜ AL", "confidence_score": 88.0, "volume_ratio": 1.04},
    {"ticker": "BTC-USD", "name": "Bitcoin (USD)", "sector": "Kripto / Dijital Altın", "category": "Crypto", "last_close": 64250.00, "change_pct": 2.10, "dist_sma200_pct": 8.15, "is_golden_cross": True, "alpha_20d_cum": 4.50, "beta": 1.10, "ai_signal": "AL", "confidence_score": 76.5, "volume_ratio": 1.18},
    {"ticker": "ETH-USD", "name": "Ethereum (USD)", "sector": "Akıllı Sözleşme Platformu", "category": "Crypto", "last_close": 2487.38, "change_pct": -1.08, "dist_sma200_pct": 21.84, "is_golden_cross": True, "alpha_20d_cum": 7.08, "beta": 1.38, "ai_signal": "NÖTR", "confidence_score": 68.3, "volume_ratio": 0.75},
    {"ticker": "SOL-USD", "name": "Solana (USD)", "sector": "Yüksek Hızlı Katman-1", "category": "Crypto", "last_close": 152.40, "change_pct": 4.35, "dist_sma200_pct": 14.60, "is_golden_cross": True, "alpha_20d_cum": 12.20, "beta": 1.85, "ai_signal": "GÜÇLÜ AL", "confidence_score": 85.0, "volume_ratio": 1.62}
]

def fetch_screener_data() -> List[Dict[str, Any]]:
    """
    Screener tablosu için piyasa genel bakış ve AI puanlama verilerini döner.
    """
    return TRACKED_UNIVERSE

