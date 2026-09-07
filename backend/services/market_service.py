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
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Otomotiv & Robotik", "category": "Tech", "last_close": 242.80, "change_pct": -2.40, "dist_sma200_pct": -4.20, "is_golden_cross": False, "alpha_20d_cum": -6.10, "beta": 1.65, "ai_signal": "AZALT", "confidence_score": 64.0, "volume_ratio": 1.35},
    {"ticker": "BTC-USD", "name": "Bitcoin (USD)", "sector": "Kripto / Dijital Altın", "category": "Crypto", "last_close": 64250.00, "change_pct": 2.10, "dist_sma200_pct": 8.15, "is_golden_cross": True, "alpha_20d_cum": 4.50, "beta": 1.10, "ai_signal": "AL", "confidence_score": 76.5, "volume_ratio": 1.18},
    {"ticker": "ETH-USD", "name": "Ethereum (USD)", "sector": "Akıllı Sözleşme Platformu", "category": "Crypto", "last_close": 2487.38, "change_pct": -1.08, "dist_sma200_pct": 21.84, "is_golden_cross": True, "alpha_20d_cum": 7.08, "beta": 1.38, "ai_signal": "NÖTR", "confidence_score": 68.3, "volume_ratio": 0.75},
    {"ticker": "SOL-USD", "name": "Solana (USD)", "sector": "Yüksek Hızlı Katman-1", "category": "Crypto", "last_close": 152.40, "change_pct": 4.35, "dist_sma200_pct": 14.60, "is_golden_cross": True, "alpha_20d_cum": 12.20, "beta": 1.85, "ai_signal": "GÜÇLÜ AL", "confidence_score": 85.0, "volume_ratio": 1.62},
    {"ticker": "THYAO.IS", "name": "Türk Hava Yolları", "sector": "Havacılık & Lojistik", "category": "BIST", "last_close": 296.50, "change_pct": 0.17, "dist_sma200_pct": -2.07, "is_golden_cross": True, "alpha_20d_cum": -4.63, "beta": 0.78, "ai_signal": "AL", "confidence_score": 74.2, "volume_ratio": 0.72},
    {"ticker": "EREGL.IS", "name": "Ereğli Demir Çelik", "sector": "Demir Çelik & Metal", "category": "BIST", "last_close": 48.20, "change_pct": 1.26, "dist_sma200_pct": 3.10, "is_golden_cross": True, "alpha_20d_cum": 2.15, "beta": 0.85, "ai_signal": "AL", "confidence_score": 71.0, "volume_ratio": 1.08},
    {"ticker": "ASELS.IS", "name": "Aselsan Elektronik", "sector": "Savunma Sanayi & Teknoloji", "category": "BIST", "last_close": 62.90, "change_pct": 3.45, "dist_sma200_pct": 18.90, "is_golden_cross": True, "alpha_20d_cum": 9.40, "beta": 1.15, "ai_signal": "GÜÇLÜ AL", "confidence_score": 89.0, "volume_ratio": 1.80},
    {"ticker": "BIMAS.IS", "name": "BİM Birleşik Mağazalar", "sector": "Perakende & Defansif", "category": "BIST", "last_close": 510.00, "change_pct": -0.58, "dist_sma200_pct": 5.40, "is_golden_cross": True, "alpha_20d_cum": 0.80, "beta": 0.65, "ai_signal": "NÖTR", "confidence_score": 69.5, "volume_ratio": 0.88},
    {"ticker": "GARAN.IS", "name": "Garanti BBVA", "sector": "Bankacılık & Finans", "category": "BIST", "last_close": 118.40, "change_pct": 2.25, "dist_sma200_pct": 22.30, "is_golden_cross": True, "alpha_20d_cum": 8.10, "beta": 1.25, "ai_signal": "GÜÇLÜ AL", "confidence_score": 86.4, "volume_ratio": 1.45}
]

def fetch_screener_data() -> List[Dict[str, Any]]:
    """
    Screener tablosu için piyasa genel bakış ve AI puanlama verilerini döner.
    """
    return TRACKED_UNIVERSE

