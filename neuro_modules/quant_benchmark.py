"""
NeuroQuant 3.0: Quantitative Benchmarking & Multi-Asset Alpha Engine
====================================================================
Bu modül; Tech, Kripto ve BIST varlık kümeleri için sektörel göreceli alfa (Relative Strength),
Purged Time-Wall doğrulama, Olasılıksal Güven Konisi ve kurumsal backtest metriklerini hesaplar.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# --- KÜME & REFERANS TANIMLARI ---
CLUSTERS = {
    "tech": {
        "name": "US Tech & Semiconductors",
        "benchmark": "SMH",  # Yarı iletken ETF
        "tickers": ["NVDA", "AAPL", "AMD", "MSFT"]
    },
    "crypto": {
        "name": "Crypto 24/7",
        "benchmark": "BTC-USD",  # Lider kripto varlık
        "tickers": ["BTC-USD", "ETH-USD", "SOL-USD"]
    },
    "bist": {
        "name": "Borsa Istanbul Macro",
        "benchmark": "XU100.IS",  # BIST 100 Endeksi
        "tickers": ["THYAO.IS", "EREGL.IS", "ASELS.IS"]
    }
}

LOOKBACK_WINDOW = 60
HORIZON = 5
TEST_DAYS = 90  # TIME-WALL: Son 90 gün kör test seti

def fetch_asset_and_benchmark(ticker: str, benchmark: str, period="5y") -> pd.DataFrame:
    """Hisse ve referans varlığın OHLCV verilerini indirip senkronize eder."""
    print(f"📡 Veri çekiliyor: {ticker} (Referans: {benchmark})...")
    df_asset = yf.download(ticker, period=period, progress=False)
    df_bench = yf.download(benchmark, period=period, progress=False)
    
    # Sütun düzleştirme (MultiIndex koruması)
    for d, s in [(df_asset, ticker), (df_bench, benchmark)]:
        if isinstance(d.columns, pd.MultiIndex):
            try: d = d.xs(s, axis=1, level=1)
            except: pass
            
    # Temiz kapanışlar
    asset_close = df_asset['Close'].copy()
    bench_close = df_bench['Close'].copy()
    if isinstance(asset_close, pd.DataFrame): asset_close = asset_close.iloc[:, 0]
    if isinstance(bench_close, pd.DataFrame): bench_close = bench_close.iloc[:, 0]

    combined = pd.DataFrame({
        'Open': df_asset['Open'].squeeze(),
        'High': df_asset['High'].squeeze(),
        'Low': df_asset['Low'].squeeze(),
        'Close': asset_close,
        'Volume': df_asset['Volume'].squeeze(),
        'Bench_Close': bench_close
    }).dropna()
    
    return combined

def compute_quant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Teknik indikatörler ve Sektörel Göreceli Alfa öznitelikleri üretir."""
    data = df.copy()
    
    # 1. Ham Getiriler
    data['ret_1d'] = data['Close'].pct_change()
    data['bench_ret_1d'] = data['Bench_Close'].pct_change()
    
    # 2. SEKTÖREL GÖRECELİ ALFA (Relative Strength)
    # Hissenin sektöründen pozitif/negatif ayrışması
    data['alpha_1d'] = data['ret_1d'] - data['bench_ret_1d']
    data['alpha_5d_cum'] = data['alpha_1d'].rolling(window=5).sum()
    data['alpha_20d_cum'] = data['alpha_1d'].rolling(window=20).sum()
    
    # Göreceli Beta (20 günlük kovaryans / varyans)
    cov = data['ret_1d'].rolling(window=20).cov(data['bench_ret_1d'])
    var = data['bench_ret_1d'].rolling(window=20).var()
    data['beta_20d'] = cov / (var + 1e-8)
    data['beta_20d'] = data['beta_20d'].clip(-3.0, 3.0)
    
    # 3. UZUN VADELİ TREND REJİMLERİ (SMA 50, SMA 200, Golden/Death Cross)
    # (Kullanıcı tercihi doğrultusunda yanıltıcı kısa vadeli RSI kaldırıldı)
    data['sma_50'] = data['Close'].rolling(window=50).mean()
    data['sma_200'] = data['Close'].rolling(window=200).mean()
    # Fiyatın SMA 50 ve SMA 200'e olan normalize yüzdesel mesafesi
    data['dist_sma50'] = (data['Close'] - data['sma_50']) / (data['sma_50'] + 1e-8)
    data['dist_sma200'] = (data['Close'] - data['sma_200']) / (data['sma_200'] + 1e-8)
    # Golden Cross Rejimi (50 günlük ortalamanın 200 günlük ortalamaya oranı / spread)
    data['sma50_200_ratio'] = (data['sma_50'] - data['sma_200']) / (data['sma_200'] + 1e-8)
    
    # MACD Farkı (Momentum)
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    data['MACD_diff'] = macd - signal
    
    # Normalize Bollinger Mesafesi: (Close - Lower) / (Upper - Lower)
    sma20 = data['Close'].rolling(window=20).mean()
    std20 = data['Close'].rolling(window=20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    data['BB_pos'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    data['BB_pos'] = data['BB_pos'].clip(-0.5, 1.5)
    
    # Parkinson Volatilitesi (High/Low tabanlı saf oynaklık)
    data['parkinson_vol'] = np.sqrt(1 / (4 * np.log(2)) * (np.log(data['High'] / data['Low']) ** 2))
    
    # 4. Hedef Değişken (5 Günlük Kümülatif Getiri)
    data['target_5d'] = data['Close'].shift(-HORIZON) / data['Close'] - 1.0
    
    return data.dropna()

def institutional_metrics(actual_returns: np.ndarray, predicted_direction: np.ndarray, risk_free_rate=0.04) -> dict:
    """
    Kurumsal düzeyde kantitatif performans metriklerini hesaplar.
    """
    # Yön Doğruluğu (Win Rate)
    correct = np.sign(predicted_direction) == np.sign(actual_returns)
    win_rate = float(np.mean(correct) * 100) if len(correct) > 0 else 0.0
    
    # Strateji Getirileri (Tahmin pozitifse uzun git, negatifse nakitte kal)
    strategy_rets = np.where(predicted_direction > 0, actual_returns, 0.0)
    
    # Sharpe Oranı (Yıllıklandırılmış 252 gün)
    rf_daily = risk_free_rate / 252
    excess_rets = strategy_rets - rf_daily
    std_ret = np.std(strategy_rets)
    sharpe = float(np.mean(excess_rets) / (std_ret + 1e-8) * np.sqrt(252)) if std_ret > 0 else 0.0
    
    # Sortino Oranı (Sadece negatif volatilite)
    downside = strategy_rets[strategy_rets < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-8
    sortino = float(np.mean(excess_rets) / (downside_std + 1e-8) * np.sqrt(252))
    
    # Kümülatif Getiri & Max Drawdown
    cum_rets = np.cumprod(1 + strategy_rets)
    running_max = np.maximum.accumulate(cum_rets)
    drawdowns = (running_max - cum_rets) / running_max
    max_dd = float(np.max(drawdowns) * 100) if len(drawdowns) > 0 else 0.0
    
    # Profit Factor
    gains = strategy_rets[strategy_rets > 0].sum()
    losses = np.abs(strategy_rets[strategy_rets < 0].sum())
    profit_factor = float(gains / (losses + 1e-8)) if losses > 0 else float(gains)
    
    return {
        "Win_Rate_%": round(win_rate, 2),
        "Sharpe_Ratio": round(sharpe, 2),
        "Sortino_Ratio": round(sortino, 2),
        "Max_Drawdown_%": round(max_dd, 2),
        "Profit_Factor": round(profit_factor, 2),
        "Total_Strategy_Return_%": round(float((cum_rets[-1] - 1) * 100), 2) if len(cum_rets) > 0 else 0.0,
        "Buy_Hold_Return_%": round(float((np.cumprod(1 + actual_returns)[-1] - 1) * 100), 2) if len(actual_returns) > 0 else 0.0
    }

class QuantileLightGBMCluster:
    """
    Sektör duyarlı, Olasılıksal Güven Konisi üreten LightGBM Küme Modeli.
    10. persentil (alt bant), 50. persentil (medyan) ve 90. persentil (üst bant) üretir.
    """
    def __init__(self, cluster_name: str):
        self.cluster_name = cluster_name
        self.feature_cols = [
            'ret_1d', 'bench_ret_1d', 'alpha_1d', 'alpha_5d_cum', 'alpha_20d_cum',
            'beta_20d', 'dist_sma50', 'dist_sma200', 'sma50_200_ratio',
            'MACD_diff', 'BB_pos', 'parkinson_vol'
        ]
        self.model_10 = lgb.LGBMRegressor(objective='quantile', alpha=0.10, n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        self.model_50 = lgb.LGBMRegressor(objective='quantile', alpha=0.50, n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        self.model_90 = lgb.LGBMRegressor(objective='quantile', alpha=0.90, n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model_10.fit(X[self.feature_cols], y)
        self.model_50.fit(X[self.feature_cols], y)
        self.model_90.fit(X[self.feature_cols], y)
        return self
        
    def predict_cone(self, X: pd.DataFrame) -> dict:
        """Medyan, Alt %80 ve Üst %80 güven konilerini döndürür."""
        q10 = self.model_10.predict(X[self.feature_cols])
        q50 = self.model_50.predict(X[self.feature_cols])
        q90 = self.model_90.predict(X[self.feature_cols])
        
        # Olasılık: Medyanın pozitif olma derecesi (Sigmoid türevi)
        # q50 pozitif ve alt bant da sıfıra yakınsa olasılık yükselir
        up_prob = 1 / (1 + np.exp(-15 * q50))
        
        return {
            "lower_80": q10,
            "median": q50,
            "upper_80": q90,
            "up_probability": np.clip(up_prob, 0.05, 0.95)
        }
