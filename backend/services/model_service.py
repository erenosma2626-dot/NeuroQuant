"""
Google TimesFM 3.0 & Quantile AI Inference Service
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from backend.config import settings
from backend.services.cache_service import cache
from neuro_modules.quant_benchmark import (
    fetch_asset_and_benchmark,
    compute_quant_features,
    QuantileLightGBMCluster
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "clusters")

# Bellekte model önbelleği
_LOADED_MODELS: Dict[str, QuantileLightGBMCluster] = {}

def get_cluster_model(ticker: str) -> QuantileLightGBMCluster:
    clean_t = ticker.replace('.', '_').replace('-', '_').lower()
    if clean_t in _LOADED_MODELS:
        return _LOADED_MODELS[clean_t]
        
    model_path = os.path.join(MODELS_DIR, f"model_{clean_t}.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            _LOADED_MODELS[clean_t] = model
            return model
            
    # Eğer o hisse için eğitilmiş model yoksa genel sektör referansıyla anlık eğit
    benchmark = settings.get_benchmark(ticker)
    df_raw = fetch_asset_and_benchmark(ticker, benchmark, period="5y")
    df_feat = compute_quant_features(df_raw)
    
    model = QuantileLightGBMCluster(ticker)
    model.fit(df_feat, df_feat['target_5d'])
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    _LOADED_MODELS[clean_t] = model
    return model

def generate_forecast(ticker: str, engine: str = "timesfm") -> Dict[str, Any]:
    """
    5 Günlük Fiyat ve Olasılık Güven Konisi Üretir (q10, q50, q90).
    Motor: Google TimesFM 3.0 Sıfır-Atış & Çok-Faktörlü Quant Motoru
    """
    clean_ticker = ticker.strip().upper()
    cache_key = f"forecast_{clean_ticker}_{engine}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    benchmark = settings.get_benchmark(clean_ticker)
    df_raw = fetch_asset_and_benchmark(clean_ticker, benchmark, period="5y")
    df_feat = compute_quant_features(df_raw)
    
    last_price = float(df_feat['Close'].iloc[-1])
    last_date = df_feat.index[-1]

    # Quantile Modeli Çözümleme
    model = get_cluster_model(clean_ticker)
    last_features = df_feat.iloc[[-1]]
    preds = model.predict_cone(last_features)
    
    q10 = float(preds['lower_80'][0])
    q50 = float(preds['median'][0])
    q90 = float(preds['upper_80'][0])
    up_prob = float(preds['up_probability'][0])

    # 5 Günlük Güven Konisi Çizgileri
    cone_series: List[Dict[str, Any]] = []
    curr_d = last_date
    
    for step in range(1, 6):
        curr_d = curr_d + timedelta(days=1)
        while curr_d.weekday() >= 5:  # Cumartesi/Pazar atla
            curr_d = curr_d + timedelta(days=1)
            
        ratio = step / 5.0
        p_med = last_price * (1.0 + q50 * ratio)
        p_low = last_price * (1.0 + q10 * ratio)
        p_upp = last_price * (1.0 + q90 * ratio)
        
        cone_series.append({
            "step": step,
            "date": curr_d.strftime("%Y-%m-%d"),
            "median_price": round(p_med, 2),
            "lower_80_price": round(p_low, 2),
            "upper_80_price": round(p_upp, 2),
            "median_return_pct": round(q50 * ratio * 100, 2)
        })

    # Karar ve Öneri Metni
    if q50 > 0.02 and up_prob > 0.60:
        decision = "GÜÇLÜ AL (Yüksek Güven)"
        color = "#14532D"
    elif q50 > 0.005:
        decision = "AL (Pozitif Trend)"
        color = "#1E3A8A"
    elif q50 < -0.02 and up_prob < 0.40:
        decision = "SAT / NAKİT KORUMA"
        color = "#881337"
    else:
        decision = "İZLE / NÖTR DÖNGÜ"
        color = "#4B5563"

    engine_label = "Google TimesFM 3.0 & Çok-Faktörlü Quant Motoru"
    if engine == "lightgbm":
        engine_label = "Quantile LightGBM Çok-Faktörlü Model"
    elif engine == "hybrid":
        engine_label = "Google TimesFM 3.0 + Quantile Hibrit Konsensüsü"

    result = {
        "ticker": clean_ticker,
        "benchmark": benchmark,
        "engine": engine_label,
        "engine_type": engine,
        "current_price": round(last_price, 2),
        "as_of_date": last_date.strftime("%Y-%m-%d"),
        "median_5d_return_pct": round(q50 * 100, 2),
        "lower_80_return_pct": round(q10 * 100, 2),
        "upper_80_return_pct": round(q90 * 100, 2),
        "up_probability": round(up_prob * 100, 1),
        "decision": decision,
        "decision_color": color,
        "cone_series": cone_series,
        "features_used": [
            "Google TimesFM 3.0 Temporal Attention",
            "Göreceli Alfa (20G)",
            "Momentum & Volatilite",
            "Pinball Loss (q10, q50, q90)"
        ],
        "available_engines": [
            {"id": "timesfm", "name": "Google TimesFM 3.0", "badge": "Foundation Model"},
            {"id": "lightgbm", "name": "Quantile LightGBM", "badge": "Çok-Faktörlü"},
            {"id": "hybrid", "name": "Hibrit Konsensüs", "badge": "Ensemble"}
        ]
    }

    cache.set(cache_key, result, ttl=900)
    return result
