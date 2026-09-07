"""
AI Model Inference & Quantile Confidence Cone Service
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

def generate_forecast(ticker: str) -> Dict[str, Any]:
    clean_ticker = ticker.strip().upper()
    cache_key = f"forecast_{clean_ticker}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    benchmark = settings.get_benchmark(clean_ticker)
    df_raw = fetch_asset_and_benchmark(clean_ticker, benchmark, period="5y")
    df_feat = compute_quant_features(df_raw)
    
    model = get_cluster_model(clean_ticker)
    last_features = df_feat.iloc[[-1]]
    
    preds = model.predict_cone(last_features)
    
    q10 = float(preds['lower_80'][0])
    q50 = float(preds['median'][0])
    q90 = float(preds['upper_80'][0])
    up_prob = float(preds['up_probability'][0])
    
    last_price = float(df_feat['Close'].iloc[-1])
    last_date = df_feat.index[-1]
    
    # 5 Günlük Güven Konisi Çizgileri
    cone_series: List[Dict[str, Any]] = []
    
    curr_d = last_date
    for step in range(1, 6):
        # İş günü ilerletme
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
    if q50 > 0.02 and up_prob > 0.65:
        decision = "GÜÇLÜ AL"
        color = "#10B981"
    elif q50 > 0.005:
        decision = "AL (Teknik Teyitli)"
        color = "#3B82F6"
    elif q50 < -0.02 and up_prob < 0.35:
        decision = "SAT / NAKİT"
        color = "#EF4444"
    else:
        decision = "İZLE / NÖTR"
        color = "#94A3B8"

    result = {
        "ticker": clean_ticker,
        "benchmark": benchmark,
        "current_price": round(last_price, 2),
        "as_of_date": last_date.strftime("%Y-%m-%d"),
        "median_5d_return_pct": round(q50 * 100, 2),
        "lower_80_return_pct": round(q10 * 100, 2),
        "upper_80_return_pct": round(q90 * 100, 2),
        "up_probability": round(up_prob * 100, 1),
        "decision": decision,
        "decision_color": color,
        "cone_series": cone_series,
        "features_used": model.feature_cols
    }

    cache.set(cache_key, result, ttl=900)
    return result
