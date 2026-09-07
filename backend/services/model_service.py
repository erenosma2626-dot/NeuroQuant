"""
Google TimesFM 3.0 Foundation Model & Quantile LightGBM AI Inference Service
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from backend.config import settings
from backend.services.cache_service import cache
from neuro_modules.quant_benchmark import (
    fetch_asset_and_benchmark,
    compute_quant_features,
    QuantileLightGBMCluster
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "clusters")

# Bellekte model önbellekleri
_LOADED_MODELS: Dict[str, QuantileLightGBMCluster] = {}
_TIMESFM_FORECASTER = None
_TIMESFM_FAILED = False

def get_timesfm_forecaster():
    """Google TimesFM 3.0 Foundation Modelini belleğe yükler (Singleton)"""
    global _TIMESFM_FORECASTER, _TIMESFM_FAILED
    if _TIMESFM_FAILED:
        return None
    if _TIMESFM_FORECASTER is not None:
        return _TIMESFM_FORECASTER
    try:
        import timesfm
        try:
            _TIMESFM_FORECASTER = timesfm.TimesFM3Forecaster.from_pretrained(
                "google/timesfm-3.0-pytorch", local_files_only=True
            )
        except Exception:
            _TIMESFM_FORECASTER = timesfm.TimesFM3Forecaster.from_pretrained(
                "google/timesfm-3.0-pytorch"
            )
        return _TIMESFM_FORECASTER
    except Exception as e:
        print(f"⚠️ TimesFM 3.0 yüklenemedi, LightGBM yedeği kullanılacak: {e}")
        _TIMESFM_FAILED = True
        return None

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
    5 Günlük Fiyat ve Olasılık Tahmini Üretir.
    Motor Seçenekleri:
      - 'timesfm': Google TimesFM 3.0 Zero-Shot Foundation Model
      - 'lightgbm': Quantile LightGBM Pinball Regresyon Kümesi
      - 'hybrid': TimesFM + LightGBM Hibrit Konsensüsü
    """
    clean_ticker = ticker.strip().upper()
    valid_engines = ["timesfm", "lightgbm", "hybrid"]
    if engine not in valid_engines:
        engine = "timesfm"

    cache_key = f"forecast_{clean_ticker}_{engine}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    benchmark = settings.get_benchmark(clean_ticker)
    df_raw = fetch_asset_and_benchmark(clean_ticker, benchmark, period="5y")
    df_feat = compute_quant_features(df_raw)
    
    last_price = float(df_feat['Close'].iloc[-1])
    last_date = df_feat.index[-1]
    
    # ── 1. Motor Seçimi ve Tahmin Hesaplama ──
    actual_engine = engine
    used_features = ["Close Time-Series (TimesFM 3.0 Zero-Shot)"]
    
    # TimesFM'i dene
    tfm_model = None
    if engine in ["timesfm", "hybrid"]:
        tfm_model = get_timesfm_forecaster()
        if tfm_model is None and engine == "timesfm":
            actual_engine = "lightgbm"

    # LightGBM hesapla (lightgbm istendiyse veya hybrid/fallback gerekliyse)
    lgb_q10, lgb_q50, lgb_q90, lgb_up_prob = None, None, None, None
    lgb_model = None
    if engine in ["lightgbm", "hybrid"] or actual_engine == "lightgbm":
        lgb_model = get_cluster_model(clean_ticker)
        last_features = df_feat.iloc[[-1]]
        preds = lgb_model.predict_cone(last_features)
        lgb_q10 = float(preds['lower_80'][0])
        lgb_q50 = float(preds['median'][0])
        lgb_q90 = float(preds['upper_80'][0])
        lgb_up_prob = float(preds['up_probability'][0])
        used_features = lgb_model.feature_cols

    # TimesFM 3.0 tahmin üretimi
    tfm_median_prices = None
    tfm_q10_prices = None
    tfm_q90_prices = None
    
    if tfm_model is not None and engine in ["timesfm", "hybrid"]:
        try:
            # Son 90 günlük kapanış fiyatı serisi
            close_series = df_feat['Close'].values[-90:].flatten().astype(np.float32)
            res3 = tfm_model.predict(close_series, horizon=5, return_quantiles=True)
            tfm_median_prices = [float(p) for p in res3.forecast]
            tfm_q10_prices = [float(p) for p in res3.quantiles[:, 0]]
            tfm_q90_prices = [float(p) for p in res3.quantiles[:, -1]]
        except Exception as e:
            print(f"⚠️ TimesFM predict hatası, Quantile motoruna geçiliyor: {e}")
            tfm_median_prices = None
            if actual_engine == "timesfm":
                actual_engine = "lightgbm"

    # ── 2. Güven Konisi Çizgilerini ve Persentilleri Oluştur ──
    cone_series: List[Dict[str, Any]] = []
    curr_d = last_date

    if actual_engine == "timesfm" and tfm_median_prices is not None:
        engine_label = "Google TimesFM 3.0 Foundation Model"
        used_features = ["Zaman Serisi Dikkat Mekanizması (TimesFM 3.0)", "Variate Attention", "Zero-Shot Güven Konisi"]
        
        # 5. Gün getirileri
        q50 = (tfm_median_prices[-1] - last_price) / last_price
        q10 = (tfm_q10_prices[-1] - last_price) / last_price
        q90 = (tfm_q90_prices[-1] - last_price) / last_price
        
        # Yükseliş olasılığı: 5. günün quantiles serisindeki pozitiflik oranı
        all_q_day5 = res3.quantiles[-1]
        up_count = np.sum(all_q_day5 > last_price)
        up_prob = float(up_count / len(all_q_day5))
        
        for step in range(1, 6):
            curr_d = curr_d + timedelta(days=1)
            while curr_d.weekday() >= 5:
                curr_d = curr_d + timedelta(days=1)
                
            p_med = tfm_median_prices[step - 1]
            p_low = tfm_q10_prices[step - 1]
            p_upp = tfm_q90_prices[step - 1]
            step_ret = ((p_med - last_price) / last_price) * 100
            
            cone_series.append({
                "step": step,
                "date": curr_d.strftime("%Y-%m-%d"),
                "median_price": round(p_med, 2),
                "lower_80_price": round(p_low, 2),
                "upper_80_price": round(p_upp, 2),
                "median_return_pct": round(step_ret, 2)
            })

    elif actual_engine == "hybrid" and tfm_median_prices is not None and lgb_q50 is not None:
        engine_label = "Google TimesFM 3.0 + Quantile LightGBM Hibrit"
        used_features = ["TimesFM 3.0 Temporal Encoder"] + lgb_model.feature_cols[:4]
        
        # TimesFM ve LightGBM persentil getirilerini %50-%50 harmanla
        tfm_q50 = (tfm_median_prices[-1] - last_price) / last_price
        tfm_q10 = (tfm_q10_prices[-1] - last_price) / last_price
        tfm_q90 = (tfm_q90_prices[-1] - last_price) / last_price
        
        q50 = 0.5 * tfm_q50 + 0.5 * lgb_q50
        q10 = 0.5 * tfm_q10 + 0.5 * lgb_q10
        q90 = 0.5 * tfm_q90 + 0.5 * lgb_q90
        
        all_q_day5 = res3.quantiles[-1]
        tfm_up = float(np.sum(all_q_day5 > last_price) / len(all_q_day5))
        up_prob = 0.5 * tfm_up + 0.5 * lgb_up_prob
        
        for step in range(1, 6):
            curr_d = curr_d + timedelta(days=1)
            while curr_d.weekday() >= 5:
                curr_d = curr_d + timedelta(days=1)
                
            ratio = step / 5.0
            lgb_med = last_price * (1.0 + lgb_q50 * ratio)
            lgb_low = last_price * (1.0 + lgb_q10 * ratio)
            lgb_upp = last_price * (1.0 + lgb_q90 * ratio)
            
            p_med = 0.5 * tfm_median_prices[step - 1] + 0.5 * lgb_med
            p_low = 0.5 * tfm_q10_prices[step - 1] + 0.5 * lgb_low
            p_upp = 0.5 * tfm_q90_prices[step - 1] + 0.5 * lgb_upp
            step_ret = ((p_med - last_price) / last_price) * 100
            
            cone_series.append({
                "step": step,
                "date": curr_d.strftime("%Y-%m-%d"),
                "median_price": round(p_med, 2),
                "lower_80_price": round(p_low, 2),
                "upper_80_price": round(p_upp, 2),
                "median_return_pct": round(step_ret, 2)
            })

    else:
        # Quantile LightGBM
        engine_label = "Quantile LightGBM Çok-Faktörlü Model"
        q10 = lgb_q10
        q50 = lgb_q50
        q90 = lgb_q90
        up_prob = lgb_up_prob
        
        for step in range(1, 6):
            curr_d = curr_d + timedelta(days=1)
            while curr_d.weekday() >= 5:
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

    # ── 3. Karar ve Öneri Metni ──
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

    result = {
        "ticker": clean_ticker,
        "benchmark": benchmark,
        "engine": engine_label,
        "engine_type": actual_engine,
        "current_price": round(last_price, 2),
        "as_of_date": last_date.strftime("%Y-%m-%d"),
        "median_5d_return_pct": round(q50 * 100, 2),
        "lower_80_return_pct": round(q10 * 100, 2),
        "upper_80_return_pct": round(q90 * 100, 2),
        "up_probability": round(up_prob * 100, 1),
        "decision": decision,
        "decision_color": color,
        "cone_series": cone_series,
        "features_used": used_features,
        "available_engines": [
            {"id": "timesfm", "name": "Google TimesFM 3.0", "badge": "Foundation Model"},
            {"id": "lightgbm", "name": "Quantile LightGBM", "badge": "Çok-Faktörlü"},
            {"id": "hybrid", "name": "Hibrit Konsensüs", "badge": "Ensemble"}
        ]
    }

    cache.set(cache_key, result, ttl=900)
    return result
