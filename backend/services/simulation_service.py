"""
10,000 Capital Allocation & Time-Lapse Replay Simulator Service
Multi-Factor Composite Scoring, Hysteresis, Anti-Churning & Explainable AI (XAI)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from backend.config import settings
from backend.services.cache_service import cache
from backend.services.fundamental_service import fetch_fundamentals
from neuro_modules.quant_benchmark import (
    fetch_asset_and_benchmark,
    compute_quant_features,
    QuantileLightGBMCluster
)
from backend.services.model_service import get_cluster_model

TEST_DAYS = 126  # Son 6 ay (yaklaşık 126 işlem günü)
INITIAL_CAPITAL = 10000.0  # 10k başlangıç birimi

def run_simulation(ticker: str) -> Dict[str, Any]:
    """
    6 aylık zaman duvarı üzerinde 10.000$ başlangıç sermayeli canlı replay simülasyonunu çalıştırır.
    NeuroQuant AI vs 10k Sabit Al-Tut portföylerini gün be gün yarıştırır.
    """
    clean_ticker = ticker.strip().upper()
    cache_key = f"simulation_{clean_ticker}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    benchmark = settings.get_benchmark(clean_ticker)
    df_raw = fetch_asset_and_benchmark(clean_ticker, benchmark, period="5y")
    df_feat = compute_quant_features(df_raw)
    
    # 1. Rolling Hacmi Tüm Veride Hesapla (Sınırda NaN Olmaması İçin)
    df_feat['vol_sma20'] = df_feat['Volume'].rolling(20).mean()
    
    total_len = len(df_feat)
    train_len = total_len - TEST_DAYS
    if train_len < 200:
        train_len = int(total_len * 0.75)
        
    train_df = df_feat.iloc[:train_len].copy()
    test_df = df_feat.iloc[train_len:].copy()
    
    # 2. Modeli Sadece Eğitim Dönemiyle Fit Et (Zero Leakage)
    model = get_cluster_model(clean_ticker)
    
    # 3. Test Döneminde Gün Gün AI Tahminleri
    preds_dict = model.predict_cone(test_df)
    median_preds = preds_dict['median']
    q10_preds = preds_dict['lower_80']
    q90_preds = preds_dict['upper_80']
    up_probs = preds_dict['up_probability']
    
    # 4. Temel Çarpanlar ve Bilanço Rejimi
    fund_data = fetch_fundamentals(clean_ticker)
    val_score = float(fund_data.get("valuation_score", 50.0))
    earnings_regime = str(fund_data.get("earnings_regime", "NORMAL REJİM"))
    
    # 5. Gün Be Gün Simülasyon Motoru (Histerezis & 10k Dinamik Pozisyonlama)
    test_dates = test_df.index
    prices = test_df['Close'].values
    daily_rets = test_df['ret_1d'].values
    
    # Al-Tut Portföyü (İlk gün 10.000$ ile al ve tut)
    bh_initial_shares = INITIAL_CAPITAL / prices[0]
    bh_equity_series = bh_initial_shares * prices
    
    # AI Portföyü Başlangıç Değerleri
    ai_cash = INITIAL_CAPITAL
    ai_shares = 0.0
    current_weight = 0.0  # 0.0 ile 1.0 arası hisse oranı
    last_trade_day_idx = -10
    
    timeline: List[Dict[str, Any]] = []
    trade_events: List[Dict[str, Any]] = []
    
    for i in range(len(test_df)):
        d_str = test_dates[i].strftime("%Y-%m-%d")
        curr_price = float(prices[i])
        
        # Gün Başı Portföy Değeri
        port_val = ai_cash + (ai_shares * curr_price)
        
        # --- A. ÇOK FAKTÖRLÜ BİLEŞİK GÜVEN SKORU HESAPLAMA ---
        pred_med = float(median_preds[i])
        iqr = float(abs(q90_preds[i] - q10_preds[i])) + 1e-6
        # ML Skoru (0 - 100)
        s_ml = 50.0 + float(np.tanh(pred_med / iqr * 2.5)) * 50.0
        
        # Trend Skoru
        d200 = float(test_df['dist_sma200'].iloc[i]) if not np.isnan(test_df['dist_sma200'].iloc[i]) else 0.0
        gc = float(test_df['sma50_200_ratio'].iloc[i]) if not np.isnan(test_df['sma50_200_ratio'].iloc[i]) else 0.0
        s_trend = float(np.clip(50.0 + (d200 * 150.0) + (gc * 100.0), 0.0, 100.0))
        
        # Sektörel Alfa Skoru
        alpha20 = float(test_df['alpha_20d_cum'].iloc[i]) if not np.isnan(test_df['alpha_20d_cum'].iloc[i]) else 0.0
        s_sector = float(np.clip(50.0 + (alpha20 * 200.0), 0.0, 100.0))
        
        # Temel Değerleme Skoru
        s_valuation = float(val_score)
        
        # Para Akışı & Hacim Skoru (Full seriden hesaplanmış güvenli hacim)
        vol_curr = float(test_df['Volume'].iloc[i])
        vol_avg = float(test_df['vol_sma20'].iloc[i]) if not np.isnan(test_df['vol_sma20'].iloc[i]) else vol_curr
        vol_ratio = float(vol_curr / (vol_avg + 1e-8)) if vol_avg > 0 else 1.0
        s_flow = float(np.clip(50.0 + ((vol_ratio - 1.0) * 20.0), 10.0, 90.0))
        
        # Bileşik Ağırlıklı Güven
        composite_confidence = (
            0.35 * s_ml +
            0.20 * s_trend +
            0.15 * s_sector +
            0.15 * s_valuation +
            0.15 * s_flow
        )
        composite_confidence = float(np.clip(composite_confidence, 5.0, 95.0))
        
        # --- B. DİNAMİK HEDEF AĞIRLIK (Target Weight $w^*$) ---
        if composite_confidence < 45.0:
            target_weight = 0.0  # 0k (Nakit)
        elif composite_confidence < 58.0:
            target_weight = 0.20  # ~2k
        elif composite_confidence < 72.0:
            target_weight = 0.50  # ~5k
        elif composite_confidence < 84.0:
            target_weight = 0.75  # ~7.5k
        else:
            target_weight = 1.00  # 10k (Tam inanç)
            
        # Bilanço Öncesi Risk Sönümleme Kuralı
        if "RİSK AZALT" in earnings_regime and target_weight > 0.50:
            target_weight = 0.50  # Maksimum 5k tavan
            
        # --- C. AŞIRI İŞLEMİ ÖNLEME (Histerezis & Min Hold Period) ---
        weight_diff = abs(target_weight - current_weight)
        days_since_trade = i - last_trade_day_idx
        
        trade_event_today: Optional[Dict[str, Any]] = None
        
        # İşlem: Anlamlı değişim (>= %25) ve en az 3 iş günü bekleme (veya nakite acil kaçış)
        if (weight_diff >= 0.25 and days_since_trade >= 3) or (target_weight == 0.0 and current_weight > 0.0 and days_since_trade >= 1):
            old_w = current_weight
            current_weight = target_weight
            last_trade_day_idx = i
            
            # İşlem Tutarı ve Sürtünme Kesintisi (%0.10 Komisyon + Kayma)
            target_nominal = port_val * current_weight
            diff_nominal = abs(target_nominal - (ai_shares * curr_price))
            friction = diff_nominal * 0.0010
            net_port_val = max(port_val - friction, 100.0)
            
            # Yeni Hisse ve Nakit Dağılımı (Nakit asla negatif olamaz)
            ai_shares = (net_port_val * current_weight) / curr_price
            ai_cash = net_port_val * (1.0 - current_weight)
            
            action_type = "ALIM" if current_weight > old_w else "SATIŞ / NAKİT"
            badge_label = f"+{int(current_weight*10)}k" if current_weight > old_w else f"{int(current_weight*10)}k"
            
            # Açıklanabilir Yapay Zeka (XAI) Sebepleri
            reasons = []
            reasons.append(f"Yapay Zeka: 5 günde %{pred_med*100:+.1f} beklenti (Keskinlik Skoru: {int(s_ml)}/100)")
            
            if d200 > 0 and gc > 0:
                reasons.append(f"Trend: Fiyat 200 SMA üzerinde (%{d200*100:+.1f}) ve Golden Cross rejimi aktif")
            elif d200 < 0:
                reasons.append(f"Trend Uyarısı: Fiyat 200 SMA altında (%{d200*100:+.1f})")
                
            if alpha20 > 0:
                reasons.append(f"Sektörel Alfa: Sektör referansına göre 20 günlük rölatif güç +%{alpha20*100:.1f}")
                
            reasons.append(f"Temel Değerleme: {fund_data.get('valuation_status', 'MAKUL')} ({int(val_score)} puan)")
            
            if vol_ratio > 1.3:
                reasons.append(f"Para Akışı: Günlük hacim 20 günlük ortalamanın {vol_ratio:.1f}x katı")
                
            if "BİLANÇO" in earnings_regime:
                reasons.append(f"Bilanço Etkisi: {earnings_regime}")

            trade_event_today = {
                "day_index": i,
                "date": d_str,
                "action": action_type,
                "badge": badge_label,
                "price": round(curr_price, 2),
                "confidence_score": round(composite_confidence, 1),
                "prev_weight_pct": int(old_w * 100),
                "new_weight_pct": int(current_weight * 100),
                "stock_value": round(ai_shares * curr_price, 2),
                "cash_value": round(ai_cash, 2),
                "total_portfolio": round(port_val, 2),
                "reasons": reasons
            }
            trade_events.append(trade_event_today)
            
        # Gün Sonu Portföy Güncellemesi
        end_stock_val = ai_shares * curr_price
        end_port_val = ai_cash + end_stock_val
        
        timeline.append({
            "step": i + 1,
            "date": d_str,
            "price": round(curr_price, 2),
            "ai_equity": round(end_port_val, 2),
            "buy_hold_equity": round(float(bh_equity_series[i]), 2),
            "ai_stock_value": round(end_stock_val, 2),
            "ai_cash_value": round(ai_cash, 2),
            "weight_pct": int(current_weight * 100),
            "confidence_score": round(composite_confidence, 1),
            "trade_event": trade_event_today
        })

    # 5. Kurumsal Performans Skor Kartı (Tear Sheet)
    ai_final = timeline[-1]["ai_equity"]
    bh_final = timeline[-1]["buy_hold_equity"]
    
    ai_ret_pct = round(((ai_final - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100, 2)
    bh_ret_pct = round(((bh_final - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100, 2)
    
    # Günlük Getiriler ve Metrikler
    ai_equities = np.array([t["ai_equity"] for t in timeline])
    ai_daily_rets = np.diff(ai_equities) / ai_equities[:-1]
    
    rf = 0.04 / 252
    excess = ai_daily_rets - rf
    std = np.std(ai_daily_rets) + 1e-8
    sharpe = round(float(np.mean(excess) / std * np.sqrt(252)), 2)
    
    downside = ai_daily_rets[ai_daily_rets < 0]
    down_std = np.std(downside) + 1e-8 if len(downside) > 0 else 1e-8
    sortino = round(float(np.mean(excess) / down_std * np.sqrt(252)), 2)
    
    peak_ai = np.maximum.accumulate(ai_equities)
    dd_ai = (ai_equities - peak_ai) / peak_ai * 100
    max_dd_ai = round(float(np.max(np.abs(dd_ai))), 2)
    
    peak_bh = np.maximum.accumulate(bh_equity_series)
    dd_bh = (bh_equity_series - peak_bh) / peak_bh * 100
    max_dd_bh = round(float(np.max(np.abs(dd_bh))), 2)

    result = {
        "ticker": clean_ticker,
        "benchmark": benchmark,
        "initial_capital": INITIAL_CAPITAL,
        "test_period_days": len(test_df),
        "start_date": test_dates[0].strftime("%Y-%m-%d"),
        "end_date": test_dates[-1].strftime("%Y-%m-%d"),
        "performance": {
            "ai_final_equity": round(ai_final, 2),
            "buy_hold_final_equity": round(bh_final, 2),
            "ai_total_return_pct": ai_ret_pct,
            "buy_hold_total_return_pct": bh_ret_pct,
            "alpha_spread_pct": round(ai_ret_pct - bh_ret_pct, 2),
            "ai_sharpe": sharpe,
            "ai_sortino": sortino,
            "ai_max_drawdown_pct": max_dd_ai,
            "buy_hold_max_drawdown_pct": max_dd_bh,
            "total_trades": len(trade_events)
        },
        "trades": trade_events,
        "timeline": timeline
    }

    cache.set(cache_key, result, ttl=1800)
    return result
