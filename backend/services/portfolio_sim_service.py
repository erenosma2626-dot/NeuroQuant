"""
NeuroQuant 3.0: 1-Year Multi-Asset Dynamic Portfolio Simulation Engine
100k USD Capital, Up to 6 Tickers + Fixed Cash Option,
Fundamental & Technical Valuation Anchors (Cheap/Expensive Zones),
Dynamic Cash Accumulation & Opportunity Capital Deployment.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

from backend.config import settings
from backend.services.cache_service import cache
from backend.services.fundamental_service import fetch_fundamentals
from neuro_modules.quant_benchmark import (
    fetch_asset_and_benchmark,
    compute_quant_features
)
from backend.services.model_service import get_cluster_model

SIM_DAYS_1Y = 252  # 1 Yıl = Yaklaşık 252 işlem günü

def run_portfolio_simulation(
    basket: List[Dict[str, Any]],
    cash_weight_pct: float = 20.0,
    max_cash_pct: float = 80.0,
    initial_capital: float = 100000.0,
    rebalance_interval_days: int = 7
) -> Dict[str, Any]:
    """
    1 Yıllık (252 işlem günü) çoklu hisse sepeti portföy simülasyonunu çalıştırır.
    
    basket: [
        {"ticker": "NVDA", "weight_pct": 30.0},
        {"ticker": "AAPL", "weight_pct": 25.0},
        {"ticker": "MSFT", "weight_pct": 25.0},
    ]
    cash_weight_pct: Başlangıç nakit ağırlığı (%20)
    max_cash_pct: Modelin nakitte tutabileceği maksimum oran (%80)
    initial_capital: Başlangıç sermayesi (100.000$)
    """
    # 1. Girdi Doğrulama ve Normalizasyon
    if not basket or len(basket) == 0:
        raise ValueError("Sepette en az 1 hisse bulunmalıdır.")
    if len(basket) > 6:
        raise ValueError("Sepette en fazla 6 hisse bulunabilir.")

    # Ağırlıkların toplamını kontrol et ve normalize et
    stock_weights = {b["ticker"].strip().upper(): max(0.0, float(b.get("weight_pct", 0.0))) for b in basket}
    total_requested = sum(stock_weights.values()) + max(0.0, cash_weight_pct)
    if total_requested <= 0:
        equal_w = 80.0 / len(stock_weights)
        stock_weights = {t: equal_w for t in stock_weights}
        cash_weight_pct = 20.0
        total_requested = 100.0

    # %100'e normalize et
    norm_factor = 100.0 / total_requested
    norm_stock_weights = {t: w * norm_factor for t, w in stock_weights.items()}
    norm_cash_weight = max(0.0, cash_weight_pct * norm_factor)
    tickers = list(norm_stock_weights.keys())

    # Önbellek anahtarı
    cache_key_parts = [f"{t}_{norm_stock_weights[t]:.1f}" for t in sorted(tickers)]
    cache_key = f"portfolio_sim_1y_{'_'.join(cache_key_parts)}_c{norm_cash_weight:.1f}_m{max_cash_pct:.1f}_{int(initial_capital)}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    # 2. Tüm Hisselerin 5 Yıllık Verisini Çek ve Özellik Matrislerini Hesapla
    asset_data: Dict[str, Dict[str, Any]] = {}
    
    for ticker in tickers:
        benchmark = settings.get_benchmark(ticker)
        df_raw = fetch_asset_and_benchmark(ticker, benchmark, period="5y")
        df_feat = compute_quant_features(df_raw)
        df_feat['vol_sma20'] = df_feat['Volume'].rolling(20).mean()
        
        # Temel Değerleme Bilgisi
        fund_data = fetch_fundamentals(ticker)
        
        asset_data[ticker] = {
            "df_feat": df_feat,
            "fund": fund_data,
            "benchmark": benchmark
        }

    # 3. Ortak 252 İşlem Gününü Hizala
    common_dates_index = None
    for ticker in tickers:
        dates_t = asset_data[ticker]["df_feat"].index
        if common_dates_index is None:
            common_dates_index = dates_t
        else:
            common_dates_index = common_dates_index.intersection(dates_t)

    if len(common_dates_index) < SIM_DAYS_1Y:
        test_dates = common_dates_index[-len(common_dates_index):]
    else:
        test_dates = common_dates_index[-SIM_DAYS_1Y:]

    n_days = len(test_dates)
    test_dates_strs = [d.strftime("%Y-%m-%d") for d in test_dates]

    # 4. Her Hisse İçin Modeli Test Öncesi Eğit ve Tahminleri Al (Zero Leakage)
    for ticker in tickers:
        df_all = asset_data[ticker]["df_feat"]
        first_test_date = test_dates[0]
        train_mask = df_all.index < first_test_date
        train_df = df_all.loc[train_mask]
        test_df = df_all.loc[test_dates].copy()
        
        model = get_cluster_model(ticker)
        preds_dict = model.predict_cone(test_df)
        
        # Orta-Uzun Vade Değerleme Çıpaları (Anchor Values)
        # Başlangıç fiyatı ve tarihsel 200 SMA / Donchian seviyeleri
        c_0 = float(test_df['Close'].iloc[0])
        sma200_0 = float(test_df['sma_200'].iloc[0]) if ('sma_200' in test_df and not np.isnan(test_df['sma_200'].iloc[0])) else c_0
        val_score = float(asset_data[ticker]["fund"].get("valuation_score", 50.0))
        
        # Ucuz Bölge: Fiyatın SMA200'ün %5 altı veya tarihsel desteğe yakın olduğu seviye
        cheap_price = round(min(sma200_0 * 0.95, c_0 * 0.92), 2)
        # Pahalı Bölge: Fiyatın SMA200'ün %25 üzeri veya aşırı primlendiği seviye
        expensive_price = round(max(sma200_0 * 1.25, c_0 * 1.20), 2)

        asset_data[ticker]["test_df"] = test_df
        asset_data[ticker]["preds"] = preds_dict
        asset_data[ticker]["cheap_price"] = cheap_price
        asset_data[ticker]["expensive_price"] = expensive_price
        asset_data[ticker]["val_score"] = val_score

    # 5. Portföy Başlangıç Tahsisi ($100k Bütçe)
    initial_cash = initial_capital * (norm_cash_weight / 100.0)
    current_cash = initial_cash
    
    holdings: Dict[str, Dict[str, float]] = {} # ticker -> {shares, price, value, weight_pct}
    bh_initial_shares: Dict[str, float] = {}

    for ticker in tickers:
        init_alloc_usd = initial_capital * (norm_stock_weights[ticker] / 100.0)
        p0 = float(asset_data[ticker]["test_df"]['Close'].iloc[0])
        # İlk gün alım sürtünmesi (%0.10)
        friction0 = init_alloc_usd * 0.0010
        net_alloc0 = max(init_alloc_usd - friction0, 0.0)
        sh0 = net_alloc0 / p0 if p0 > 0 else 0.0
        
        holdings[ticker] = {
            "shares": sh0,
            "price": p0,
            "value": sh0 * p0,
            "weight_pct": norm_stock_weights[ticker]
        }
        # Karşılaştırma için Al-Tut başlangıç lotları
        bh_initial_shares[ticker] = (init_alloc_usd / p0) if p0 > 0 else 0.0

    timeline: List[Dict[str, Any]] = []
    trade_events: List[Dict[str, Any]] = []
    
    last_rebalance_idx = 0

    # 6. Günlük 1 Yıllık Dinamik Portföy Simülasyonu Döngüsü
    for i in range(n_days):
        date_str = test_dates_strs[i]
        
        # Güncel fiyatlar ve hisse değerleri
        total_stock_value = 0.0
        asset_daily_info: Dict[str, Any] = {}
        opportunity_scores: Dict[str, float] = {}
        
        for ticker in tickers:
            row_t = asset_data[ticker]["test_df"].iloc[i]
            p_t = float(row_t['Close'])
            sh_t = holdings[ticker]["shares"]
            val_t = sh_t * p_t
            total_stock_value += val_t
            
            holdings[ticker]["price"] = p_t
            holdings[ticker]["value"] = val_t
            
            # Fırsat Skoru Hesaplama (Orta-Uzun Vadeli Model)
            pred_med = float(asset_data[ticker]["preds"]["median"][i])
            iqr = float(abs(asset_data[ticker]["preds"]["upper_80"][i] - asset_data[ticker]["preds"]["lower_80"][i])) + 1e-6
            s_ml = 50.0 + float(np.tanh(pred_med / iqr * 2.5)) * 50.0
            
            d200 = float(row_t['dist_sma200']) if not np.isnan(row_t['dist_sma200']) else 0.0
            gc = float(row_t['sma50_200_ratio']) if not np.isnan(row_t['sma50_200_ratio']) else 0.0
            s_trend = float(np.clip(50.0 + (d200 * 150.0) + (gc * 100.0), 0.0, 100.0))
            
            s_val = asset_data[ticker]["val_score"]
            
            # Değerleme / Çıpa Bölgesi Etkisi
            ch_p = asset_data[ticker]["cheap_price"]
            ex_p = asset_data[ticker]["expensive_price"]
            
            zone_bonus = 0.0
            if p_t <= ch_p:
                zone_bonus = +20.0  # Ucuz bölge: Alım fırsatı bonusu
            elif p_t >= ex_p:
                zone_bonus = -25.0  # Pahalı bölge: Kâr alma cezası
            
            opp_score = float(np.clip(
                0.35 * s_ml + 0.30 * s_trend + 0.20 * s_val + 0.15 * 50.0 + zone_bonus,
                5.0, 95.0
            ))
            opportunity_scores[ticker] = opp_score
            
            # Başlangıçtan bugüne getiri %
            p0 = float(asset_data[ticker]["test_df"]['Close'].iloc[0])
            cum_ret = ((p_t - p0) / p0) * 100.0 if p0 > 0 else 0.0
            
            asset_daily_info[ticker] = {
                "price": round(p_t, 2),
                "shares": round(sh_t, 4),
                "value": round(val_t, 2),
                "return_pct": round(cum_ret, 2),
                "opportunity_score": round(opp_score, 1),
                "is_cheap": p_t <= ch_p,
                "is_expensive": p_t >= ex_p,
                "cheap_price": ch_p,
                "expensive_price": ex_p
            }

        # Güncel Portföy Değeri
        portfolio_equity = current_cash + total_stock_value
        
        # Ağırlıkları güncelle
        for ticker in tickers:
            holdings[ticker]["weight_pct"] = (holdings[ticker]["value"] / portfolio_equity * 100.0) if portfolio_equity > 0 else 0.0
            asset_daily_info[ticker]["weight_pct"] = round(holdings[ticker]["weight_pct"], 1)

        # Karşılaştırma: Eşit Ağırlıklı Sabit Al-Tut Sepeti + Sabit Nakit
        bh_stock_sum = sum(bh_initial_shares[t] * holdings[t]["price"] for t in tickers)
        benchmark_equity = initial_cash + bh_stock_sum

        trades_today: List[Dict[str, Any]] = []

        # --- ORTA-UZUN VADELİ REBALANS & NAKİT YÖNETİMİ ---
        # Belirli periyotlarda veya hisse aşırı pahalı/ucuz bölgeye girdiğinde karar verilir
        days_since_reb = i - last_rebalance_idx
        should_rebalance = (days_since_reb >= rebalance_interval_days) or any(
            asset_daily_info[t]["is_expensive"] or asset_daily_info[t]["is_cheap"] for t in tickers
        )

        if should_rebalance and i > 0 and i < n_days - 1:
            # 1. Aşama: Pahalı veya Bozulmuş Hisselerden Kâr Realizasyonu (Nakite Kaçış)
            for ticker in tickers:
                info = asset_daily_info[ticker]
                curr_w = holdings[ticker]["weight_pct"]
                
                # Eğer hisse pahalı bölgedeyse veya fırsat skoru < 42 ise ağırlık azalt / sat
                if (info["is_expensive"] or info["opportunity_score"] < 42.0) and curr_w > 5.0:
                    target_w = 5.0 if info["opportunity_score"] >= 35.0 else 0.0
                    w_cut = curr_w - target_w
                    
                    if w_cut >= 8.0:  # Anlamlı kâr satışı
                        sell_nominal = (w_cut / 100.0) * portfolio_equity
                        friction = sell_nominal * 0.0010
                        shares_to_sell = sell_nominal / info["price"]
                        
                        holdings[ticker]["shares"] = max(0.0, holdings[ticker]["shares"] - shares_to_sell)
                        current_cash += (sell_nominal - friction)
                        
                        trade_ev = {
                            "day_index": i,
                            "date": date_str,
                            "ticker": ticker,
                            "action": "KÂR AL / NAKİT",
                            "badge": f"-{int(w_cut)}%",
                            "price": info["price"],
                            "shares": round(shares_to_sell, 2),
                            "delta_notional": round(sell_nominal, 2),
                            "friction_cost": round(friction, 2),
                            "prev_weight_pct": round(curr_w, 1),
                            "new_weight_pct": round(target_w, 1),
                            "reason": f"Fiyat (${info['price']}) hedef değerin üzerine çıktı ve pahalı bölgeye girdi. Kâr realize edilerek nakite geçildi."
                        }
                        trades_today.append(trade_ev)
                        trade_events.append(trade_ev)

            # 2. Aşama: Birikmiş Nakit ile Ucuzluk & Yüksek Fırsat Hisselerine Sermaye Dağıtımı
            # Mevcut nakit oranını kontrol et
            current_cash_pct = (current_cash / portfolio_equity * 100.0) if portfolio_equity > 0 else 0.0
            
            # En cazip fırsatları sırala
            attractive_tickers = sorted(
                [t for t in tickers if opportunity_scores[t] >= 60.0 or asset_daily_info[t]["is_cheap"]],
                key=lambda t: opportunity_scores[t],
                reverse=True
            )

            # Eğer nakit varsa ve çok cazip bir hisse varsa nakitten alım yap
            if current_cash_pct > 10.0 and len(attractive_tickers) > 0:
                top_pick = attractive_tickers[0]
                top_score = opportunity_scores[top_pick]
                top_info = asset_daily_info[top_pick]
                
                # Boşta olan nakdin bir kısmını (örneğin %40'ını) fırsata tahsis et
                invest_budget = current_cash * 0.45
                if invest_budget >= (0.04 * portfolio_equity):  # En az portföyün %4'ü
                    friction = invest_budget * 0.0010
                    net_invest = invest_budget - friction
                    shares_bought = net_invest / top_info["price"]
                    
                    old_w = holdings[top_pick]["weight_pct"]
                    holdings[top_pick]["shares"] += shares_bought
                    current_cash -= invest_budget
                    
                    new_w = old_w + (invest_budget / portfolio_equity * 100.0)
                    
                    trade_ev = {
                        "day_index": i,
                        "date": date_str,
                        "ticker": top_pick,
                        "action": "ALIM / FIRSAT",
                        "badge": f"+{int(invest_budget / portfolio_equity * 100)}%",
                        "price": top_info["price"],
                        "shares": round(shares_bought, 2),
                        "delta_notional": round(invest_budget, 2),
                        "friction_cost": round(friction, 2),
                        "prev_weight_pct": round(old_w, 1),
                        "new_weight_pct": round(new_w, 1),
                        "reason": f"Fiyat (${top_info['price']}) cazip destek/ucuzluk seviyesine geldi (Skor: {top_score:.1f}). Biriken nakit değerlendirildi."
                    }
                    trades_today.append(trade_ev)
                    trade_events.append(trade_ev)

            # Maksimum Nakit Sınırı Koruması: Nakit max_cash_pct'yi geçerse en dengeli hisselere eşit dağıt
            if (current_cash / portfolio_equity * 100.0) > max_cash_pct:
                excess_cash = current_cash - (portfolio_equity * (max_cash_pct / 100.0))
                if excess_cash > 2000 and len(attractive_tickers) > 0:
                    per_asset_cash = excess_cash / len(attractive_tickers[:2])
                    for t_pick in attractive_tickers[:2]:
                        p_pick = asset_daily_info[t_pick]["price"]
                        f_pick = per_asset_cash * 0.0010
                        sh_pick = (per_asset_cash - f_pick) / p_pick
                        holdings[t_pick]["shares"] += sh_pick
                        current_cash -= per_asset_cash

            if len(trades_today) > 0:
                last_rebalance_idx = i

        # Gün sonu değerleme
        total_stock_value = sum(holdings[t]["shares"] * holdings[t]["price"] for t in tickers)
        portfolio_equity = current_cash + total_stock_value
        cash_pct = (current_cash / portfolio_equity * 100.0) if portfolio_equity > 0 else 0.0

        timeline.append({
            "step": i + 1,
            "date": date_str,
            "portfolio_equity": round(portfolio_equity, 2),
            "benchmark_equity": round(benchmark_equity, 2),
            "cash_value": round(current_cash, 2),
            "cash_pct": round(cash_pct, 1),
            "assets": asset_daily_info,
            "trades_today": trades_today
        })

    # 7. Nihai Performans Metrikleri (1 Yıllık Tear Sheet)
    final_eq = timeline[-1]["portfolio_equity"]
    final_bh = timeline[-1]["benchmark_equity"]
    
    port_ret_pct = round(((final_eq - initial_capital) / initial_capital) * 100.0, 2)
    bh_ret_pct = round(((final_bh - initial_capital) / initial_capital) * 100.0, 2)
    alpha_spread = round(port_ret_pct - bh_ret_pct, 2)

    # Günlük Getiriler ve Risk İstatistikleri
    equities_arr = np.array([t["portfolio_equity"] for t in timeline])
    daily_rets = np.diff(equities_arr) / equities_arr[:-1]
    
    rf_daily = 0.045 / 252
    excess_rets = daily_rets - rf_daily
    std_ret = np.std(daily_rets) + 1e-8
    sharpe = round(float(np.mean(excess_rets) / std_ret * np.sqrt(252)), 2)
    
    downside_rets = daily_rets[daily_rets < 0]
    down_std = np.std(downside_rets) + 1e-8 if len(downside_rets) > 0 else 1e-8
    sortino = round(float(np.mean(excess_rets) / down_std * np.sqrt(252)), 2)
    
    peak_eq = np.maximum.accumulate(equities_arr)
    drawdowns = (equities_arr - peak_eq) / peak_eq * 100.0
    max_dd = round(float(np.max(np.abs(drawdowns))), 2)

    # Bireysel Hisse 1 Yıllık Al-Tut Getirileri
    asset_summaries = []
    for ticker in tickers:
        p0 = float(asset_data[ticker]["test_df"]['Close'].iloc[0])
        p_last = float(asset_data[ticker]["test_df"]['Close'].iloc[-1])
        ret_1y = round(((p_last - p0) / p0) * 100.0, 2)
        asset_summaries.append({
            "ticker": ticker,
            "initial_weight_pct": round(norm_stock_weights[ticker], 1),
            "final_weight_pct": round(timeline[-1]["assets"][ticker]["weight_pct"], 1),
            "p0": round(p0, 2),
            "p_final": round(p_last, 2),
            "buy_hold_return_pct": ret_1y,
            "cheap_price": asset_data[ticker]["cheap_price"],
            "expensive_price": asset_data[ticker]["expensive_price"],
            "valuation_status": asset_data[ticker]["fund"].get("valuation_status", "MAKUL")
        })

    result = {
        "initial_capital": initial_capital,
        "test_period_days": n_days,
        "start_date": test_dates_strs[0],
        "end_date": test_dates_strs[-1],
        "cash_initial_pct": round(norm_cash_weight, 1),
        "max_cash_pct": round(max_cash_pct, 1),
        "assets_summary": asset_summaries,
        "performance": {
            "ai_final_equity": round(final_eq, 2),
            "buy_hold_final_equity": round(final_bh, 2),
            "ai_total_return_pct": port_ret_pct,
            "buy_hold_total_return_pct": bh_ret_pct,
            "alpha_spread_pct": alpha_spread,
            "ai_sharpe": sharpe,
            "ai_sortino": sortino,
            "ai_max_drawdown_pct": max_dd,
            "total_trades": len(trade_events)
        },
        "trades": trade_events,
        "timeline": timeline
    }

    cache.set(cache_key, result, ttl=1800)
    return result
