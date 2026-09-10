import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from neuro_modules.quant_benchmark import (
    fetch_asset_and_benchmark,
    compute_quant_features,
    QuantileLightGBMCluster
)

def test_gold_daily_rolling_5d(ticker="GC=F", test_days=252):
    """
    Test 1: Günlük veride 1 haftalık (5 iş günü) ileri pencereye baka baka gitme
    """
    print(f"\n=======================================================")
    print(f"🥇 TEST 1: {ticker} GÜNLÜK GRAFİKTE 1 HAFTALIK (5 İŞ GÜNÜ) İLERİ TAHMİN")
    print(f"=======================================================")
    
    # 5 Yıllık veri çek (Referans olarak DXY veya SPY)
    df_raw = fetch_asset_and_benchmark(ticker, "SPY", period="5y")
    df_feat = compute_quant_features(df_raw)
    
    valid_mask = ~df_feat['target_5d'].isna()
    df_valid = df_feat.loc[valid_mask].copy()
    
    total_len = len(df_valid)
    train_len = total_len - test_days
    
    train_df = df_valid.iloc[:train_len].copy()
    test_df = df_valid.iloc[train_len:].copy()
    
    train_start = train_df.index[0].strftime('%Y-%m-%d')
    train_end = train_df.index[-1].strftime('%Y-%m-%d')
    test_start = test_df.index[0].strftime('%Y-%m-%d')
    test_end = test_df.index[-1].strftime('%Y-%m-%d')
    
    print(f"📅 Eğitim: {train_start} -> {train_end} ({len(train_df)} gün)")
    print(f"🎯 Test (Kör Veri): {test_start} -> {test_end} ({len(test_df)} gün - ~{test_days//5} hafta)")
    
    # Sıfır sızıntı ile eğit
    model = QuantileLightGBMCluster(ticker)
    model.fit(train_df, train_df['target_5d'])
    
    preds = model.predict_cone(test_df)
    actual_5d = test_df['target_5d'].values
    pred_5d = preds['median']
    lower_80 = preds['lower_80']
    upper_80 = preds['upper_80']
    
    # Yön Doğruluğu
    correct_dir = np.sign(pred_5d) == np.sign(actual_5d)
    dir_acc = np.mean(correct_dir) * 100.0
    
    long_mask = pred_5d > 0
    long_prec = np.mean(actual_5d[long_mask] > 0) * 100.0 if np.sum(long_mask) > 0 else 0.0
    short_mask = pred_5d < 0
    short_prec = np.mean(actual_5d[short_mask] < 0) * 100.0 if np.sum(short_mask) > 0 else 0.0
    
    # %80 Kapsama
    coverage = np.mean((actual_5d >= lower_80) & (actual_5d <= upper_80)) * 100.0
    mae = np.mean(np.abs(pred_5d - actual_5d)) * 100.0
    
    # Basit Strateji Simülasyonu (Model Yükseliş diyorsa Long, değilse Nakit)
    # Day t-1 tahmini ile Day t pozisyonu (Sıfır lookahead bias)
    positions = pd.Series(np.where(pred_5d > 0, 1.0, 0.0), index=test_df.index).shift(1).fillna(0).values
    daily_rets = test_df['ret_1d'].values
    strat_rets = positions * daily_rets
    
    cum_strat = (np.cumprod(1 + strat_rets)[-1] - 1) * 100.0
    cum_bh = (np.cumprod(1 + daily_rets)[-1] - 1) * 100.0
    
    # Sharpe & DD
    rf = 0.04 / 252
    excess = strat_rets - rf
    sharpe = float(np.mean(excess) / (np.std(strat_rets) + 1e-8) * np.sqrt(252))
    peak = np.maximum.accumulate(np.cumprod(1 + strat_rets))
    dd = (np.cumprod(1 + strat_rets) - peak) / peak * 100.0
    max_dd = float(np.max(np.abs(dd)))
    
    print(f"📊 SONUÇLAR (Günlükte 1 Haftalık İleri Ufuk):")
    print(f"   • Yön Doğruluğu (Win Rate): %{dir_acc:.2f}")
    print(f"   • Yükseliş İsabeti (Long Precision): %{long_prec:.2f} ({int(np.sum(long_mask))} sinyal)")
    print(f"   • Düşüş İsabeti (Short Precision): %{short_prec:.2f} ({int(np.sum(short_mask))} sinyal)")
    print(f"   • %80 Güven Konisi Kapsama: %{coverage:.2f}")
    print(f"   • Ortalama Hata Payı (MAE): %{mae:.2f}")
    print(f"   • Model Strateji Net Getiri: %{cum_strat:.2f}")
    print(f"   • Altın Al-Tut Getirisi:    %{cum_bh:.2f}")
    print(f"   • Sharpe Oranı: {sharpe:.2f} | Maks. Drawdown: -%{max_dd:.2f}")

    return {
        "dir_acc": dir_acc,
        "long_prec": long_prec,
        "short_prec": short_prec,
        "coverage": coverage,
        "mae": mae,
        "cum_strat": cum_strat,
        "cum_bh": cum_bh,
        "sharpe": sharpe,
        "max_dd": max_dd
    }

def test_gold_pure_weekly(ticker="GC=F", test_weeks=52):
    """
    Test 2: Saf Haftalık Barlar (W-FRI) Üzerinde Model Eğitip Gelecek Haftanın Yönünü Tahmin Etme
    """
    print(f"\n=======================================================")
    print(f"🕯️ TEST 2: {ticker} SAF HAFTALIK GRAFİK (1W MUMLARLA HAFTA HAFTA YÖN TAHMİNİ)")
    print(f"=======================================================")
    
    # 10 Yıllık Günlük Veri İndirip Haftalığa Resample Et
    df_raw = yf.download(ticker, period="10y", interval="1d", progress=False, threads=False)
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if isinstance(df_raw.columns, pd.MultiIndex):
        try: df_raw = df_raw.xs(ticker, axis=1, level=1)
        except: pass
    df_raw = df_raw[[c for c in cols if c in df_raw.columns]].dropna()
    
    # Haftalık barlar (W-FRI: Cuma kapanışları)
    df_w = pd.DataFrame()
    df_w['Open'] = df_raw['Open'].resample('W-FRI').first()
    df_w['High'] = df_raw['High'].resample('W-FRI').max()
    df_w['Low'] = df_raw['Low'].resample('W-FRI').min()
    df_w['Close'] = df_raw['Close'].resample('W-FRI').last()
    df_w['Volume'] = df_raw['Volume'].resample('W-FRI').sum()
    df_w = df_w.dropna()
    
    # Haftalık İndikatörler ve Özellikler
    df_w['ret_1w'] = df_w['Close'].pct_change()
    df_w['ret_4w_cum'] = df_w['Close'].pct_change(4)
    df_w['ret_12w_cum'] = df_w['Close'].pct_change(12)
    
    df_w['sma_10'] = df_w['Close'].rolling(10).mean()
    df_w['sma_40'] = df_w['Close'].rolling(40).mean() # ~200 günlük eşdeğeri
    df_w['dist_sma10'] = (df_w['Close'] - df_w['sma_10']) / df_w['sma_10']
    df_w['dist_sma40'] = (df_w['Close'] - df_w['sma_40']) / df_w['sma_40']
    df_w['sma_ratio'] = (df_w['sma_10'] - df_w['sma_40']) / df_w['sma_40']
    
    # Volatilite
    df_w['vol_8w'] = df_w['ret_1w'].rolling(8).std()
    
    # HEDEF DEĞİŞKEN: Gelecek 1 Haftanın Getirisi (target_next_week)
    df_w['target_next_week'] = df_w['Close'].shift(-1) / df_w['Close'] - 1.0
    
    feature_cols = ['ret_1w', 'ret_4w_cum', 'ret_12w_cum', 'dist_sma10', 'dist_sma40', 'sma_ratio', 'vol_8w']
    df_clean = df_w.dropna().copy()
    
    total_w = len(df_clean)
    train_w = total_w - test_weeks
    
    train_data = df_clean.iloc[:train_w].copy()
    test_data = df_clean.iloc[train_w:].copy()
    
    print(f"📅 Eğitim: {train_data.index[0].strftime('%Y-%m-%d')} -> {train_data.index[-1].strftime('%Y-%m-%d')} ({len(train_data)} hafta)")
    print(f"🎯 Test (Kör Veri): {test_data.index[0].strftime('%Y-%m-%d')} -> {test_data.index[-1].strftime('%Y-%m-%d')} ({len(test_data)} hafta = ~{len(test_data)/52:.1f} yıl)")
    
    # LightGBM Quantile Modelleri (q10, q50, q90)
    m10 = lgb.LGBMRegressor(objective='quantile', alpha=0.10, n_estimators=80, learning_rate=0.03, random_state=42, verbose=-1)
    m50 = lgb.LGBMRegressor(objective='quantile', alpha=0.50, n_estimators=80, learning_rate=0.03, random_state=42, verbose=-1)
    m90 = lgb.LGBMRegressor(objective='quantile', alpha=0.90, n_estimators=80, learning_rate=0.03, random_state=42, verbose=-1)
    
    m10.fit(train_data[feature_cols], train_data['target_next_week'])
    m50.fit(train_data[feature_cols], train_data['target_next_week'])
    m90.fit(train_data[feature_cols], train_data['target_next_week'])
    
    pred_10 = m10.predict(test_data[feature_cols])
    pred_50 = m50.predict(test_data[feature_cols])
    pred_90 = m90.predict(test_data[feature_cols])
    
    actual_next = test_data['target_next_week'].values
    
    # 1. Yön Doğruluğu (Haftalık Win Rate)
    correct_dir = np.sign(pred_50) == np.sign(actual_next)
    weekly_accuracy = np.mean(correct_dir) * 100.0
    
    # 2. Long İsabeti (Model gelecek hafta artacak dediğinde ne oldu?)
    long_mask = pred_50 > 0
    long_count = int(np.sum(long_mask))
    long_prec = np.mean(actual_next[long_mask] > 0) * 100.0 if long_count > 0 else 0.0
    
    # 3. Short İsabeti
    short_mask = pred_50 < 0
    short_count = int(np.sum(short_mask))
    short_prec = np.mean(actual_next[short_mask] < 0) * 100.0 if short_count > 0 else 0.0
    
    # 4. Kapsama ve Hata
    coverage = np.mean((actual_next >= pred_10) & (actual_next <= pred_90)) * 100.0
    mae = np.mean(np.abs(pred_50 - actual_next)) * 100.0
    
    # 5. Haftalık Al-Sat Stratejisi
    # Gelecek haftanın sinyali bu haftanın Cuma kapanışında verilir
    # Pozisyon: Model > 0 ise 1 (Long), değilse 0 (Nakit)
    positions = np.where(pred_50 > 0, 1.0, 0.0)
    strat_rets = positions * actual_next
    
    cum_strat = (np.cumprod(1 + strat_rets)[-1] - 1) * 100.0
    cum_bh = (np.cumprod(1 + actual_next)[-1] - 1) * 100.0
    
    # Sharpe & Max Drawdown
    rf_w = 0.04 / 52
    excess = strat_rets - rf_w
    sharpe = float(np.mean(excess) / (np.std(strat_rets) + 1e-8) * np.sqrt(52))
    peak = np.maximum.accumulate(np.cumprod(1 + strat_rets))
    dd = (np.cumprod(1 + strat_rets) - peak) / peak * 100.0
    max_dd = float(np.max(np.abs(dd)))
    
    print(f"📊 SONUÇLAR (Saf Haftalık Barlarla Test):")
    print(f"   • Haftalık Yön Doğruluğu (Weekly Directional Accuracy): %{weekly_accuracy:.2f}")
    print(f"   • Yükseliş İsabeti (Long Precision): %{long_prec:.2f} ({long_count}/{len(test_data)} hafta)")
    print(f"   • Düşüş İsabeti (Short Precision):   %{short_prec:.2f} ({short_count}/{len(test_data)} hafta)")
    print(f"   • %80 Güven Konisi Kapsama Oranı:    %{coverage:.2f}")
    print(f"   • Haftalık Ortalama Hata (MAE):      %{mae:.2f}")
    print(f"   • Model Strateji Net Getirisi:       %{cum_strat:.2f}")
    print(f"   • Altın Al-Tut (Buy & Hold) Getirisi:%{cum_bh:.2f}")
    print(f"   • Yıllık Sharpe Oranı: {sharpe:.2f} | Maks. Drawdown: -%{max_dd:.2f}")
    
    # Son 10 Haftanın Canlı Detayı
    print(f"\n🔍 SON 10 HAFTANIN TAHMİN / GERÇEKLEŞME TABLOSU:")
    print(f"   {'Tarih':<12} | {'Tahmin Edilen Yön':<18} | {'Tahmin %':<10} | {'Gerçekleşen %':<14} | {'İsabet'}")
    print(f"   {'-'*65}")
    for i in range(max(0, len(test_data)-10), len(test_data)):
        d_str = test_data.index[i].strftime('%Y-%m-%d')
        p_val = pred_50[i] * 100.0
        a_val = actual_next[i] * 100.0
        p_dir = "YÜKSELİŞ 🟢" if p_val > 0 else "DÜŞÜŞ 🔴"
        is_hit = "✓ DOĞRU" if np.sign(p_val) == np.sign(a_val) else "✗ YANLIŞ"
        print(f"   {d_str:<12} | {p_dir:<18} | %{p_val:>+6.2f}    | %{a_val:>+6.2f}        | {is_hit}")

    return {
        "weekly_accuracy": weekly_accuracy,
        "long_prec": long_prec,
        "short_prec": short_prec,
        "coverage": coverage,
        "mae": mae,
        "cum_strat": cum_strat,
        "cum_bh": cum_bh,
        "sharpe": sharpe,
        "max_dd": max_dd
    }

if __name__ == "__main__":
    print("🌟 NEUROQUANT 3.0: ONS ALTIN (GC=F / GOLD) YÖN DOĞRULUĞU ANALİZİ")
    test_gold_daily_rolling_5d("GC=F", test_days=252)
    test_gold_pure_weekly("GC=F", test_weeks=52)
    test_gold_pure_weekly("GLD", test_weeks=104) # 2 Yıllık Test
