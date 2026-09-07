"""
NeuroQuant 3.0: Rigorous Historical Time-Wall Backtest (Production Grade)
========================================================================
Eğitim: 2021/2022 - Son 6 ay öncesi (~3.5 yıl)
Test: Son 6 ay (126 iş günü - Kesinlikle kör veri, sıfır sızıntı / zero leakage)
Pozisyonlama: Day t-1 tahmini ile Day t pozisyonu (Sıfır lookahead bias)
"""

import os
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from neuro_modules.quant_benchmark import (
    fetch_asset_and_benchmark,
    compute_quant_features,
    QuantileLightGBMCluster
)

# Dark Sovereign Theme
plt.style.use("dark_background")
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#1F242D"
plt.rcParams["axes.linewidth"] = 1.0

TEST_DAYS = 126  # Son 6 ay (yaklaşık 126 borsa işlem günü)

def run_asset_backtest(ticker="NVDA", benchmark="SMH", title_label="NVIDIA Corp. vs. SMH Semiconductor ETF"):
    print(f"\n=======================================================")
    print(f"🔬 {ticker} İÇİN 6 AYLIK ZAMAN DUVARI (TIME-WALL) TESTİ")
    print(f"=======================================================")
    
    # 1. 5 Yıllık Veri Çek
    df_raw = fetch_asset_and_benchmark(ticker, benchmark, period="5y")
    df_feat = compute_quant_features(df_raw)
    
    total_len = len(df_feat)
    train_len = total_len - TEST_DAYS
    
    train_df = df_feat.iloc[:train_len].copy()
    test_df = df_feat.iloc[train_len:].copy()
    
    train_start = train_df.index[0].strftime("%Y-%m-%d")
    train_end = train_df.index[-1].strftime("%Y-%m-%d")
    test_start = test_df.index[0].strftime("%Y-%m-%d")
    test_end = test_df.index[-1].strftime("%Y-%m-%d")
    
    print(f"📅 EĞİTİM DÖNEMİ: {train_start} -> {train_end} ({len(train_df)} gün)")
    print(f"🛑 ZAMAN DUVARI KESİMİ: {train_end}")
    print(f"🎯 KÖR TEST DÖNEMİ:  {test_start} -> {test_end} ({len(test_df)} gün)")
    
    # 2. Modeli SADECE Eğitim Dönemiyle Eğit (Kör veri sızıntısız)
    model = QuantileLightGBMCluster(ticker)
    model.fit(train_df, train_df['target_5d'])
    
    # 3. Kör Test Döneminde Simülasyon
    preds = model.predict_cone(test_df)
    
    actual_5d = test_df['target_5d'].values
    pred_5d = preds['median']
    lower_80 = preds['lower_80']
    upper_80 = preds['upper_80']
    
    # Gerçek 5 Günlük Yön Tutarlılığı (Win Rate)
    win_rate = np.mean(np.sign(pred_5d) == np.sign(actual_5d)) * 100
    
    # 4. Sızıntısız Pozisyonlama (Lookahead-Safe) & Kurumsal Trend Rejim Filtresi
    # Model RSI kullanmaz. Bunun yerine Fiyat > SMA 200 veya Golden Cross (SMA 50 > SMA 200) filtresi uygulanır.
    # Ayı rejimindeyken (Death Cross / SMA 200 altı) sermayeyi korumak için long sinyalleri filtrelenir.
    trend_regime = (test_df['dist_sma200'] > 0) | (test_df['sma50_200_ratio'] > 0)
    raw_signal = np.where((pred_5d > 0) & trend_regime, 1.0, 0.0)
    positions = pd.Series(raw_signal, index=test_df.index).shift(1).fillna(0).values
    
    daily_rets = test_df['ret_1d'].values
    strategy_rets = positions * daily_rets
    
    # Kümülatif Getiriler
    equity_strategy = np.cumprod(1 + strategy_rets)
    equity_buyhold = np.cumprod(1 + daily_rets)
    
    # Sharpe & Sortino (Yıllıklandırılmış)
    rf = 0.04 / 252
    excess_strat = strategy_rets - rf
    sharpe = float(np.mean(excess_strat) / (np.std(strategy_rets) + 1e-8) * np.sqrt(252))
    
    downside = strategy_rets[strategy_rets < 0]
    sortino = float(np.mean(excess_strat) / (np.std(downside) + 1e-8) * np.sqrt(252)) if len(downside) > 0 else 0.0
    
    # Drawdown Hesaplama
    peak_strat = np.maximum.accumulate(equity_strategy)
    dd_strat = (equity_strategy - peak_strat) / peak_strat * 100
    max_dd_strat = float(np.max(np.abs(dd_strat)))
    
    peak_bh = np.maximum.accumulate(equity_buyhold)
    dd_bh = (equity_buyhold - peak_bh) / peak_bh * 100
    max_dd_bh = float(np.max(np.abs(dd_bh)))
    
    # Profit Factor
    gains = strategy_rets[strategy_rets > 0].sum()
    losses = np.abs(strategy_rets[strategy_rets < 0].sum())
    profit_factor = float(gains / (losses + 1e-8)) if losses > 0 else float(gains)
    
    total_strat_ret = float((equity_strategy[-1] - 1) * 100)
    total_bh_ret = float((equity_buyhold[-1] - 1) * 100)
    
    print("\n📊 6 AYLIK KÖR TEST PERFORMANS RAPORU:")
    print(f"   • 5-Günlük Yön Doğruluğu:  %{win_rate:.2f}")
    print(f"   • Yıllık Sharpe Oranı:     {sharpe:.2f}")
    print(f"   • Sortino Oranı:            {sortino:.2f}")
    print(f"   • Strateji Max Drawdown:    %{max_dd_strat:.2f} (Piyasa Max DD: %{max_dd_bh:.2f})")
    print(f"   • Profit Factor:            {profit_factor:.2f}")
    print(f"   • AI Strateji Net Getiri:   %{total_strat_ret:.2f}")
    print(f"   • Al-Tut (Buy & Hold):      %{total_bh_ret:.2f}")
    
    # 5. YÜKSEK ÇÖZÜNÜRLÜKLÜ GRAFİK ÇİZİMİ (3 Panel - Sovereign Minimalist)
    test_dates = test_df.index
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True, 
                                        gridspec_kw={'height_ratios': [2.2, 2.2, 1.2]})
    
    # --- PANEL 1: Gerçek Fiyat Hareketi ve AI Pozisyon Rejimi ---
    prices = test_df['Close'].values
    ax1.plot(test_dates, prices, color="#F8FAFC", linewidth=2.0, label=f"{ticker} Gerçek Fiyat")
    
    # Pozisyonda olunan günleri yarı saydam zümrüt yeşili ile boya
    in_pos = False
    start_d = None
    for i in range(len(positions)):
        if positions[i] == 1.0 and not in_pos:
            in_pos = True
            start_d = test_dates[i]
        elif positions[i] == 0.0 and in_pos:
            in_pos = False
            ax1.axvspan(start_d, test_dates[i], color="#10B981", alpha=0.15)
    if in_pos:
        ax1.axvspan(start_d, test_dates[-1], color="#10B981", alpha=0.15, label="NeuroQuant Pozisyonda (Long)")
        
    ax1.set_title(f"{title_label} - 6 Aylık Zaman Duvarı Testi ({test_start} - {test_end})", 
                  fontsize=14, fontweight="bold", color="#F8FAFC", pad=12)
    ax1.set_ylabel("Fiyat ($)", color="#94A3B8", fontsize=11)
    ax1.legend(loc="upper left", framealpha=0.25, facecolor="#08090A", edgecolor="#1F242D")
    ax1.grid(True, linestyle="--", alpha=0.15, color="#334155")
    
    # --- PANEL 2: Kümülatif Getiri Eğrisi (AI Stratejisi vs Buy & Hold) ---
    ax2.plot(test_dates, equity_strategy, color="#10B981", linewidth=2.5, 
             label=f"NeuroQuant Stratejisi (%{total_strat_ret:+.1f} | Sharpe: {sharpe:.2f})")
    ax2.plot(test_dates, equity_buyhold, color="#64748B", linewidth=1.8, linestyle="--", 
             label=f"Al-Tut / Buy & Hold (%{total_bh_ret:+.1f} | Max DD: %{max_dd_bh:.1f})")
    ax2.axhline(1.0, color="#475569", linestyle=":", linewidth=1)
    ax2.set_ylabel("Sermaye Katsayısı (1.0 = Başlangıç)", color="#94A3B8", fontsize=11)
    ax2.legend(loc="upper left", framealpha=0.25, facecolor="#08090A", edgecolor="#1F242D")
    ax2.grid(True, linestyle="--", alpha=0.15, color="#334155")
    
    # --- PANEL 3: Sualtı / Drawdown Karşılaştırması (%) ---
    ax3.fill_between(test_dates, dd_strat, 0, color="#10B981", alpha=0.35, label=f"AI Çekilme (Max %{max_dd_strat:.1f})")
    ax3.plot(test_dates, dd_strat, color="#10B981", linewidth=1.2)
    ax3.plot(test_dates, dd_bh, color="#EF4444", linewidth=1.2, linestyle="--", label=f"Piyasa Çekilme (Max %{max_dd_bh:.1f})")
    ax3.set_ylabel("Drawdown (%)", color="#94A3B8", fontsize=11)
    ax3.set_xlabel("Tarih", color="#94A3B8", fontsize=11)
    ax3.legend(loc="lower left", framealpha=0.25, facecolor="#08090A", edgecolor="#1F242D")
    ax3.grid(True, linestyle="--", alpha=0.15, color="#334155")
    
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.tight_layout()
    
    os.makedirs("assets", exist_ok=True)
    safe_name = ticker.replace('.', '_').replace('-', '_').lower()
    out_path = f"assets/backtest_6mo_{safe_name}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    
    # Artifact kopyası (kullanıcı UI görüntülemesi için)
    artifact_dir = "/Users/erenosma/.gemini/antigravity-ide/brain/14f11852-cd5d-48ee-9099-8078f173345d"
    artifact_copy = os.path.join(artifact_dir, f"backtest_6mo_{safe_name}.png")
    shutil.copy2(out_path, artifact_copy)
    
    print(f"💾 Grafik Kaydedildi: {out_path}")
    print(f"🖼️ Artifact Kopyalandı: {artifact_copy}")
    
    return {
        "ticker": ticker,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_strat": max_dd_strat,
        "max_dd_bh": max_dd_bh,
        "profit_factor": profit_factor,
        "total_strat_ret": total_strat_ret,
        "total_bh_ret": total_bh_ret,
        "chart_path": out_path,
        "artifact_path": artifact_copy
    }

if __name__ == "__main__":
    r_nvda = run_asset_backtest("NVDA", "SMH", "NVIDIA Corp. vs. SMH Semiconductor ETF")
    r_btc = run_asset_backtest("BTC-USD", "BTC-USD", "Bitcoin 24/7 Market Simulation")
    r_thyao = run_asset_backtest("THYAO.IS", "XU100.IS", "THYAO vs. BIST 100 Endeksi")
