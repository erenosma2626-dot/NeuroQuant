"""
NeuroQuant 3.0: 10,000 Capital Dynamic Simulation Plotter
Generates 3-Panel Sovereign Visuals with Exact Buy/Sell Execution Badges & Allocation Breakdown
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.services.simulation_service import run_simulation
from backend.services.cache_service import cache

# Sovereign Quant Dark Palette
plt.style.use("dark_background")
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#1F242D"
plt.rcParams["axes.linewidth"] = 1.0

ARTIFACT_DIR = "/Users/erenosma/.gemini/antigravity-ide/brain/14f11852-cd5d-48ee-9099-8078f173345d"

def plot_single_simulation(ticker: str, title: str):
    print(f"\n🎨 Çiziliyor: {ticker}...")
    sim_data = run_simulation(ticker)
    
    timeline = sim_data["timeline"]
    trades = sim_data["trades"]
    perf = sim_data["performance"]
    
    dates = [datetime.strptime(t["date"], "%Y-%m-%d") for t in timeline]
    prices = [t["price"] for t in timeline]
    ai_equity = [t["ai_equity"] for t in timeline]
    bh_equity = [t["buy_hold_equity"] for t in timeline]
    weights = [t["weight_pct"] for t in timeline]
    confidences = [t["confidence_score"] for t in timeline]
    
    # 3-Panel Sovereign Layout
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True,
                                        gridspec_kw={'height_ratios': [2.6, 2.2, 1.4]})
    
    # --- PANEL 1: Fiyat Hareketi ve Kesin AL/SAT Noktaları ---
    ax1.plot(dates, prices, color="#F8FAFC", linewidth=2.0, label=f"{ticker} Gerçek Fiyat ($)")
    
    # Pozisyonda olunan günleri yarı saydam yeşil ile tara
    in_pos = False
    start_d = None
    for i, w in enumerate(weights):
        if w > 0 and not in_pos:
            in_pos = True
            start_d = dates[i]
        elif w == 0 and in_pos:
            in_pos = False
            ax1.axvspan(start_d, dates[i], color="#10B981", alpha=0.12)
    if in_pos:
        ax1.axvspan(start_d, dates[-1], color="#10B981", alpha=0.12, label="AI Hissede (Aktif Pozisyon)")

    # AL ve SAT İşaretçileri (Markers & Badges)
    for tr in trades:
        tr_date = datetime.strptime(tr["date"], "%Y-%m-%d")
        tr_price = tr["price"]
        badge = tr["badge"]
        action = tr["action"]
        
        if "ALIM" in action:
            # Yeşil Yukarı Ok
            ax1.scatter(tr_date, tr_price * 0.985, color="#10B981", s=140, marker="^", zorder=5, edgecolors="#08090A", linewidth=1.2)
            ax1.annotate(
                badge,
                (tr_date, tr_price * 0.965),
                fontsize=8.5,
                fontweight="bold",
                color="#10B981",
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#08090A", edgecolor="#10B981", alpha=0.9)
            )
        else:
            # Kırmızı Aşağı Ok
            ax1.scatter(tr_date, tr_price * 1.015, color="#EF4444", s=140, marker="v", zorder=5, edgecolors="#08090A", linewidth=1.2)
            ax1.annotate(
                badge,
                (tr_date, tr_price * 1.035),
                fontsize=8.5,
                fontweight="bold",
                color="#EF4444",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#08090A", edgecolor="#EF4444", alpha=0.9)
            )

    ax1.set_title(f"{title} — 10.000$ Simülasyonu & AL/SAT İşlem Haritası ({sim_data['start_date']} -> {sim_data['end_date']})",
                  fontsize=14, fontweight="bold", color="#F8FAFC", pad=12)
    ax1.set_ylabel("Fiyat ($)", color="#94A3B8", fontsize=11)
    ax1.legend(loc="upper left", framealpha=0.3, facecolor="#08090A", edgecolor="#1F242D")
    ax1.grid(True, linestyle="--", alpha=0.15, color="#334155")
    
    # --- PANEL 2: 10.000$ Portföy Yarışı (AI vs. Al-Tut) ---
    ai_ret = perf["ai_total_return_pct"]
    bh_ret = perf["buy_hold_total_return_pct"]
    sharpe = perf["ai_sharpe"]
    max_dd = perf["ai_max_drawdown_pct"]
    trades_count = perf["total_trades"]
    
    ax2.plot(dates, ai_equity, color="#10B981", linewidth=2.6,
             label=f"NeuroQuant AI 10k: ${ai_equity[-1]:,.2f} (%{ai_ret:+.2f} | Sharpe: {sharpe} | DD: %{max_dd})")
    ax2.plot(dates, bh_equity, color="#64748B", linewidth=1.8, linestyle="--",
             label=f"10k Sabit Al-Tut (Buy & Hold): ${bh_equity[-1]:,.2f} (%{bh_ret:+.2f} | DD: %{perf['buy_hold_max_drawdown_pct']}%)")
    
    ax2.axhline(10000.0, color="#475569", linestyle=":", linewidth=1.2, label="Başlangıç ($10.000)")
    ax2.set_ylabel("Portföy Değeri ($)", color="#94A3B8", fontsize=11)
    ax2.legend(loc="upper left", framealpha=0.3, facecolor="#08090A", edgecolor="#1F242D")
    ax2.grid(True, linestyle="--", alpha=0.15, color="#334155")
    
    # --- PANEL 3: Dinamik Varlık Dağılımı (% Hisse vs % Nakit) & Güven Skoru ---
    ax3.fill_between(dates, 0, weights, color="#10B981", alpha=0.45, label="Hisse Ağırlığı (%)")
    ax3.fill_between(dates, weights, 100, color="#334155", alpha=0.25, label="Nakit Rezervi (%)")
    ax3.plot(dates, weights, color="#10B981", linewidth=1.5)
    
    # İkincil Eksen: Bileşik Güven Skoru
    ax3_conf = ax3.twinx()
    ax3_conf.plot(dates, confidences, color="#38BDF8", linewidth=1.2, linestyle=":", alpha=0.85, label="Bileşik Güven Skoru")
    ax3_conf.axhline(50.0, color="#38BDF8", linestyle="--", alpha=0.25, linewidth=0.8)
    ax3_conf.set_ylabel("Güven Skoru (0-100)", color="#38BDF8", fontsize=10)
    ax3_conf.tick_params(colors="#38BDF8")
    ax3_conf.set_ylim(0, 100)
    
    ax3.set_ylabel("Hisse Oranı (%)", color="#94A3B8", fontsize=10)
    ax3.set_xlabel("Tarih", color="#94A3B8", fontsize=11)
    ax3.set_ylim(0, 105)
    ax3.legend(loc="upper left", framealpha=0.3, facecolor="#08090A", edgecolor="#1F242D")
    ax3.grid(True, linestyle="--", alpha=0.15, color="#334155")
    
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.tight_layout()
    
    os.makedirs("assets", exist_ok=True)
    safe_name = ticker.replace('.', '_').replace('-', '_').lower()
    out_file = f"assets/simulation_10k_{safe_name}.png"
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close()
    
    artifact_copy = os.path.join(ARTIFACT_DIR, f"simulation_10k_{safe_name}.png")
    shutil.copy2(out_file, artifact_copy)
    
    print(f"✅ Kaydedildi: {out_file}")
    print(f"🖼️ Artifact Kopyalandı: {artifact_copy}")
    return out_file, artifact_copy

if __name__ == "__main__":
    cache.clear()
    plot_single_simulation("NVDA", "NVIDIA Corp. (NVDA vs. SMH)")
    plot_single_simulation("BTC-USD", "Bitcoin (BTC-USD 24/7 Market)")
    plot_single_simulation("THYAO.IS", "Türk Hava Yolları (THYAO.IS vs. XU100)")
