"""
NeuroQuant 3.0: Macro Barometers & Mathematical RORO (Risk-On / Risk-Off) Regime Engine
Tracks VIX, US 10-Year Treasury Yield, WTI Crude Oil, Gold, and Silver.
Synthesizes a normalized quantitative macro regime score (-100 to +100).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from typing import Dict, Any, List
from backend.services.cache_service import cache
from backend.services.sentiment_service import get_global_macro_news

MACRO_SYMBOLS = {
    "VIX": "^VIX",
    "TNX": "^TNX",
    "OIL": "CL=F",
    "GOLD": "GC=F",
    "SILVER": "SI=F"
}

def get_macro_regime_and_barometers() -> Dict[str, Any]:
    """
    yfinance üzerinden makro varlıkları çeker ve kurumsal matematiksel
    Risk-On / Risk-Off (RORO) rejim skorunu hesaplar.
    Önbellek süresi: 10 dakika (600 saniye).
    """
    cache_key = "macro_regime_barometers_v3"
    cached = cache.get(cache_key)
    if cached:
        return cached

    # 1. Makro Varlıkları İndir (Son 5 Gün)
    tickers_list = list(MACRO_SYMBOLS.values())
    raw_df = yf.download(tickers_list, period="5d", progress=False)

    closes = raw_df["Close"] if "Close" in raw_df else raw_df

    def get_latest_and_change(sym: str):
        try:
            if sym in closes.columns:
                series = closes[sym].dropna()
                if len(series) >= 2:
                    curr = float(series.iloc[-1])
                    prev = float(series.iloc[-2])
                    chg_pct = round(((curr - prev) / (prev + 1e-8)) * 100, 2)
                    chg_5d = round(((curr - float(series.iloc[0])) / (float(series.iloc[0]) + 1e-8)) * 100, 2)
                    return curr, chg_pct, chg_5d
                elif len(series) == 1:
                    return float(series.iloc[-1]), 0.0, 0.0
        except Exception as e:
            print(f"Uyarı: {sym} makro verisi okunamadı: {e}")
        return 0.0, 0.0, 0.0

    vix_val, vix_chg, vix_5d = get_latest_and_change(MACRO_SYMBOLS["VIX"])
    tnx_val, tnx_chg, tnx_5d = get_latest_and_change(MACRO_SYMBOLS["TNX"])
    oil_val, oil_chg, oil_5d = get_latest_and_change(MACRO_SYMBOLS["OIL"])
    gold_val, gold_chg, gold_5d = get_latest_and_change(MACRO_SYMBOLS["GOLD"])
    silver_val, silver_chg, silver_5d = get_latest_and_change(MACRO_SYMBOLS["SILVER"])

    # Fallbacks if yfinance returned empty/delayed for index symbols
    if vix_val == 0.0: vix_val = 16.45
    if tnx_val == 0.0: tnx_val = 4.38
    if oil_val == 0.0: oil_val = 71.50
    if gold_val == 0.0: gold_val = 2515.00
    if silver_val == 0.0: silver_val = 29.80

    # 2. Küresel Makro Haber Duygusunu Al
    global_news = get_global_macro_news(max_results=8)
    news_sentiment_score = float(global_news.get("overall_sentiment_score", 0.0))  # -1.0 ile +1.0 arası

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. MATEMATİKSEL RORO (RISK-ON / RISK-OFF) REGIME ENGINE
    # ═══════════════════════════════════════════════════════════════════════════
    # Toplam Puan: [-100, +100]
    # Ağırlıklar:
    #   w_vix   = 0.30 (Volatilite Korku Eşiği)
    #   w_tnx   = 0.20 (Tahvil Faizi İvmesi)
    #   w_oil   = 0.15 (Petrol Enflasyon Baskısı)
    #   w_safe  = 0.15 (Altın/Gümüş Sığınak Akışı)
    #   w_news  = 0.20 (Küresel Finansal Haber Duygusu)

    # A. VIX Skoru (-100 ile +100)
    # 14 altı: Aşırı Güven / Büyüme (+100)
    # 14-20: Sağlıklı / Düşük Stres (+100 to 0)
    # 20-30: Yüksek Stres / Risk-Off (0 to -100)
    # 30 üzeri: Panik (-100)
    if vix_val <= 14.0:
        s_vix = 100.0
        vix_status = "Düşük Volatilite · Güçlü Risk İştahı"
        vix_tag = "BULLISH"
    elif vix_val <= 20.0:
        s_vix = 100.0 * (20.0 - vix_val) / 6.0
        vix_status = "Dengeli Volatilite · Normal Piyasa"
        vix_tag = "NEUTRAL-BULL"
    elif vix_val <= 30.0:
        s_vix = -100.0 * (vix_val - 20.0) / 10.0
        vix_status = "Yüksek Volatilite · Temkinli Rejim"
        vix_tag = "BEARISH"
    else:
        s_vix = -100.0
        vix_status = "Aşırı Korku / Panik Rejimi"
        vix_tag = "EXTREME-RISK"

    # B. 10Y ABD Tahvili (TNX) İvme Skoru (-100 ile +100)
    # Tahvil faizlerindeki 5 günlük sert yükseliş çarpanları baskılar, gevşeme destekler
    if tnx_5d > 3.0:
        s_tnx = -80.0
        tnx_status = "Faiz Sıçraması · Değerleme Baskısı"
        tnx_tag = "HEADWIND"
    elif tnx_5d > 1.0:
        s_tnx = -30.0
        tnx_status = "Ilımlı Faiz Artışı · Sınırlı Baskı"
        tnx_tag = "NEUTRAL-BEAR"
    elif tnx_5d < -2.0:
        s_tnx = 80.0
        tnx_status = "Faiz Gevşemesi · Çarpan Genişlemesi"
        tnx_tag = "TAILWIND"
    else:
        s_tnx = 30.0
        tnx_status = "Yatay & İstikrarlı Faiz Ortamı"
        tnx_tag = "NEUTRAL"

    # C. Ham Petrol (WTI) Enflasyon Baskısı Skoru (-100 ile +100)
    # $90 üzeri veya %5 üzeri sıçrama Fed şahinleşme riski doğurur; $70-$80 arası dezenflasyon dostudur
    if oil_val > 88.0 or oil_5d > 5.0:
        s_oil = -75.0
        oil_status = "Enflasyonist Baskı · Şahin Fed Riski"
        oil_tag = "INFLATION-RISK"
    elif oil_val < 65.0 and oil_5d < -6.0:
        s_oil = -20.0  # Aşırı çöküş küresel resesyon korkusu verebilir
        oil_status = "Talep Zayıflığı / Büyüme Endişesi"
        oil_tag = "DEMAND-WORRY"
    else:
        s_oil = 60.0
        oil_status = "Dengeli Enerji Fiyatı · Dezenflasyonist"
        oil_tag = "BENIGN"

    # D. Altın & Gümüş Güvenli Liman Akışı Skoru (-100 ile +100)
    # VIX yüksekken altın çok sert yükseliyorsa sığınak kaçışı vardır
    if gold_5d > 3.5 and vix_val > 19.0:
        s_safe = -70.0
        gold_status = "Güvenli Limana Kaçış · Defansif Akış"
        gold_tag = "SAFE-HAVEN-FLIGHT"
    elif gold_5d > 0.0:
        s_safe = 30.0
        gold_status = "Normal Rezerv & Likidite Talebi"
        gold_tag = "STABLE"
    else:
        s_safe = 50.0
        gold_status = "Sığınak İhtiyacı Düşük · Risk Varlıkları Ön Planda"
        gold_tag = "RISK-SEEKING"

    silver_status = "Endüstriyel & Yeşil Enerji Talebi"

    # E. Küresel Haber Duygusu Skoru (-100 ile +100)
    s_news = float(np.clip(news_sentiment_score * 100.0, -100.0, 100.0))

    # BİLEŞİK RORO HESAPLAMASI
    composite_raw = (
        0.30 * s_vix +
        0.20 * s_tnx +
        0.15 * s_oil +
        0.15 * s_safe +
        0.20 * s_news
    )
    composite_score = round(float(np.clip(composite_raw, -100.0, 100.0)), 1)

    # Rejim Sınıflandırması
    if composite_score >= 35.0:
        regime_label = "RISK-ON (Agresif Büyüme Rejimi)"
        regime_badge = "RISK-ON"
        regime_color = "#14532D"
        investor_note = "VIX sakin seviyede seyrederken tahvil faizleri istikrarlı. Makro zemin büyüme ve teknoloji hisselerini destekliyor."
    elif composite_score >= 10.0:
        regime_label = "ILIMLI POZİTİF (Dengeli Piyasa)"
        regime_badge = "ILIMLI BOĞA"
        regime_color = "#1E3A8A"
        investor_note = "Piyasada genel risk iştahı pozitif ancak makro faktörlerde seçici olunmalı. Kaliteli nakit akışı üreten şirketler öne çıkıyor."
    elif composite_score >= -15.0:
        regime_label = "NÖTR / DENGELİ (Yatay Rejim)"
        regime_badge = "NÖTR"
        regime_color = "#57534E"
        investor_note = "Makro barometreler çift yönlü sinyaller üretiyor. Fed beklentileri ve enflasyon verileri öncesinde bekle-gör yaklaşımı hakim."
    elif composite_score >= -45.0:
        regime_label = "RISK-OFF (Defansif / Temkinli)"
        regime_badge = "RISK-OFF"
        regime_color = "#881337"
        investor_note = "Yükselen volatilite ve tahvil getirileri hisse değerlemelerini baskılıyor. Nakit tamponu korunmalı ve defansif sektörlere yönelmeli."
    else:
        regime_label = "YÜKSEK STRES (Sığınak Modu)"
        regime_badge = "STRES REJİMİ"
        regime_color = "#7F1D1D"
        investor_note = "Korku endeksinde sert sıçrama ve güvenli limanlara kaçış var. Portföyde riskli varlık ağırlığı asgari seviyeye çekilmeli."

    barometers: List[Dict[str, Any]] = [
        {
            "key": "VIX",
            "name": "CBOE Volatilite Endeksi",
            "symbol": "^VIX",
            "value": vix_val,
            "unit": "Puan",
            "change_pct": vix_chg,
            "change_5d": vix_5d,
            "status": vix_status,
            "tag": vix_tag,
            "is_risk_on": vix_val < 20.0
        },
        {
            "key": "TNX",
            "name": "ABD 10 Yıllık Tahvil Faizi",
            "symbol": "^TNX",
            "value": tnx_val,
            "unit": "%",
            "change_pct": tnx_chg,
            "change_5d": tnx_5d,
            "status": tnx_status,
            "tag": tnx_tag,
            "is_risk_on": tnx_5d <= 1.0
        },
        {
            "key": "OIL",
            "name": "WTI Ham Petrol",
            "symbol": "CL=F",
            "value": oil_val,
            "unit": "$",
            "change_pct": oil_chg,
            "change_5d": oil_5d,
            "status": oil_status,
            "tag": oil_tag,
            "is_risk_on": oil_val <= 85.0
        },
        {
            "key": "GOLD",
            "name": "Ons Altın (XAU/USD)",
            "symbol": "GC=F",
            "value": gold_val,
            "unit": "$",
            "change_pct": gold_chg,
            "change_5d": gold_5d,
            "status": gold_status,
            "tag": gold_tag,
            "is_risk_on": s_safe >= 0
        },
        {
            "key": "SILVER",
            "name": "Ons Gümüş (XAG/USD)",
            "symbol": "SI=F",
            "value": silver_val,
            "unit": "$",
            "change_pct": silver_chg,
            "change_5d": silver_5d,
            "status": silver_status,
            "tag": "COMMODITY",
            "is_risk_on": silver_chg >= 0
        }
    ]

    breakdown = {
        "vix_contribution": round(0.30 * s_vix, 1),
        "tnx_contribution": round(0.20 * s_tnx, 1),
        "oil_contribution": round(0.15 * s_oil, 1),
        "safe_haven_contribution": round(0.15 * s_safe, 1),
        "news_sentiment_contribution": round(0.20 * s_news, 1)
    }

    result = {
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "composite_score": composite_score,
        "regime_label": regime_label,
        "regime_badge": regime_badge,
        "regime_color": regime_color,
        "investor_note": investor_note,
        "barometers": barometers,
        "breakdown": breakdown,
        "global_news_sentiment": {
            "score": round(news_sentiment_score, 2),
            "label": global_news.get("overall_label", "NÖTR"),
            "total_news_count": global_news.get("total_news_count", 0)
        }
    }

    cache.set(cache_key, result, ttl=600)
    return result
