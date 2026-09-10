"""
NeuroQuant 3.0: Macro Barometers & Quantitative Regime Engine
Integrates:
- CBOE Volatility (VIX)
- US 10-Year Treasury Yield (TNX) with Structural Long-Term Trend (SMA50, SMA200, 60d Trend)
- Fed Rate Stance Engine (3M T-Bill IRX, 10Y-3M Yield Curve Spread, Rate Momentum)
- CNN Official Fear & Greed Index (Live Scraper from CNN Dataviz API)
- WTI Crude Oil (Inflation/De-inflation pressure)
- Continuous Wall Street Macro News Sentiment
"""

import json
import urllib.request
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from typing import Dict, Any, List
from backend.services.cache_service import cache
from backend.services.sentiment_service import get_global_macro_news


def get_cnn_fear_and_greed() -> Dict[str, Any]:
    """
    CNN'in resmi dataviz API'sinden canlı Fear & Greed Endeksi'ni çeker.
    Skor (0-100), derece (Korku / Açgözlülük vb.) ve haftalık değişimi döndürür.
    """
    cache_key = "cnn_fear_and_greed_live_v2"
    cached = cache.get(cache_key)
    if cached:
        return cached

    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.cnn.com",
        "Referer": "https://www.cnn.com/markets/fear-and-greed",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }

    try:
        res = requests.get(url, headers=headers, timeout=6)
        if res.status_code == 200:
            data = res.json()
            fg = data.get("fear_and_greed", {})
            score = round(float(fg.get("score", 50.0)), 1)
            rating_raw = str(fg.get("rating", "neutral")).lower()
            prev_close = round(float(fg.get("previous_close", score)), 1)
            prev_1w = round(float(fg.get("previous_1_week", score)), 1)
            chg_1w = round(score - prev_1w, 1)

            rating_map = {
                "extreme fear": "Aşırı Korku",
                "fear": "Korku",
                "neutral": "Nötr",
                "greed": "Açgözlülük",
                "extreme greed": "Aşırı Açgözlülük"
            }
            rating_tr = rating_map.get(rating_raw, rating_raw.title())

            result = {
                "score": score,
                "rating": rating_tr,
                "rating_raw": rating_raw,
                "previous_close": prev_close,
                "change_1w": chg_1w,
                "is_live": True
            }
            cache.set(cache_key, result, ttl=900)
            return result
    except Exception as e:
        print(f"CNN Fear & Greed API uyarısı: {e}")
        return {
            "score": 45.0,
            "rating": "Nötr / Dengeli",
            "rating_raw": "neutral",
            "previous_close": 45.0,
            "change_1w": -2.0,
            "is_live": False
        }


def get_macro_regime_and_barometers() -> Dict[str, Any]:
    """
    Tüm makro göstergeleri sentezler:
    - VIX
    - 10Y Tahvil (Uzun vadeli trend, SMA50/SMA200, 60G ivmesi)
    - Fed Faiz Duruşu (3M IRX faiz, 10Y-3M Verim Eğrisi, Politika Eğilimi)
    - CNN Fear & Greed Endeksi
    - WTI Ham Petrol
    - Küresel Duygu Skoru
    """
    cache_key = "macro_regime_barometers_v4"
    cached = cache.get(cache_key)
    if cached:
        return cached

    # 1. VIX, TNX, IRX, OIL için verileri indir
    try:
        vix_df = yf.Ticker("^VIX").history(period="1mo")
        oil_df = yf.Ticker("CL=F").history(period="1mo")
        tnx_df = yf.Ticker("^TNX").history(period="1y")
        irx_df = yf.Ticker("^IRX").history(period="6mo")
    except Exception as e:
        print(f"Makro veri indirme hatası: {e}")
        vix_df, oil_df, tnx_df, irx_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # A. VIX Değerleri
    if not vix_df.empty and len(vix_df) >= 2:
        vix_val = round(float(vix_df["Close"].iloc[-1]), 2)
        vix_chg = round(((vix_val - float(vix_df["Close"].iloc[-2])) / float(vix_df["Close"].iloc[-2])) * 100, 2)
        vix_5d = round(((vix_val - float(vix_df["Close"].iloc[-5])) / float(vix_df["Close"].iloc[-5])) * 100, 2) if len(vix_df) >= 5 else 0.0
    else:
        vix_val, vix_chg, vix_5d = 16.50, 0.0, 0.0

    # B. WTI Ham Petrol Değerleri
    if not oil_df.empty and len(oil_df) >= 2:
        oil_val = round(float(oil_df["Close"].iloc[-1]), 2)
        oil_chg = round(((oil_val - float(oil_df["Close"].iloc[-2])) / float(oil_df["Close"].iloc[-2])) * 100, 2)
        oil_5d = round(((oil_val - float(oil_df["Close"].iloc[-5])) / float(oil_df["Close"].iloc[-5])) * 100, 2) if len(oil_df) >= 5 else 0.0
    else:
        oil_val, oil_chg, oil_5d = 71.50, 0.0, 0.0

    # C. 10Y ABD Tahvili (TNX) Uzun Vadeli Trend & İvme
    if not tnx_df.empty and len(tnx_df) >= 20:
        tnx_val = round(float(tnx_df["Close"].iloc[-1]), 3)
        tnx_chg = round(((tnx_val - float(tnx_df["Close"].iloc[-2])) / float(tnx_df["Close"].iloc[-2])) * 100, 2)
        tnx_5d = round(((tnx_val - float(tnx_df["Close"].iloc[-5])) / float(tnx_df["Close"].iloc[-5])) * 100, 2) if len(tnx_df) >= 5 else 0.0
        
        # 50 ve 200 Günlük Hareketli Ortalamalar
        sma50_val = tnx_df["Close"].rolling(50).mean().iloc[-1]
        sma200_val = tnx_df["Close"].rolling(200).mean().iloc[-1]
        tnx_sma50 = round(float(sma50_val), 3) if not np.isnan(sma50_val) else tnx_val
        tnx_sma200 = round(float(sma200_val), 3) if not np.isnan(sma200_val) else tnx_val
        
        # 60 Günlük (Çeyreklik) Yapısal Trend
        tnx_60d = round(((tnx_val - float(tnx_df["Close"].iloc[-60])) / float(tnx_df["Close"].iloc[-60])) * 100, 2) if len(tnx_df) >= 60 else tnx_5d
        dist_sma200 = round(((tnx_val - tnx_sma200) / tnx_sma200) * 100, 2)
    else:
        tnx_val, tnx_chg, tnx_5d, tnx_60d = 4.35, 0.0, 0.0, 0.0
        tnx_sma50, tnx_sma200, dist_sma200 = 4.25, 4.10, 6.10

    # D. 3 Aylık Hazine Bonosu (IRX) & Verim Eğrisi
    if not irx_df.empty and len(irx_df) >= 20:
        irx_val = round(float(irx_df["Close"].iloc[-1]), 3)
        irx_chg = round(((irx_val - float(irx_df["Close"].iloc[-2])) / float(irx_df["Close"].iloc[-2])) * 100, 2)
        irx_30d = round(((irx_val - float(irx_df["Close"].iloc[-30])) / float(irx_df["Close"].iloc[-30])) * 100, 2) if len(irx_df) >= 30 else 0.0
    else:
        irx_val, irx_chg, irx_30d = 4.80, 0.0, -1.5

    # 10Y - 3M Verim Eğrisi Eğimi
    yield_spread = round(tnx_val - irx_val, 3)

    # 2. CNN Fear & Greed Endeksi
    cnn_fg = get_cnn_fear_and_greed()
    fg_score = cnn_fg["score"]

    # 3. Küresel Haber Duygusu
    global_news = get_global_macro_news(max_results=12)
    news_sentiment_score = float(global_news.get("overall_sentiment_score", 0.0))

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. MATEMATİKSEL FAKTÖR PUANLAMALARI (-100 ile +100)
    # ═══════════════════════════════════════════════════════════════════════════

    # A. VIX Skoru (0.25 ağırlık)
    if vix_val <= 14.0:
        s_vix = 100.0
        vix_status = "Düşük Volatilite · Güçlü İştah"
        vix_tag = "DÜŞÜK RİSK"
    elif vix_val <= 20.0:
        s_vix = 100.0 * (20.0 - vix_val) / 6.0
        vix_status = "Dengeli Volatilite · Normal Seyir"
        vix_tag = "ILIMLI"
    elif vix_val <= 28.0:
        s_vix = -100.0 * (vix_val - 20.0) / 8.0
        vix_status = "Yüksek Oynaklık · Temkinli Rejim"
        vix_tag = "YÜKSEK RİSK"
    else:
        s_vix = -100.0
        vix_status = "Panik & Aşırı Stres"
        vix_tag = "PANİK"

    # B. Tahvil Faizi Uzun Vadeli Trend Skoru (0.20 ağırlık)
    # 200 Günlük SMA üstü ve 60 günlük artış çarpanları baskılar
    if tnx_val > tnx_sma200 and tnx_60d > 4.0:
        s_tnx_trend = -80.0
        tnx_status = "SMA 200 Üstü · Yapısal Sıkılaşma"
        tnx_tag = "DEĞERLEME BASKISI"
    elif tnx_val > tnx_sma200:
        s_tnx_trend = -30.0
        tnx_status = "SMA 200 Üzerinde Konsolidasyon"
        tnx_tag = "NÖTR-NEGATİF"
    elif tnx_val < tnx_sma200 and tnx_60d < -4.0:
        s_tnx_trend = 80.0
        tnx_status = "SMA 200 Altı · Çarpan Genişleme Desteği"
        tnx_tag = "LİKİDİTE DOSTU"
    else:
        s_tnx_trend = 30.0
        tnx_status = "Dengeli Uzun Vadeli Bant"
        tnx_tag = "İSTİKRARLI"

    # C. Fed Faiz Duruşu Skoru (0.20 ağırlık)
    # Verim eğrisi (10Y-3M) + Kısa vadeli faiz momentumu (IRX 30G) + Faiz kısıtlayıcılığı
    # 1. Eğim faktörü:
    if yield_spread < -0.30:
        s_slope = -70.0  # Ters Verim Eğrisi (Resesyon riski)
    elif yield_spread < 0.20:
        s_slope = 0.0    # Düzleşen / Geçiş eğrisi
    else:
        s_slope = 70.0   # Normal Pozitif Eğim (Büyüme dostu)

    # 2. İvme faktörü:
    if irx_30d < -1.5:
        s_irx_mom = 75.0  # Piyasa faiz indirimi fiyatlıyor
    elif irx_30d > 1.5:
        s_irx_mom = -65.0 # Faiz artışı / kalıcı sıkılık
    else:
        s_irx_mom = 15.0  # İstikrarlı faiz

    # 3. Kısıtlayıcılık seviyesi:
    if irx_val > 4.75:
        s_irx_lvl = -40.0
    elif irx_val > 3.75:
        s_irx_lvl = 10.0
    else:
        s_irx_lvl = 60.0

    s_fed = round(0.45 * s_slope + 0.35 * s_irx_mom + 0.20 * s_irx_lvl, 1)

    if s_fed >= 25.0:
        fed_stance_label = "GÜVERCİN (Faiz İndirimi Beklentisi)"
        fed_badge = "GÜVERCİN"
        fed_tag = "LİKİDİTE DESTEĞİ"
    elif s_fed >= -20.0:
        fed_stance_label = "NÖTR / DENGELİ (Bekle-Gör Politikası)"
        fed_badge = "NÖTR"
        fed_tag = "BEKLE-GÖR"
    else:
        fed_stance_label = "ŞAHİN / KISITLAYICI (Yüksek Faiz Baskısı)"
        fed_badge = "ŞAHİN"
        fed_tag = "SIKILAŞTIRMA"

    # D. CNN Fear & Greed Skoru (0.15 ağırlık)
    # [0, 100] skalasını [-100, +100] aralığına doğrusal dönüştürme
    s_cnn = round((fg_score - 50.0) * 2.0, 1)

    # E. Ham Petrol Enflasyon Baskısı (0.10 ağırlık)
    if oil_val > 88.0 or oil_5d > 5.0:
        s_oil = -75.0
        oil_status = "Enflasyonist Baskı · Maliyet Riski"
        oil_tag = "ENFLASYON RİSKİ"
    elif oil_val < 62.0 and oil_5d < -6.0:
        s_oil = -20.0
        oil_status = "Talep Zayıflığı / Resesyon Endişesi"
        oil_tag = "BÜYÜME ENDİŞESİ"
    else:
        s_oil = 60.0
        oil_status = "Dengeli Enerji Fiyatı · Dezenflasyonist"
        oil_tag = "DEZENFLASYON"

    # F. Küresel Finans Haber Duygusu (0.10 ağırlık)
    s_news = float(np.clip(news_sentiment_score * 100.0, -100.0, 100.0))

    # BİLEŞİK RORO HESAPLAMASI
    composite_raw = (
        0.25 * s_vix +
        0.20 * s_tnx_trend +
        0.20 * s_fed +
        0.15 * s_cnn +
        0.10 * s_oil +
        0.10 * s_news
    )
    composite_score = round(float(np.clip(composite_raw, -100.0, 100.0)), 1)

    # Rejim Sınıflandırması
    if composite_score >= 35.0:
        regime_label = "RISK-ON (Agresif Büyüme Rejimi)"
        regime_badge = "RISK-ON"
        regime_color = "#14532D"
    elif composite_score >= 10.0:
        regime_label = "ILIMLI POZİTİF (Dengeli Büyüme)"
        regime_badge = "ILIMLI BOĞA"
        regime_color = "#1E3A8A"
    elif composite_score >= -15.0:
        regime_label = "NÖTR / DENGELİ (Yatay Rejim)"
        regime_badge = "NÖTR"
        regime_color = "#57534E"
    elif composite_score >= -45.0:
        regime_label = "RISK-OFF (Defansif / Temkinli)"
        regime_badge = "RISK-OFF"
        regime_color = "#881337"
    else:
        regime_label = "YÜKSEK STRES (Sığınak Modu)"
        regime_badge = "STRES"
        regime_color = "#7F1D1D"

    # Barometre Kartları
    barometers: List[Dict[str, Any]] = [
        {
            "key": "VIX",
            "name": "CBOE Volatilite (VIX)",
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
            "name": "ABD 10Y Tahvil Faizi",
            "symbol": "^TNX",
            "value": tnx_val,
            "unit": "%",
            "change_pct": tnx_chg,
            "change_5d": tnx_5d,
            "trend_60d": tnx_60d,
            "sma50": tnx_sma50,
            "sma200": tnx_sma200,
            "dist_sma200": dist_sma200,
            "status": tnx_status,
            "tag": tnx_tag,
            "is_risk_on": s_tnx_trend >= 0
        },
        {
            "key": "FED",
            "name": "Fed Faiz Duruşu & Eğri",
            "symbol": "10Y - 3M",
            "value": yield_spread,
            "unit": "% Makas",
            "change_pct": irx_30d,
            "irx_rate": irx_val,
            "status": fed_stance_label,
            "tag": fed_tag,
            "fed_badge": fed_badge,
            "is_risk_on": s_fed >= 0
        },
        {
            "key": "FEAR_GREED",
            "name": "CNN Fear & Greed Endeksi",
            "symbol": "CNN:F&G",
            "value": fg_score,
            "unit": "/100",
            "change_pct": cnn_fg["change_1w"],
            "rating": cnn_fg["rating"],
            "status": f"{cnn_fg['rating']} Rejimi",
            "tag": cnn_fg["rating"].upper(),
            "is_risk_on": fg_score >= 50.0
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
        }
    ]

    breakdown = {
        "vix_contribution": round(0.25 * s_vix, 1),
        "tnx_trend_contribution": round(0.20 * s_tnx_trend, 1),
        "fed_stance_contribution": round(0.20 * s_fed, 1),
        "cnn_fg_contribution": round(0.15 * s_cnn, 1),
        "oil_contribution": round(0.10 * s_oil, 1),
        "news_contribution": round(0.10 * s_news, 1)
    }

    result = {
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "composite_score": composite_score,
        "regime_label": regime_label,
        "regime_badge": regime_badge,
        "regime_color": regime_color,
        "fed_stance": {
            "score": s_fed,
            "label": fed_stance_label,
            "badge": fed_badge,
            "yield_spread": yield_spread,
            "short_rate_3m": irx_val
        },
        "cnn_fear_greed": cnn_fg,
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
