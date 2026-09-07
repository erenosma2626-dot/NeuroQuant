"""
Fundamental Multiples, Valuation & Earnings Event Service
"""

import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from backend.services.cache_service import cache

def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Hissenin temel çarpanlarını (F/K, F/DD, PEG) ve bilanço sürpriz/takvim verilerini analiz eder.
    """
    clean_ticker = ticker.strip().upper()
    cache_key = f"fundamentals_{clean_ticker}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    # Kripto varlıklar için çarpanlar uygulanamaz
    if "-USD" in clean_ticker or clean_ticker in ["BTC", "ETH", "SOL"]:
        result = {
            "ticker": clean_ticker,
            "is_equity": False,
            "trailing_pe": None,
            "forward_pe": None,
            "price_to_book": None,
            "peg_ratio": None,
            "market_cap": None,
            "valuation_status": "KRİPTO / EMTİA REJİMİ",
            "valuation_score": 50,  # Nötr
            "earnings_regime": "Uygulanamaz",
            "days_to_earnings": None,
            "next_earnings_date": None,
            "last_eps_surprise_pct": None,
            "earnings_history": []
        }
        cache.set(cache_key, result, ttl=3600)
        return result

    t = yf.Ticker(clean_ticker)
    info = t.info or {}

    trailing_pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    price_to_book = info.get("priceToBook")
    peg_ratio = info.get("pegRatio")
    market_cap = info.get("marketCap")
    
    # Temel Değerleme Skoru (0 - 100): Yüksek puan = Ucuz / İskontolu
    # F/K ve PEG çarpanlarına göre normalize skorlama
    val_score = 50.0  # Nötr başlangıç
    
    if trailing_pe and trailing_pe > 0:
        if trailing_pe < 15:
            val_score += 25
        elif trailing_pe < 25:
            val_score += 10
        elif trailing_pe > 50:
            val_score -= 20
            
    if peg_ratio and peg_ratio > 0:
        if peg_ratio < 1.0:
            val_score += 20  # Büyümesine göre çok ucuz
        elif peg_ratio > 2.5:
            val_score -= 15  # Büyümesine göre pahalı

    val_score = float(np.clip(val_score, 10.0, 95.0))

    if val_score >= 70:
        val_status = "AŞIRI UCUZ / İSKONTOLU"
    elif val_score >= 45:
        val_status = "MAKUL / ADİL DEĞER"
    else:
        val_status = "PAHALI / PRİMLİ"

    # Bilanço Takvimi & Sürpriz Analizi
    earnings_history: List[Dict[str, Any]] = []
    days_to_earnings: Optional[int] = None
    next_earnings_date: Optional[str] = None
    last_eps_surprise_pct: Optional[float] = None
    earnings_regime = "NORMAL REJİM"

    try:
        ed = t.get_earnings_dates(limit=6)
        if ed is not None and not ed.empty:
            now_dt = datetime.now(timezone.utc)
            for idx, row in ed.iterrows():
                # Tarihi timezone-aware yap
                dt = idx.to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                    
                diff_days = (dt - now_dt).days
                surprise = row.get("Surprise(%)")
                surprise_val = float(surprise) if pd.notna(surprise) else None
                
                eps_est = float(row.get("EPS Estimate")) if pd.notna(row.get("EPS Estimate")) else None
                eps_rep = float(row.get("Reported EPS")) if pd.notna(row.get("Reported EPS")) else None
                
                d_str = dt.strftime("%Y-%m-%d")
                
                if diff_days >= 0 and next_earnings_date is None:
                    next_earnings_date = d_str
                    days_to_earnings = diff_days
                elif diff_days < 0 and last_eps_surprise_pct is None and surprise_val is not None:
                    last_eps_surprise_pct = round(surprise_val, 2)
                    
                earnings_history.append({
                    "date": d_str,
                    "days_diff": diff_days,
                    "eps_estimate": eps_est,
                    "reported_eps": eps_rep,
                    "surprise_pct": surprise_val
                })
    except Exception:
        pass

    # Bilanço Rejimi Kuralı
    if days_to_earnings is not None:
        if 0 <= days_to_earnings <= 5:
            earnings_regime = "BİLANÇO ÖNCESİ (RİSK AZALT / DE-RISK)"
        elif -10 <= days_to_earnings < 0:
            if last_eps_surprise_pct and last_eps_surprise_pct > 5.0:
                earnings_regime = "BİLANÇO SONRASI (POZİTİF KATALİZÖR)"
            elif last_eps_surprise_pct and last_eps_surprise_pct < -5.0:
                earnings_regime = "BİLANÇO SONRASI (NEGATİF BASKI)"

    result = {
        "ticker": clean_ticker,
        "is_equity": True,
        "trailing_pe": round(trailing_pe, 2) if trailing_pe else None,
        "forward_pe": round(forward_pe, 2) if forward_pe else None,
        "price_to_book": round(price_to_book, 2) if price_to_book else None,
        "peg_ratio": round(peg_ratio, 2) if peg_ratio else None,
        "market_cap": market_cap,
        "valuation_status": val_status,
        "valuation_score": round(val_score, 1),
        "earnings_regime": earnings_regime,
        "days_to_earnings": days_to_earnings,
        "next_earnings_date": next_earnings_date,
        "last_eps_surprise_pct": last_eps_surprise_pct,
        "earnings_history": earnings_history[:4]
    }

    cache.set(cache_key, result, ttl=1800)
    return result
