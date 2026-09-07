"""
News Scraping & Exponential Time-Decay Sentiment Service
"""

import math
import feedparser
import urllib.parse
import requests
from datetime import datetime, timezone
from typing import Dict, Any, List
from backend.services.cache_service import cache

# Finansal pozitif/negatif anahtar kelimeler (Hızlı ve deterministik skorlayıcı)
POSITIVE_KEYWORDS = [
    "surge", "soar", "jump", "record", "profit", "beat", "rally", "growth",
    "upgrade", "bull", "buy", "gain", "high", "breakthrough", "outperform",
    "dividend", "expansion", "revenue", "yükseliş", "rekor", "kâr", "artış"
]

NEGATIVE_KEYWORDS = [
    "crash", "plunge", "fall", "drop", "loss", "miss", "bear", "sell",
    "warning", "lawsuit", "downgrade", "probe", "investigation", "fraud",
    "cut", "decline", "recession", "slump", "düşüş", "zarar", "soruşturma"
]

def score_headline(text: str) -> float:
    lower_t = text.lower()
    pos_matches = sum(1 for w in POSITIVE_KEYWORDS if w in lower_t)
    neg_matches = sum(1 for w in NEGATIVE_KEYWORDS if w in lower_t)
    
    total = pos_matches + neg_matches
    if total == 0:
        return 0.0
    return (pos_matches - neg_matches) / total

def get_news_and_sentiment(ticker: str, max_results: int = 10) -> Dict[str, Any]:
    clean_ticker = ticker.strip().upper()
    cache_key = f"news_sentiment_{clean_ticker}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    query = urllib.parse.quote(f"{clean_ticker} stock financial news")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/114.0.0.0 Safari/537.36"
    }
    
    news_items: List[Dict[str, Any]] = []
    
    try:
        res = requests.get(rss_url, headers=headers, timeout=8)
        if res.status_code == 200:
            feed = feedparser.parse(res.content)
            now_dt = datetime.now(timezone.utc)
            
            for entry in feed.entries[:max_results]:
                dt_obj = now_dt
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    dt_obj = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    
                elapsed_hours = max((now_dt - dt_obj).total_seconds() / 3600.0, 0.1)
                
                # 24 Saat Yarılanma Ömürlü Üstel Ağırlık: w = exp(-lambda * dt)
                decay_weight = math.exp(-0.0288 * elapsed_hours)
                
                headline = entry.title
                score = score_headline(headline)
                
                label = "POZİTİF" if score > 0.15 else "NEGATİF" if score < -0.15 else "NÖTR"
                
                news_items.append({
                    "title": headline,
                    "link": entry.link,
                    "source": entry.source.title if hasattr(entry, 'source') else "Google News",
                    "published": dt_obj.strftime("%Y-%m-%d %H:%M"),
                    "elapsed_hours": round(elapsed_hours, 1),
                    "decay_weight": round(decay_weight, 3),
                    "score": round(score, 2),
                    "label": label
                })
    except Exception as e:
        print(f"Haber çekme uyarısı: {e}")

    # Ağırlıklı Ortalama Duygu Skoru
    if news_items:
        total_w = sum(n["decay_weight"] for n in news_items)
        weighted_score = sum(n["score"] * n["decay_weight"] for n in news_items) / (total_w + 1e-8)
    else:
        weighted_score = 0.0

    weighted_score = float(max(min(weighted_score, 1.0), -1.0))
    overall_label = "POZİTİF" if weighted_score > 0.12 else "NEGATİF" if weighted_score < -0.12 else "NÖTR"
    
    riskiest = next((n for n in news_items if n["score"] <= -0.5), None)
    catalyst = next((n for n in news_items if n["score"] >= 0.5), None)

    result = {
        "ticker": clean_ticker,
        "total_news_count": len(news_items),
        "overall_sentiment_score": round(weighted_score, 2),
        "overall_label": overall_label,
        "riskiest_headline": riskiest["title"] if riskiest else None,
        "top_catalyst_headline": catalyst["title"] if catalyst else None,
        "news": news_items
    }

    cache.set(cache_key, result, ttl=900)
    return result
