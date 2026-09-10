"""
News Scraping & Continuous Time-Decay Sentiment Service (Loughran-McDonald & VADER inspired)
"""

import math
import feedparser
import urllib.parse
import requests
from datetime import datetime, timezone
from typing import Dict, Any, List
from backend.services.cache_service import cache

# Dereceli finansal pozitif / negatif sözlük (Şiddet katsayıları)
LEXICON_POS = {
    # Yüksek Şiddet (1.4 - 1.8)
    "soar": 1.6, "surge": 1.5, "record": 1.4, "blowout": 1.8, "breakthrough": 1.5,
    "skyrocket": 1.7, "outperform": 1.4, "rekor": 1.5, "patlama": 1.4,
    # Orta Şiddet (0.8 - 1.2)
    "beat": 1.0, "rally": 1.0, "growth": 0.9, "upgrade": 1.1, "bull": 0.9,
    "gain": 0.8, "profit": 1.0, "expansion": 0.9, "dividend": 0.8, "kâr": 1.0, "artış": 0.8,
    # Ilımlı / Destekleyici (0.4 - 0.7)
    "rise": 0.5, "higher": 0.5, "advance": 0.6, "steady": 0.4, "recovery": 0.7,
    "optimism": 0.6, "up": 0.4, "iyileşme": 0.6, "güçlü": 0.7
}

LEXICON_NEG = {
    # Yüksek Şiddet (1.4 - 1.8)
    "crash": 1.8, "plunge": 1.6, "collapse": 1.7, "fraud": 1.8, "bankruptcy": 1.8,
    "meltdown": 1.7, "rout": 1.5, "çöküş": 1.8, "iflas": 1.8, "dolandırıcılık": 1.8,
    # Orta Şiddet (0.8 - 1.2)
    "loss": 1.0, "miss": 1.0, "bear": 0.9, "selloff": 1.1, "warning": 1.0,
    "lawsuit": 1.1, "downgrade": 1.1, "probe": 1.0, "investigation": 1.0,
    "recession": 1.2, "slump": 1.0, "zarar": 1.0, "soruşturma": 1.0, "düşüş": 0.8,
    # Ilımlı / Risk Faktörleri (0.4 - 0.7)
    "fall": 0.5, "drop": 0.6, "cut": 0.6, "decline": 0.6, "slip": 0.5,
    "dip": 0.4, "down": 0.4, "concern": 0.5, "worry": 0.5, "headwind": 0.6,
    "endise": 0.5, "baskı": 0.6
}

def score_headline(text: str) -> float:
    """
    Hiperbolik tanjant tabanlı sürekli ve yumuşak duygu skorlayıcı.
    Skoru asla keyfi olarak -1 veya +1'e sabitlemez; kelime yoğunluğuna göre
    -1.0 ile +1.0 arasında pürüzsüz ve gerçekçi bir dağılım üretir.
    """
    lower_t = text.lower()
    
    pos_score = sum(weight for word, weight in LEXICON_POS.items() if word in lower_t)
    neg_score = sum(weight for word, weight in LEXICON_NEG.items() if word in lower_t)
    
    diff = pos_score - neg_score
    if diff == 0.0:
        return 0.0
    
    # Skalayı 2.5 sabitiyle normalize ederek tanh eğrisine aktarır
    smooth_score = math.tanh(diff / 2.5)
    return round(float(smooth_score), 2)


def extract_top_3_impactful(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Tüm haberler arasından etki katsayısına (|Skor| * Zaman Ağırlığı) göre
    en belirleyici ve kritik ilk 3 haberi seçer.
    """
    if not news_items:
        return []

    def compute_impact(item: Dict[str, Any]) -> float:
        score_mag = abs(item.get("score", 0.0))
        decay = item.get("decay_weight", 1.0)
        return score_mag * (0.55 + 0.45 * decay)

    sorted_items = sorted(news_items, key=compute_impact, reverse=True)
    top_3 = []
    
    for item in sorted_items[:3]:
        score = item["score"]
        if score >= 0.20:
            impact_type = "GÜÇLÜ KATALİZÖR"
            badge_color = "#14532D"
        elif score <= -0.20:
            impact_type = "KRİTİK RİSK"
            badge_color = "#881337"
        else:
            impact_type = "MAKRO AKIŞ"
            badge_color = "#1E3A8A"
            
        top_3.append({
            "title": item["title"],
            "link": item["link"],
            "source": item["source"],
            "published": item["published"],
            "elapsed_hours": item["elapsed_hours"],
            "score": score,
            "impact_type": impact_type,
            "badge_color": badge_color,
        })
        
    return top_3


def get_news_and_sentiment(ticker: str, max_results: int = 15) -> Dict[str, Any]:
    """
    Hisseye özel haberleri çeker, tüm evren üzerinden zaman çürümeli genel skoru hesaplar,
    ve frontend için en belirleyici ilk 3 haberi (top_3_impactful) filtreler.
    """
    clean_ticker = ticker.strip().upper()
    cache_key = f"news_sentiment_{clean_ticker}_v4"
    cached = cache.get(cache_key)
    if cached:
        return cached

    query = urllib.parse.quote(f"{clean_ticker} stock earnings financial news")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
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
                # 24 Saat Yarılanma Ömürlü Üstel Ağırlık
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
                    "score": score,
                    "label": label
                })
    except Exception as e:
        print(f"Haber çekme uyarısı: {e}")

    # Ağırlıklı Ortalama Duygu Skoru (Tüm haberler hesaba katılır)
    if news_items:
        total_w = sum(n["decay_weight"] for n in news_items)
        weighted_score = sum(n["score"] * n["decay_weight"] for n in news_items) / (total_w + 1e-8)
    else:
        weighted_score = 0.0

    weighted_score = float(max(min(weighted_score, 1.0), -1.0))
    overall_label = "POZİTİF" if weighted_score > 0.12 else "NEGATİF" if weighted_score < -0.12 else "NÖTR"
    
    top_3 = extract_top_3_impactful(news_items)

    result = {
        "ticker": clean_ticker,
        "total_news_count": len(news_items),
        "overall_sentiment_score": round(weighted_score, 2),
        "overall_label": overall_label,
        "top_3_impactful": top_3,
        "news": news_items
    }

    cache.set(cache_key, result, ttl=900)
    return result


def get_global_macro_news(max_results: int = 15) -> Dict[str, Any]:
    """
    Wall Street, Fed faiz kararları, enflasyon ve küresel ekonomi haberlerini çeker
    ve üstel zaman çürümeli ağırlıklı duygu skoru hesaplar.
    """
    cache_key = "news_sentiment_global_macro_v4"
    cached = cache.get(cache_key)
    if cached:
        return cached

    query = urllib.parse.quote("wall street stock market economy fed inflation interest rates")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
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
                    "score": score,
                    "label": label
                })
    except Exception as e:
        print(f"Küresel haber çekme uyarısı: {e}")

    if news_items:
        total_w = sum(n["decay_weight"] for n in news_items)
        weighted_score = sum(n["score"] * n["decay_weight"] for n in news_items) / (total_w + 1e-8)
    else:
        weighted_score = 0.0

    weighted_score = float(max(min(weighted_score, 1.0), -1.0))
    overall_label = "POZİTİF" if weighted_score > 0.10 else "NEGATİF" if weighted_score < -0.10 else "NÖTR"

    top_3 = extract_top_3_impactful(news_items)

    result = {
        "ticker": "MACRO",
        "title": "Küresel Makro & Wall Street",
        "total_news_count": len(news_items),
        "overall_sentiment_score": round(weighted_score, 2),
        "overall_label": overall_label,
        "top_3_impactful": top_3,
        "news": news_items
    }

    cache.set(cache_key, result, ttl=900)
    return result

