"""
Gemini Structured Strategist Report Service
"""

import os
import json
from typing import Dict, Any, Optional
import google.generativeai as genai
from backend.config import settings
from backend.services.cache_service import cache

def generate_strategist_report(
    ticker: str,
    market_data: Dict[str, Any],
    forecast_data: Dict[str, Any],
    fundamental_data: Dict[str, Any],
    sentiment_data: Dict[str, Any]
) -> Dict[str, Any]:
    clean_ticker = ticker.strip().upper()
    cache_key = f"agent_report_{clean_ticker}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    api_key = os.getenv("GEMINI_API_KEY", settings.GEMINI_API_KEY)
    
    # API Key yoksa akıllı yerel kurumsal şablon üret
    if not api_key:
        fallback = _generate_deterministic_report(
            clean_ticker, market_data, forecast_data, fundamental_data, sentiment_data
        )
        cache.set(cache_key, fallback, ttl=600)
        return fallback

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')

        prompt = f"""
        Sen Wall Street'te görev yapan kıdemli bir kantitatif portföy yöneticisi ve stratejistsin.
        Aşağıdaki verileri inceleyip {clean_ticker} için kurumsal düzeyde yapılandırılmış bir yatırım raporu üret.
        Yanıtını SADECE geçerli bir JSON nesnesi olarak ver (başka markdown veya açıklama ekleme):

        VERİLER:
        - Fiyat: {market_data.get('current_price')} (24s Değişim: %{market_data.get('change_pct')})
        - 200 Günlük SMA Mesafesi: %{market_data.get('dist_sma200_pct')} | Golden Cross: {market_data.get('is_golden_cross')}
        - Sektörel Alfa (20g): %{market_data.get('alpha_20d_cum')} | Beta: {market_data.get('beta')}
        - Quant Model 5-Günlük Medyan Beklenti: %{forecast_data.get('median_5d_return_pct')} (Yükseliş Olasılığı: %{forecast_data.get('up_probability')})
        - %80 Güven Konisi: %{forecast_data.get('lower_80_return_pct')} ile %{forecast_data.get('upper_80_return_pct')}
        - Temel Değerleme: {fundamental_data.get('valuation_status')} (F/K: {fundamental_data.get('trailing_pe')}, Forward F/K: {fundamental_data.get('forward_pe')}, PEG: {fundamental_data.get('peg_ratio')})
        - Bilanço Rejimi: {fundamental_data.get('earnings_regime')} (Kalan Gün: {fundamental_data.get('days_to_earnings')})
        - Haber Duygusu: {sentiment_data.get('overall_label')} (Skor: {sentiment_data.get('overall_sentiment_score')})
        - En Riskli Başlık: {sentiment_data.get('riskiest_headline')}
        - En Güçlü Katalizör: {sentiment_data.get('top_catalyst_headline')}

        İSTENEN JSON ŞEMASI:
        {{
            "executive_summary": "1-2 cümlelik nihai portföy kararı ve ana özet",
            "technical_regime": "Makro trend, 200 SMA ve volatilite değerlendirmesi",
            "fundamental_valuation": "Çarpanlar, kâr büyümesi ve bilanço beklentisi",
            "sentiment_and_catalysts": "Haber akışı ve kurumsal para girişleri",
            "risk_factors": ["Risk faktörü 1", "Risk faktörü 2"],
            "suggested_action": "Örn: Kademeli %50 (5k$) Alım, 200 SMA altına inmedikçe taşıma"
        }}
        """

        response = model.generate_content(prompt)
        text_resp = response.text.strip()
        if text_resp.startswith("```json"):
            text_resp = text_resp[7:]
        if text_resp.startswith("```"):
            text_resp = text_resp[3:]
        if text_resp.endswith("```"):
            text_resp = text_resp[:-3]

        parsed = json.loads(text_resp.strip())
        cache.set(cache_key, parsed, ttl=1800)
        return parsed
    except Exception as e:
        print(f"Gemini API çağrısı yerel şablona devredildi: {e}")
        fallback = _generate_deterministic_report(
            clean_ticker, market_data, forecast_data, fundamental_data, sentiment_data
        )
        cache.set(cache_key, fallback, ttl=300)
        return fallback

def _generate_deterministic_report(
    ticker: str,
    market: Dict[str, Any],
    forecast: Dict[str, Any],
    fund: Dict[str, Any],
    sent: Dict[str, Any]
) -> Dict[str, Any]:
    """Gemini API yokken veya hata anında dönen deterministik profesyonel quant raporu."""
    is_gc = market.get("is_golden_cross", False)
    above_200 = market.get("is_above_sma200", False)
    med_ret = forecast.get("median_5d_return_pct", 0.0)
    up_prob = forecast.get("up_probability", 50.0)
    val_status = fund.get("valuation_status", "MAKUL")
    regime = fund.get("earnings_regime", "NORMAL REJİM")
    
    if above_200 and med_ret > 1.0 and up_prob > 60:
        summary = f"{ticker} için teknik rejim güçlü boğa bölgesinde. Fiyat 200 SMA üzerinde ve quant model %{med_ret:+.1f} yukarı potansiyel öngörüyor."
        action = "Kademeli Alım (5k - 7.5k). 200 günlük ortalama üzerinde kalındığı sürece pozisyon korunmalı."
    elif not above_200 and med_ret < 0:
        summary = f"{ticker} kurumsal Death Cross / ayı rejiminde. Sermaye koruması öncelikli tutulmalı."
        action = "Nakit Ağırlıklı Bekle-Gör (0k - 2k). Tepki alımlarında düşen bıçağı tutmaktan kaçının."
    else:
        summary = f"{ticker} nötr-dengeli rejimde hareket ediyor. Temel çarpanlar {val_status.lower()} seviyede."
        action = "Piyasa teyidi beklenmeli; mevcut hisse ağırlığı %50 seviyesinde tutulabilir."

    return {
        "executive_summary": summary,
        "technical_regime": f"Fiyat 200 SMA'ya göre %{market.get('dist_sma200_pct', 0.0):+.1f} mesafede. Golden Cross durumu: {'Aktif' if is_gc else 'Pasif'}. Beta katsayısı: {market.get('beta', 1.0)}.",
        "fundamental_valuation": f"Hisse {val_status} olarak değerlendirildi. F/K: {fund.get('trailing_pe', '-')}, Forward F/K: {fund.get('forward_pe', '-')}. Bilanço rejimi: {regime}.",
        "sentiment_and_catalysts": f"Haber akışı {sent.get('overall_label', 'NÖTR')} ağırlıklı (Skor: {sent.get('overall_sentiment_score', 0.0)}).",
        "risk_factors": [
            "200 günlük ortalamanın altına sarkma halinde stop-loss disiplini",
            f"Bilanço yaklaşma sürecinde artan volatilite ({regime})"
        ],
        "suggested_action": action
    }
