"""
Gemini Structured Strategist Report Service
"""

import os
import json
from typing import Dict, Any, List
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
    cache_key = f"agent_report_{clean_ticker}_v4"
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
        model = genai.GenerativeModel('gemini-2.5-flash')

        is_crypto = clean_ticker.endswith("-USD") or not fundamental_data.get("is_equity", True)

        prompt = f"""
        Sen Wall Street'te görev yapan kıdemli bir portföy stratejistisin.
        Aşağıdaki verileri inceleyip {clean_ticker} için sade, net, kullanıcının zekasını küçümsemeyen, listeli bir yatırım değerlendirmesi hazırla.
        Önerini rastgele bir portföy yüzdesi (örn: '%50 tut') olarak değil, taktiksel yaklaşım, kilit seviyeler ve risk yönetimi adımları olarak açıkla.
        Varlık kripto ise ({is_crypto}) anlamsız F/K veya bilanço çarpanı yazma, zincir üstü aktivite ve likidite üzerinden değerlendir.
        Yanıtını SADECE geçerli bir JSON nesnesi olarak ver:

        VERİLER:
        - Fiyat: {market_data.get('current_price')} (24s Değişim: %{market_data.get('change_pct')})
        - 200 Günlük SMA Mesafesi: %{market_data.get('dist_sma200_pct')} | Golden Cross: {market_data.get('is_golden_cross')}
        - Sektörel Alfa (20g): %{market_data.get('alpha_20d_cum')} | Beta: {market_data.get('beta')}
        - Model 5-Günlük Medyan Beklenti: %{forecast_data.get('median_5d_return_pct')} (Yükseliş Olasılığı: %{forecast_data.get('up_probability')})
        - Temel Değerleme: {fundamental_data.get('valuation_status')} (F/K: {fundamental_data.get('trailing_pe')})
        - Haber Duygusu: {sentiment_data.get('overall_label')} (Skor: {sentiment_data.get('overall_sentiment_score')})

        İSTENEN JSON ŞEMASI:
        {{
            "executive_summary": "1-2 cümlelik net piyasa duruşu ve ana özet",
            "technical_regime": "200 SMA mesafesi, trend gücü ve momentum değerlendirmesi",
            "fundamental_valuation": "Değerleme çarpanları ve bilanço/likidite durumu",
            "sentiment_and_catalysts": "Haber akışı ve piyasa iştahı",
            "suggested_action": "Net taktiksel başlık (örn: 'Trend Desteğinde Kademeli Birikim')",
            "strategic_actions": [
                {{"title": "Taktiksel Görünüm", "detail": "Trend ve pozisyon alma yaklaşımı"}},
                {{"title": "Kilit Seviyeler & Katalizör", "detail": "Takip edilecek destek/direnç veya olaylar"}},
                {{"title": "Risk Yönetimi", "detail": "Zarar kes ve risk koruma adımları"}}
            ]
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
    """Gemini API yokken dönen deterministik, temiz, listelemeli profesyonel stratejist raporu."""
    is_gc = market.get("is_golden_cross", False)
    above_200 = market.get("is_above_sma200", False)
    dist_200 = market.get("dist_sma200_pct", 0.0)
    med_ret = forecast.get("median_5d_return_pct", 0.0)
    up_prob = forecast.get("up_probability", 50.0)
    val_status = fund.get("valuation_status", "MAKUL")
    is_crypto = ticker.endswith("-USD") or not fund.get("is_equity", True)

    # 1. Yönetici Özeti & Aksiyon Başlığı
    if above_200 and med_ret > 0.5:
        summary = f"{ticker}, 200 günlük hareketli ortalamasının üzerinde pozitif momentumla işlem görüyor. Kısa vadeli model projeksiyonu yukarı potansiyeli destekliyor."
        action_title = "Trend Yönünde Kademeli Takip"
        action_1 = "200 SMA üzerindeki ana yükseliş trendi korunuyor; geri çekilmeler alım yönlü takip edilebilir."
        action_2 = f"200 günlük ortalama (%{dist_200:+.1f} mesafede) birincil dinamik destek bölgesi."
        action_3 = "Oynaklık artışında veya kısa vadeli trend desteği kırılımlarında disiplinli kâr koruma uygulanmalı."
    elif not above_200 and med_ret < 0:
        summary = f"{ticker}, uzun vadeli ortalamalarının altında defansif bölgede bulunuyor. Sermaye koruması ve tepki alımlarında teyit beklenmesi önceliklidir."
        action_title = "Nakit Ağırlıklı Bekle-Gör"
        action_1 = "Zayıf teknik görünüm nedeniyle yeni pozisyon açılışlarında acele edilmemeli, taban oluşumu izlenmeli."
        action_2 = "200 SMA direnç konumunda; direnç aşılmadıkça yükselişler tepki hareketi olarak değerlendirilmeli."
        action_3 = "Mevcut pozisyonlarda zarar kes seviyeleri sıkı tutulmalı ve nakit tamponu korunmalı."
    else:
        summary = f"{ticker}, nötr bantta konsolide oluyor. Yön tayini için kırılım ve hacim teyidi izleniyor."
        action_title = "Piyasa Teyidi & Konsolidasyon Takibi"
        action_1 = "Yatay fiyatlama aralığında bant içi hareketler izlenmeli, net kırılım yönü beklenmeli."
        action_2 = f"Teknik göstergeler dengeli; haber akışı ({sent.get('overall_label', 'NÖTR')}) takip edilmeli."
        action_3 = "Bant sınırlarında destek ve direnç seviyelerine göre kademeli pozisyon yönetimi yapılabilir."

    # 2. Temel Değerleme Metni (Crypto "None / Uygulanamaz" saçmalığı temizlendi)
    if is_crypto:
        fundamental_text = f"{ticker} kripto varlık sınıfında yer almaktadır; geleneksel şirket çarpanları (F/K, F/DD) yerine ağ aktivitesi, borsa rezervleri ve küresel likidite koşullarıyla fiyatlanmaktadır."
    else:
        pe = fund.get("trailing_pe")
        pe_str = f"F/K: {pe:.1f}" if isinstance(pe, (int, float)) else "F/K: N/A"
        fwd_pe = fund.get("forward_pe")
        fwd_pe_str = f"İleri F/K: {fwd_pe:.1f}" if isinstance(fwd_pe, (int, float)) else ""
        days_earn = fund.get("days_to_earnings")
        earn_str = f"Bilanço: ~{days_earn} gün" if isinstance(days_earn, int) and days_earn > 0 else "Bilanço dönemi normal"
        fundamental_text = f"Değerleme profili: {val_status}. {pe_str} {fwd_pe_str}. {earn_str}."

    technical_text = f"Fiyat 200 SMA'ya göre %{dist_200:+.1f} mesafede. Golden Cross: {'Aktif' if is_gc else 'Pasif'}. Beta: {market.get('beta', 1.0)}."

    return {
        "executive_summary": summary,
        "technical_regime": technical_text,
        "fundamental_valuation": fundamental_text,
        "sentiment_and_catalysts": f"Haber duygu konsensüsü {sent.get('overall_label', 'NÖTR')} (Skor: {sent.get('overall_sentiment_score', 0.0):+.2f}).",
        "suggested_action": action_title,
        "strategic_actions": [
            {"title": "Taktiksel Görünüm", "detail": action_1},
            {"title": "Kilit Seviyeler & Katalizör", "detail": action_2},
            {"title": "Risk Yönetimi", "detail": action_3},
        ]
    }
