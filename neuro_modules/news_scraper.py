import feedparser
import urllib.parse
import requests
from datetime import datetime
import pandas as pd

def get_google_news(ticker_symbol="NVDA", max_results=10):
    """
    Google News RSS servisini kullanarak, belirtilen hisse hakkındaki
    son haberleri çeker ve YENİDEN ESKİYE sıralar.
    """
    
    # 1. URL OLUŞTURMA
    query = urllib.parse.quote(f"{ticker_symbol} stock news")
    # 'when:7d' parametresi ile son 7 güne odaklanabiliriz ama şimdilik genel kalsın
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    print(f"📡 Haberler çekiliyor: {ticker_symbol}...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(rss_url, headers=headers, timeout=10)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except Exception as e:
        print(f"⚠️ Bağlantı Hatası: {e}")
        return []
    
    if not feed.entries:
        return []
    
    news_list = []
    
    for entry in feed.entries:
        # Tarih formatlama ve Sıralama için Ham Tarihi Alma
        dt_obj = datetime.now() # Varsayılan (Eğer tarih yoksa)
        date_str = "Tarih Yok"
        
        if hasattr(entry, 'published_parsed'):
            # feedparser tarihi (Yıl, Ay, Gün, Saat...) tuple olarak verir
            dt_obj = datetime(*entry.published_parsed[:6])
            date_str = dt_obj.strftime('%Y-%m-%d %H:%M')

        news_item = {
            'title': entry.title,
            'link': entry.link,
            'published': date_str,
            'source': entry.source.title if hasattr(entry, 'source') else 'Unknown',
            'dt_obj': dt_obj # Sıralama için geçici olarak ekliyoruz (Gizli Kahraman)
        }
        news_list.append(news_item)
        
    # --- KRİTİK DOKUNUŞ: SIRALAMA ---
    # Listeyi 'dt_obj' anahtarına göre TERS (Yeniden Eskiye) sırala
    news_list.sort(key=lambda x: x['dt_obj'], reverse=True)
    
    # Şimdi sadece ilk 'max_results' kadarını al (En yeniler)
    final_list = news_list[:max_results]
    
    print(f"✅ Toplam {len(final_list)} haber çekildi ve sıralandı.")
    return final_list

if __name__ == "__main__":
    try:
        results = get_google_news("NVDA", max_results=10)
        if results:
            df = pd.DataFrame(results)
            # dt_obj sütununu ekranda göstermeye gerek yok
            print(df[['published', 'source', 'title']])
        else:
            print("Liste boş.")
    except Exception as e:
        print(f"Hata: {e}")