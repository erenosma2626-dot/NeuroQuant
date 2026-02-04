import yfinance as yf
import pandas as pd
import numpy as np

def get_rich_market_data(ticker="NVDA", period="2y", interval="1d"):
    """
    Belirtilen hisse için OHLCV verisini çeker ve Teknik İndikatörleri (RSI, MACD) ekler.
    
    Args:
        ticker (str): Hisse kodu (örn: NVDA)
        period (str): Ne kadarlık veri çekileceği (örn: '2y', '5y')
        interval (str): Veri aralığı (örn: '1d')
        
    Returns:
        pd.DataFrame: İçinde Close, RSI, MACD sütunları olan temiz veri seti.
    """
    print(f"📡 Veri çekiliyor: {ticker} ({period})...")
    
    # 1. Ham Veriyi Çek
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    
    if df.empty:
        raise ValueError("Veri çekilemedi! İnternet bağlantısını veya Ticker'ı kontrol et.")

    # Gereksiz sütunları temizle (Dividends vb.)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # 2. TEKNİK İNDİKATÖRLERİ HESAPLA (Feature Engineering)
    
    # --- RSI (14) ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # --- MACD (12, 26, 9) ---
    # EMA (Exponential Moving Average) hesaplamaları
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # --- SMA (20 & 50) Trend Takibi ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # 3. Temizlik (İlk satırlarda NaN oluşur hesaplamadan dolayı, onları atalım)
    df.dropna(inplace=True)
    
    print(f"✅ Veri Hazır! Son Fiyat: {df['Close'].iloc[-1]:.2f}$ | RSI: {df['RSI'].iloc[-1]:.2f}")
    return df

# --- TEST BLOĞU (Sadece bu dosya çalıştırılırsa devreye girer) ---
if __name__ == "__main__":
    # Dosyayı test etmek için terminale 'python neuro_modules/market_data.py' yaz
    try:
        data = get_rich_market_data()
        print(data.tail()) # Son 5 satırı göster
    except Exception as e:
        print(f"Hata oluştu: {e}")