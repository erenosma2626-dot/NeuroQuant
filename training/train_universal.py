import os
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- AYARLAR ---
TICKERS = ['NVDA', 'AAPL', 'MSFT', 'BTC-USD', 'SPY', 'TSLA', 'AMZN', 'GOOGL']
LOOKBACK = 60
PREDICT_DAYS = 5

# --- KRİTİK AYAR: ZAMAN DUVARI ---
# Model bugünden önceki son 90 günü ASLA görmeyecek.
# O verileri "Test" için saklayacağız.
TEST_DAYS = 90 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

def get_honest_data():
    print(f"📡 Dürüst Eğitim Başlıyor: Veriler çekiliyor...")
    all_X, all_y = [], []
    
    # Bitiş tarihini ayarla (Bugün - 90 gün)
    cutoff_date = datetime.now() - timedelta(days=TEST_DAYS)
    cutoff_str = cutoff_date.strftime('%Y-%m-%d')
    print(f"🛑 ZAMAN DUVARI: {cutoff_str} tarihinden sonrası EĞİTİME ALINMAYACAK.")

    for ticker in TICKERS:
        try:
            # Sadece Cutoff tarihine kadar olan veriyi indir
            # end=cutoff_str diyerek geleceği gizliyoruz
            df = yf.download(ticker, start="2022-01-01", end=cutoff_str, progress=False, threads=False)
            
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs(ticker, axis=1, level=1)
                except: pass
            
            df = df[[c for c in cols if c in df.columns]]
            
            # Veri yetersizse geç
            if len(df) < LOOKBACK + PREDICT_DAYS: continue

            # % DEĞİŞİM (Evrenselleştirme)
            df_pct = df.pct_change().dropna().replace([np.inf, -np.inf], 0)
            data = df_pct['Close'].values 
            
            # Eğitim Setini Oluştur
            for i in range(LOOKBACK, len(data) - PREDICT_DAYS):
                all_X.append(data[i-LOOKBACK:i])
                all_y.append(data[i:i+PREDICT_DAYS])
                
            print(f"   ✅ {ticker}: {len(data)} gün eklendi (Gelecek gizlendi).")
        except Exception as e:
            print(f"   ⚠️ Hata {ticker}: {e}")

    if not all_X: raise ValueError("Veri Yok!")
    return np.array(all_X), np.array(all_y)

def train():
    print("\n🌲 DÜRÜST RANDOM FOREST EĞİTİMİ...")
    
    # 1. Veriyi Al (Gelecekten arındırılmış)
    X, y = get_honest_data()
    print(f"📊 Toplam Eğitim Senaryosu: {X.shape[0]}")
    
    # 2. Modeli Eğit
    # n_estimators=200 yaptık, biraz daha güçlensin.
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42, max_depth=20)
    model.fit(X, y)
    
    # 3. Kaydet
    joblib.dump(model, os.path.join(MODEL_DIR, 'universal_rf.pkl'))
    print("✅ EĞİTİM BİTTİ (Ezbersiz Model Hazır)")
    print(f"📂 Kayıt: {os.path.join(MODEL_DIR, 'universal_rf.pkl')}")

if __name__ == "__main__":
    train()