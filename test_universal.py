import os
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- AYARLAR ---
TEST_TICKERS = ['NVDA', 'BTC-USD'] # Hem hisse hem kripto ile test edelim
LOOKBACK = 60
MODEL_PATH = 'models/universal_rf.pkl'

def test_model():
    if not os.path.exists(MODEL_PATH):
        print("🚨 HATA: Model dosyası bulunamadı!")
        return

    print("🧠 Model yükleniyor...")
    model = joblib.load(MODEL_PATH)
    
    for ticker in TEST_TICKERS:
        print(f"\n🔎 {ticker} İÇİN TEST BAŞLIYOR...")
        
        # Son 6 ayın verisini çekelim (Test için taze veri)
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(ticker, axis=1, level=1)
            except: pass
        df = df[[c for c in cols if c in df.columns]]
        
        # Veriyi Hazırla (% Değişim)
        prices = df['Close'].values
        pct_changes = df['Close'].pct_change().fillna(0).values
        
        predictions = []
        actuals = []
        dates = []
        
        # Simülasyon Döngüsü
        # Geçmiş 60 günü alıp, bir sonraki günü tahmin ettireceğiz
        print("   ⏳ Simülasyon çalışıyor...")
        for i in range(LOOKBACK, len(pct_changes) - 5):
            # Modelin Girdisi: Geçmiş 60 gün
            input_feat = pct_changes[i-LOOKBACK:i].reshape(1, -1)
            
            # Model Tahmini (5 günlük vektör veriyor, biz ilk güne bakalım)
            pred_vector = model.predict(input_feat)[0]
            pred_day_1 = pred_vector[0] # Yarınki değişim tahmini
            
            # Gerçekleşen (Yarınki gerçek değişim)
            actual_day_1 = pct_changes[i]
            
            predictions.append(pred_day_1)
            actuals.append(actual_day_1)
            dates.append(df.index[i])
            
        # --- SONUÇLARI HESAPLA ---
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # 1. YÖN DOĞRULUĞU (Directional Accuracy)
        # Eğer ikisi de pozitifse veya ikisi de negatifse bildi demektir.
        correct_direction = np.sign(predictions) == np.sign(actuals)
        win_rate = np.mean(correct_direction) * 100
        
        # 2. KAR TABLOSU (Kümülatif Getiri)
        # Model "Al" (Pozitif) dediyse o günkü gerçek değişimi kazanırız.
        strategy_returns = np.cumsum(np.where(predictions > 0, actuals, 0))
        buy_hold_returns = np.cumsum(actuals)
        
        print(f"   🎯 Yön Bilme Oranı: %{win_rate:.2f}")
        
        # Grafiği Çiz
        plt.figure(figsize=(10, 5))
        plt.plot(dates, strategy_returns, label='AI Stratejisi (Model)', color='green')
        plt.plot(dates, buy_hold_returns, label='Al-Tut (Piyasa)', color='gray', linestyle='--')
        plt.title(f"{ticker} - Yapay Zeka vs. Piyasa")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    test_model()