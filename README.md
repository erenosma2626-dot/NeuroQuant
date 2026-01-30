# NeuroQuant: Duygu Analizi Destekli Borsa Tahmin Modeli
Bu proje, hisse senedi verilerini kullanarak teknik indikatörler ve 
haber duygu analizi (sentiment analysis) ile hibrit bir fiyat tahminleme sistemi kurmayı amaçlar.

## 🛠 Kullanılan Araçlar
* **Veri Kaynağı:** Yahoo Finance (yfinance)
* **Veri Analizi:** Pandas, NumPy
* **İstatistiksel Testler:** Statsmodels (ADF Testi)
* **Teknik Analiz:** Custom EMA, RSI, MACD ve Bollinger Band hesaplamaları
* **NLP (Doğal Dil İşleme):** TextBlob (Duygu Analizi)
* **Model (Gelecek):** LSTM (Deep Learning)

## 📈 Metodoloji

### 1. Veri Durağanlığı (Stationarity)
Zaman serisi verileri genellikle durağan değildir. Projede, serinin birim kök içerip içermediğini kontrol etmek için **Augmented Dickey-Fuller (ADF)** testi uygulanmıştır. Veri, 1. derece fark alma (differencing) yöntemiyle durağanlaştırılmıştır.

### 2. Teknik İndikatörler
* **Bollinger Bantları:** Fiyatın $\pm 2\sigma$ (standart sapma) aralığındaki hareketleri izlenerek istatistiksel anomaliler tespit edilir.
* **RSI & MACD:** Momentum ve trend değişimleri EMA (Üstel Hareketli Ortalama) tabanlı hesaplamalarla takip edilir.

### 3. Duygu Analizi (Sentiment Analysis)
Finansal haberler çekilerek metin üzerinden duygu skoru üretilir. Bu, modele sadece fiyat verisini değil, piyasa psikolojisini de girdi olarak vermemizi sağlar.
