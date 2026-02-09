# 🧠 NeuroQuant: AI-Powered Algorithmic Trading Bot

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas)
![Status](https://img.shields.io/badge/Status-Phase_1_Completed-green?style=for-the-badge)

**NeuroQuant**, finansal piyasalardaki karmaşık veri örüntülerini çözmek için tasarlanmış uçtan uca (end-to-end) bir Yapay Zeka projesidir. Geleneksel teknik analizi, **Derin Öğrenme (LSTM)** ve **Doğal Dil İşleme (NLP/Sentiment Analysis)** ile birleştirerek hibrit bir karar mekanizması oluşturmayı hedefler.


---

## 🛠️ Kullanılan Teknolojiler ve Araçlar

Bu projede modern veri bilimi ve yapay zeka kütüphaneleri kullanılmıştır:

* **Veri Toplama:** `yfinance` (Yahoo Finance API)
* **Veri İşleme & Manipülasyon:** `pandas`, `numpy`
* **Görselleştirme:** `matplotlib`
* **Derin Öğrenme (Deep Learning):** `tensorflow`, `keras` (LSTM Katmanları)
* **Veri Ölçeklendirme:** `scikit-learn` (MinMaxScaler)
* **Geliştirme Ortamı:** Google Colab / Jupyter Notebook

---

## ⚙️ Metodoloji

Proje, **CRISP-DM** (Cross-Industry Standard Process for Data Mining) döngüsüne sadık kalınarak şu adımlarla geliştirilmiştir:

### 1. Veri Madenciliği (Data Mining)
Geçmişe dönük 5+ yıllık hisse senedi verileri (Açılış, Yüksek, Düşük, Kapanış, Hacim) çekildi. Veri setinin tutarlılığı kontrol edildi ve eksik veriler temizlendi.

### 2. Öznitelik Mühendisliği (Feature Engineering)
Modelin sadece fiyata bakarak değil, piyasa dinamiklerini anlayarak öğrenmesi için veriye Teknik İndikatörler eklendi:
* **RSI (Relative Strength Index):** Aşırı alım/satım bölgelerini tespiti.
* **MACD:** Trend dönüşümlerinin tespiti.
* **Bollinger Bands:** Volatilite ölçümü.

### 3. Ön İşleme (Preprocessing)
* **Scaling:** LSTM modellerinin performansı için veriler `MinMaxScaler` ile 0-1 aralığına sıkıştırıldı.
* **Windowing (Pencereleme):** Zaman serisi verisi, son 60 günü (Lookback) girdi olarak alıp, bir sonraki günü tahmin edecek şekilde (X, y) matrislerine dönüştürüldü.

### 4. Model Mimarisi (LSTM)
Zaman serilerindeki uzun vadeli bağımlılıkları öğrenmesi için **LSTM (Long Short-Term Memory)** mimarisi seçildi.
* **LSTM Layers:** Geçmiş verideki patternleri ezberlemek yerine öğrenmek için.
* **Dropout (%20):** Overfitting (Aşırı öğrenme) riskini önlemek için rastgele nöron kapatma.
* **Dense Layer:** Sonuç çıktısını tek bir fiyat tahminine indirgemek için.

### 5. Strateji ve Backtest
Modelin ham fiyat tahminleri, bir **"Threshold (Eşik)"** algoritmasından geçirildi. Sadece belirli bir güven aralığını (%0.5 - %1.0) aşan değişimlerde **AL/SAT** sinyali üretilmesi sağlanarak gürültü (noise) engellendi.

---

## 📅 Proje Yol Haritası

Şu anda projenin **2. Adımdayız** ve temel teknik analiz motorunu tamamladık.

* ✅ **Adım 1:** Veri Madenciliği, Temizlik ve Görselleştirme.
* ✅ **Adım 2:** LSTM Modellemesi, Eğitim ve Strateji Backtest'i.
* 🚧 **Adım 3 (Sırada):** Sentiment Analizi (Finansal Haberlerin NLP ile işlenmesi).
* ⏳ **Adım 4:** Entegrasyon, Otomasyon ve Final Canlı Test.

---

## 📊 Sonuçlar (Şimdilik)

**LSTM Modeli Performansı (Notebook 3):**
* **Market Getirisi (Buy & Hold):** x1.68
* **NeuroQuant Stratejisi:** x1.52 (Daha düşük risk profili ile)
* **Not:** Model, özellikle düşüş trendlerinde nakite geçerek portföyü koruma ("Stop-Loss" etkisi) konusunda başarılı olmuştur.

---

## 🚀 Kurulum ve Çalıştırma

1.  Repoyu klonlayın.
2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install pandas numpy matplotlib tensorflow scikit-learn yfinance
    ```
3.  Notebookları sırasıyla çalıştırın (`01` -> `02` -> `03`).
