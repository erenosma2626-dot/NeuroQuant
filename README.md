# 🧠 NeuroQuant 3.0 | The Sovereign Quant Platform

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20Async-009688)
![Frontend](https://img.shields.io/badge/Frontend-Vite%20React%2019-61DAFB)
![LightGBM](https://img.shields.io/badge/ML-LightGBM%20Quantile-yellow)
![TimesFM](https://img.shields.io/badge/Foundation-Google%20TimesFM%203.0-orange)
![TradingView](https://img.shields.io/badge/Charts-Lightweight%20Charts%20v5-blueviolet)
![Status](https://img.shields.io/badge/Release-v3.0%20Institutional-success)

**NeuroQuant 3.0**, bireysel ve kurumsal yatırımcılar için geliştirilmiş, **ayrık mimarili (Decoupled)**, çok faktörlü (Multi-Factor) kantitatif zekâ ve simülasyon platformudur.

Kullanıcıyı yanıltan tekil nokta tahminleri (Point Forecasts) ve aşırı alım-satım üreten geleneksel indikatörler (RSI vb.) yerine; **olasılıksal güven konileri ($q_{10}, q_{50}, q_{90}$)**, **temel değerleme çarpanları (F/K, PEG, F/DD)**, **bilanço sürprizleri**, **50 & 200 SMA trend mesafeleri**, **sektörel alfa ($R_{asset} - R_{bench}$)** ve **10.000$ dinamik sermaye tahsis simülasyonu** sunar.

---

## ⚡ Temel Özellikler

### 1. 10.000$ Dinamik Sermaye Simülasyon Laboratuvarı
* **Akıllı Pozisyon Büyüklüğü:** Model her gün $10.000$ sermaye evreninde o an ne kadar nakit ve hisse taşıyacağına bileşik güven skoruna göre karar verir ($0k, 2.5k, 5k, 7.5k, 10k$).
* **Anti-Churning & Histerezis Filtresi:** Küçük gürültülerde portföyü sürekli al-sat komisyonuna boğmamak için en az %25 ağırlık farkı ($|w_t^* - w_{t-1}| \ge 0.25$) ve 3 günlük asgari tutma süresi uygulanır.
* **Friction & Kayma:** Her işlemde %0.10 kurumsal komisyon ve spread maliyeti hesaba katılır.
* **Canlı Çizgi Yarışı (Equity Race):** Modelin getiri eğrisi ile 10k sabit Al-Tut (Buy & Hold) karşılaştırması saniyelik kare hızında akıcı olarak izlenebilir.
* **Açıklanabilir Yapay Zeka (XAI) Modalı:** Grafikteki veya işlem kütüğündeki herhangi bir ALIM/SATIM noktasına tıklandığında, modelin o kararı almasındaki faktörler (Yapay Zeka tahmini, Temel Değerleme, Sektörel Alfa, Bilanço Takvimi, Para Akışı) gerekçeleriyle listelenir.

### 2. Olasılıksal Güven Konisi (Quantile Regression)
* **LightGBM Pinball Loss ($q_{10}, q_{50}, q_{90}$):** Gelecek 5 iş günü için piyasanın %80 olasılıkla hareket edeceği tavan ve taban bandı hesaplar.
* **Belirsizlik Yayılımı:** Volatilite patlamalarında bantlar genişleyerek risk iştahını otomatik kısar.

### 3. Temel Değerleme & Bilanço Radarı
* **Çarpan Matrisi:** Cari F/K, İleriye Dönük (Forward) F/K, PEG Oranı ve F/DD ile hissenin pahalı mı yoksa adil değerde mi olduğunu puanlar.
* **Bilanço Koruma Kalkanı:** Kazanç açıklamasına 7 günden az kalan dönemlerde risk primi artırılarak sermaye nakde çekilir (De-risking).
* **Tarihsel Bilanço Sürprizleri:** Geçmiş EPS sürpriz oranları tablolanır.

### 4. Üstel Zaman Çürümeli Duygu Analizi (Sentiment Decay)
* 24 saatlik yarılanma ömrü ($e^{-\lambda \Delta t}$) ile 3 gün önceki haberlerin model kararlarını yapay olarak kirletmesi engellenir.

---

## 🏗️ Mimari Şema

```
NeuroQuant 3.0
├── backend/                  # FastAPI Asenkron Yüksek Performanslı API (Port 8000)
│   ├── routers/             # market, forecast, fundamentals, simulation, news, agent
│   ├── services/            # lightgbm, timesfm, simulation, market_service, gemini
│   └── config.py            # Ayarlar, TTL önbellekleme (15 dk), benchmark eşleşmeleri
├── frontend/                 # Vite + React 19 + TypeScript Terminal (Port 5173)
│   ├── src/components/      # TradingViewChart, SimulationLab, QuantMatrix, FundamentalRadar
│   └── src/index.css        # The Sovereign Quant (Obsidian dark, JetBrains Mono, Hairline)
├── models/clusters/         # Global küme modelleri (Tech, Crypto, BIST, Defensive)
└── run_dev.sh               # Tek komutla backend + frontend ayağa kaldırma betiği
```

---

## 🚀 Hızlı Başlangıç

### Gereksinimler
* Python 3.11+
* Node.js v18+ ve npm

### 1. Tek Komutla Başlatma
Terminalinizden tek bir komutla hem FastAPI backend hem de Vite frontend'i çalıştırabilirsiniz:

```bash
chmod +x run_dev.sh
./run_dev.sh
```

Açılan adresler:
* 🌐 **Web Terminali:** [http://127.0.0.1:5173](http://127.0.0.1:5173)
* ⚡ **API Dokümantasyonu (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Ayrı Ayrı Çalıştırma (Opsiyonel)

**Backend:**
```bash
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## 📊 10.000$ Simülasyon Test Sonuçları (6 Aylık Zaman Duvarı)

| Varlık | Strateji | Başlangıç | Bitiş Sermayesi | Net Getiri | Sharpe | Max Drawdown | İşlem Sayısı |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVDA** | NeuroQuant AI | $10.000 | **$13.370,96** | **+%33,71** | **3,16** | **%4,60** | 26 |
| NVDA | Sabit Al-Tut | $10.000 | $11.936,40 | +%19,36 | 1,48 | %19,10 | 0 |
| **THYAO.IS** | NeuroQuant AI | $10.000 | **$12.071,89** | **+%20,72** | **3,00** | **%1,91** | 22 |
| THYAO.IS | Sabit Al-Tut | $10.000 | $9.845,28 | -%1,55 | -0,08 | %17,66 | 0 |
| **BTC-USD** | NeuroQuant AI | $10.000 | **$10.521,60** | **+%5,22** | **1,85** | **%0,98** | 5 |
| BTC-USD | Sabit Al-Tut | $10.000 | $10.130,55 | +%1,31 | 0,22 | %28,71 | 0 |

> **Analiz:** Model düşüş trendlerinde nakde (%0 pozisyon) geçerek yatırımcıyı piyasanın sert çekilmelerinden korumuş; yükseliş trendlerinde ise kaldıraçsız şekilde k, 2k, 3k dilimleriyle pozisyon artırarak maksimum alfa üretmiştir.

---

## ⚠️ Yasal Uyarı
Bu proje, açık kaynak kodlu bir kantitatif araştırma ve finansal mühendislik çalışmasıdır. Burada sunulan veriler, modeller ve simülasyon sonuçları **kesinlikle yatırım tavsiyesi niteliğinde değildir.**
