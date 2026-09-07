# 🧠 NeuroQuant: Kapsamlı Proje Keşif ve Mimari Analiz Raporu

**Tarih:** 7 Eylül 2026  
**Analiz Edilen Dizin:** `/Users/erenosma/Downloads/NeuroQuant`  
**Git Durumu:** `main` dalı (origin/main ile senkronize, son commit: `b0ae243`)  
**Canlı Uygulama:** [Streamlit Cloud](https://neuroquant-s37f6vuhtauzpcqhf3kgfh.streamlit.app)

---

## 1. 🤖 Masaüstünden Çekilen ve Entegre Edilen Ajan Becerileri (Agent Skills)

Masaüstünüzdeki `~/Desktop/agent-skills-catalog` taranarak projeye ve genel ortama uygun **15 kritik ajan yeteneği** projeye aktarıldı.

### 📍 Kurulum Konumları:
- **Proje Özelinde:** [`.agents/skills/`](file:///Users/erenosma/Downloads/NeuroQuant/.agents/skills)
- **Global Ortam:** `~/.gemini/config/skills/` (Tüm eksikler senkronize edildi)

### 📦 Aktarılan Yetenek Kataloğu:

| Kategori | Beceri (Skill) | NeuroQuant'taki Kullanım Alanı |
| :--- | :--- | :--- |
| **01. Zaman Serisi & Kestirim** | `timesfm-forecasting` | Google Foundation Model ile hisse/kripto çok değişkenli zaman serisi tahminlemesi. |
| | `aeon` | Finansal serilerde anomali tespiti (flash crash, rejim değişimi) ve sınıflandırma. |
| | `scikit-survival` | Pozisyon taşıma süresi / stop-loss sağkalım modelleri (RUL / censoring-safe). |
| **02. Matematik & İstatistik** | `statsmodels` | ADF durağanlık testleri, ACF/PACF otokorelasyon, ARIMA/GARCH oynaklık modelleri. |
| | `pymc` | Bayesyen olasılıksal modelleme ve belirsizlik aralığı (Uncertainty intervals). |
| | `sympy` | Analitik türev ve kapalı form finansal matematik denklemleri. |
| | `pymoo` | Çok amaçlı portföy optimizasyonu (Risk vs. Getiri Pareto sınırı). |
| | `uncertainty-and-units` | Gürültü modellemesi ve hata yayılımı. |
| **03. Veri Bilimi & ML** | `scikit-learn` | Sızıntısız (leakage-safe) `TimeSeriesSplit` ve `Pipeline` mimarisi. |
| | `shap` | Modelin verdiği AL/SAT kararında hangi indikatörün baskın olduğunu açıklama (XAI). |
| | `polars` | Yüksek hızlı tick/günlük veri işleme motoru. |
| | `exploratory-data-analysis` | Veri seti kalite ve sızıntı denetimi. |
| | `scientific-visualization` | Yayın kalitesinde finansal ve istatistiksel grafikler. |
| **06. Frontend & Tasarım** | `frontend-design` | Üst düzey estetik, tipografi, modern renk paletleri ve dark mode tasarımı. |
| | `ui-ux-pro-max` | Terminal / Web / Streamlit için interaktif deneyim ve anti-slop ilkeleri. |

> [!NOTE]
> `~/Desktop/agent-brain` hafıza kasanız (Obsidian Vault / Karpathy LLM Wiki) incelendi. Projeyle ilgili alınacak mimari kararlar ve kazanımlar `index.md` ve `log.md` üzerinden kalıcı hafızaya aktarılmaya hazırdır.

---

## 2. 🏛️ Proje Mimarisi ve Dosya Yapısı

NeuroQuant, finansal piyasalardaki duygusal kararları minimize etmek amacıyla geliştirilmiş, **"No-Cheating" (Time-Wall)** felsefesine sahip hibrit (Teknik İndikatörler + LSTM/ML + FinBERT Duygu Analizi + Gemini Flash Yorumu) bir karar destek sistemidir.

```
NeuroQuant/
├── app.py                      # Ana Streamlit uygulama girişi ve UI orkestrasyonu
├── neuro_modules/              # Çekirdek iş mantığı ve motorlar
│   ├── ai_engine.py            # Model yükleme, LSTM projeksiyon, FinBERT ve Gemini entegrasyonu
│   ├── market_data.py          # yfinance veri çekme, RSI, MACD, SMA, Bollinger hesaplamaları
│   ├── news_scraper.py         # Google News RSS çekici, tarih sıralaması ve filtreleme
│   └── ui.py                   # Plotly grafikleri, özel HTML/CSS bileşenleri, karar paneli
├── models/                     # Eğitilmiş model ağırlıkları ve scaler dosyaları
│   ├── universal_lstm.h5       # Universal LSTM modeli (5 günlük yüzde tahmini)
│   ├── universal_scaler.pkl    # LSTM girdi/çıktı MinMaxScaler
│   ├── universal_rf.pkl        # Random Forest Regressor modeli (14 MB)
│   ├── neuroquant_lstm.h5      # Eski versiyon tekil hisse modeli
│   └── scaler.pkl              # Eski scaler
├── training/
│   └── train_universal.py      # 90 günlük Time-Wall ile eğitilen Random Forest betiği
├── test_universal.py           # Görmediği son 6 ay üzerinde yön doğruluğu & backtest simülasyonu
├── notebooks/ (.ipynb)
│   ├── 01_Veri_Toplama.ipynb   # Veri çekme ve ön hazırlık süreci
│   ├── 02_LSTM_Model_Egitimi.ipynb # LSTM mimarisi, training & test döngüsü
│   ├── 03_Strateji_ve_Backtest.ipynb # Basit strateji testleri
│   └── 04_Sentiment_Analizi.ipynb # FinBERT pipeline testleri
├── stocks_hazir_veri.csv       # 1003 satırlık indikatörlü hazır veri seti
└── requirements.txt            # Python bağımlılık listesi
```

---

## 3. 🔍 Katman Katman İnceleme ve Çalışma Mantığı

### A. Veri Katmanı ([`market_data.py`](file:///Users/erenosma/Downloads/NeuroQuant/neuro_modules/market_data.py) & [`news_scraper.py`](file:///Users/erenosma/Downloads/NeuroQuant/neuro_modules/news_scraper.py))
- **Piyasa Verisi:** `yfinance` üzerinden OHLCV çekilir.
  - RSI(14), MACD(12, 26, 9), EMA, SMA(20), Bollinger Bantları (Upper/Lower) hesaplanır.
  - `dropna()` ile indikatör başlangıç NaN'ları temizlenir.
- **Haber Akışı:** Google News RSS üzerinden son haberler çekilir.
  - Tarihe göre yeniden eskiye (`dt_obj`) kesin sıralama uygulanır.

### B. Zeka & Model Katmanı ([`ai_engine.py`](file:///Users/erenosma/Downloads/NeuroQuant/neuro_modules/ai_engine.py))
- **LSTM Tahmini:** Son 60 günlük `% değişim` (pct_change) dizisini alır, `universal_scaler` ile ölçekler, 5 günlük projeksiyon üretir.
- **Volatilite Emniyeti:** Tahmin edilen günlük değişimler ±%10 aralığında kırpılır (clipping).
- **FinBERT Sentiment & Veto:** `yiyanghkust/finbert-tone` modeli her haber başlığını skorlar (`Positive`, `Negative`, `Neutral`). Eğer en riskli haber skoru < -0.20 ise ve ortalama duygu negatifse **teknik sinyal veto edilir**.
- **Gemini Flash Entegrasyonu:** `gemini-3-flash-preview` modeli ile hem teknik göstergeler (RSI, MACD, Fiyat) hem de son 3 haber başlığı birleştirilerek finansal stratejist perspektifinden 3-4 cümlelik Türkçe sentez üretilir.

### C. Arayüz Katmanı ([`app.py`](file:///Users/erenosma/Downloads/NeuroQuant/app.py) & [`ui.py`](file:///Users/erenosma/Downloads/NeuroQuant/neuro_modules/ui.py))
- **Session State:** Hisse değiştirilmediği sürece sayfa yenilenmelerinde durum korunur (`analiz_aktif`).
- **Sekmeli Gösterim (Tabs):**
  1. *🚀 Ana Özet:* Karar göstergesi (GÜÇLÜ AL, AL, İZLE, SAT, RİSKLİ), 14 günlük geçmiş + 5 günlük AI projeksiyon Plotly grafiği.
  2. *📊 Teknik Detaylar:* Hacim grafiği, RSI momentum grafiği, Bollinger Bantları & MACD.
  3. *📰 Haber Masası:* FinBERT duygu ikonlu (🐂 Boğa, 🐻 Ayı, 😐 Nötr) ve skorlu editoryal kartlar.
- **Algoritmik Sinyal Özeti:** RSI, MACD ve Bollinger için eşik bazlı kural tablosu.
- **Dışa Aktarma:** Analiz edilen verilerin CSV olarak indirilmesi.

---

## 4. ⚠️ Tespit Edilen Kritik Bulgular, Uyumsuzluklar ve Teknik Borçlar

1. **Eksik Python Kütüphanesi (`google-generativeai`):**
   - `requirements.txt` içinde `google-generativeai==0.8.3` listelenmesine rağmen yerel sanal ortamda (`.venv`) bu paket kurulu değil (`ModuleNotFoundError`). Streamlit arayüzünde Gemini butonu tıklandığında hata fırlatır.
2. **`app.py` 1. Satırındaki Hatalı Import:**
   - `from xml.parsers.expat import model` satırı kaza eseri eklenmiş gereksiz bir satırdır.
3. **Model Mimari Çelişkisi (LSTM vs. Random Forest):**
   - Streamlit arayüzü (`app.py` & `ai_engine.py`) `universal_lstm.h5` modelini kullanıyor.
   - Buna karşılık `training/train_universal.py` ve `test_universal.py` betikleri `universal_rf.pkl` (Random Forest) modelini eğitiyor ve test ediyor. LSTM'in güncel yeniden eğitim kodu bir Python scripti olarak değil, yalnızca `02_LSTM_Model_Egitimi.ipynb` defterinde bulunuyor.
4. **Tahmin Kırpma (Hardcoded Clipping):**
   - `predict_future` fonksiyonundaki `if pct > 0.10: pct = 0.10` emniyet mekanizması, kripto varlıklarda (örn: BTC) doğal volatiliteyi baskılayabilir veya yapay yataylaşmaya neden olabilir.
5. **Backtest ve Kantitatif Metrik Eksikliği:**
   - Projede yön doğruluğu (Directional Accuracy) hesaplansa da finansal açıdan kritik olan **Sharpe Oranı**, **Sortino Oranı**, **Maksimum Çekilme (Max Drawdown)** ve **Komisyon/Kayma (Slippage) maliyetleri** simülasyona dahil edilmemiş.

---

## 5. 🚀 Sıradaki Adım İçin Hazırlık Durumu

- Tüm proje bileşenleri, veri akışı ve modeller eksiksiz olarak haritalandı.
- Masaüstünüzdeki gerekli tüm ajan yetenekleri projeye entegre edildi.
- Sistem, ileteceğiniz yeni yol haritasını, geliştirmeleri ve mimari revizyonları uygulamaya tamamen hazırdır!
