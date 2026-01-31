🧠 NeuroQuant: AI-Powered Algorithmic Trading Bot
NeuroQuant, finansal piyasalardaki karmaşık veri örüntülerini çözmek için tasarlanmış uçtan uca (end-to-end) bir Yapay Zeka projesidir. Geleneksel teknik analizi, Derin Öğrenme (LSTM) ve Doğal Dil İşleme (NLP/Sentiment Analysis) ile birleştirerek hibrit bir karar mekanizması oluşturmayı hedefler.


🛠️ Kullanılan Teknolojiler ve Araçlar
Bu projede modern veri bilimi ve yapay zeka kütüphaneleri kullanılmıştır:

Veri Toplama: yfinance (Yahoo Finance API)

Veri İşleme & Manipülasyon: pandas, numpy

Görselleştirme: matplotlib

Derin Öğrenme (Deep Learning): tensorflow, keras (LSTM Katmanları)

Veri Ölçeklendirme: scikit-learn (MinMaxScaler)

Geliştirme Ortamı: Google Colab / Jupyter Notebook



📅 Proje Yol Haritası 
Şu anda projenin 2. Günündeyiz ve temel teknik analiz motorunu tamamladık.

✅ Veri Madenciliği ve Analiz (Tamamlandı)

Yahoo Finance API üzerinden geçmiş hisse verilerinin (OHLCV) çekilmesi.

Teknik İndikatörlerin (RSI, MACD, Bollinger Bands) hesaplanması ve veriye işlenmesi.

Veri görselleştirme ve korelasyon analizleri.

✅ Yapay Zeka Motoru & LSTM (Tamamlandı)

Zaman serisi tahmini için LSTM (Long Short-Term Memory) ağlarının kurulması.

MinMaxScaler ve Windowing (Pencereleme) yöntemleriyle veri ön işleme.

Backtesting: Modelin geçmiş veriler üzerinde al/sat stratejilerinin simülasyonu.

Threshold Stratejisi: Gürültüyü (Noise) filtreleyen akıllı algoritma ile %50'ye varan simülasyon getirisi.

🚧 Sentiment Analizi (NLP) (Sırada...)

Finansal haberlerin ve başlıkların taranması.

BERT / FinBERT modelleri ile haberlerin "Duygu Skorunun" (Pozitif/Negatif) çıkarılması.

"Rakamlar Düşüşte ama Haberler İyi" gibi çelişkili durumlarda karar verme yeteneği.

⏳ Otomasyon ve Canlı Bot

Teknik (LSTM) ve Sözel (NLP) beyinlerin birleştirilmesi.

Canlı veriye bağlanıp anlık sinyal (Al/Sat/Tut) üreten sistemin inşası.

📂 Dosya Yapısı
01_Veri_Hazirligi.ipynb: Ham verinin çekilmesi ve indikatörlerin eklenmesi.

02_Model_Egitimi_LSTM.ipynb: Derin öğrenme modelinin eğitilmesi ve validasyonu.

03_Strateji_Backtest.ipynb: Eğitilen modelin stratejiye dönüştürülmesi ve kârlılık testi.

models/: Eğitilmiş .h5 model dosyaları ve scaler.pkl dosyaları.

data/: İşlenmiş ve hazır veri setleri (.csv).

📊 Sonuçlar (Şimdilik)
LSTM Modeli Performansı (Notebook 3):

Market Getirisi (Buy & Hold): x1.68

NeuroQuant Stratejisi: x1.52 (Daha az risk, düşüşlerde nakite geçiş özelliği ile)

Not: Model, özellikle "Ayı Piyasası" (Düşüş) dönemlerinde portföyü koruma konusunda başarılı performans sergilemiştir.


⚙️ Metodoloji
Proje, CRISP-DM (Cross-Industry Standard Process for Data Mining) döngüsüne sadık kalınarak şu adımlarla geliştirilmiştir:

1. Veri Madenciliği (Data Mining)

Geçmişe dönük 5+ yıllık hisse senedi verileri (Açılış, Yüksek, Düşük, Kapanış, Hacim) çekildi. Veri setinin tutarlılığı kontrol edildi ve eksik veriler temizlendi.

2. Öznitelik Mühendisliği (Feature Engineering)

Modelin sadece fiyata bakarak değil, piyasa dinamiklerini anlayarak öğrenmesi için veriye Teknik İndikatörler eklendi:

RSI (Relative Strength Index): Aşırı alım/satım bölgelerini tespiti.

MACD: Trend dönüşümlerinin tespiti.

Bollinger Bands: Volatilite ölçümü.

3. Ön İşleme (Preprocessing)

Scaling: LSTM modellerinin performansı için veriler MinMaxScaler ile 0-1 aralığına sıkıştırıldı.

Windowing (Pencereleme): Zaman serisi verisi, son 60 günü (Lookback) girdi olarak alıp, bir sonraki günü tahmin edecek şekilde (X, y) matrislerine dönüştürüldü.

4. Model Mimarisi (LSTM)

Zaman serilerindeki uzun vadeli bağımlılıkları öğrenmesi için LSTM (Long Short-Term Memory) mimarisi seçildi.

LSTM Layers: Geçmiş verideki patternleri ezberlemek yerine öğrenmek için.

Dropout (%20): Overfitting (Aşırı öğrenme) riskini önlemek için rastgele nöron kapatma.

Dense Layer: Sonuç çıktısını tek bir fiyat tahminine indirgemek için.

5. Strateji ve Backtest

Modelin ham fiyat tahminleri, bir "Threshold (Eşik)" algoritmasından geçirildi. Sadece belirli bir güven aralığını (%0.5 - %1.0) aşan değişimlerde AL/SAT sinyali üretilmesi sağlanarak gürültü (noise) engellendi.
