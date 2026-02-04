import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from transformers import pipeline
import os

# --- 1. AYARLAR VE YÜKLEME ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

@st.cache_resource
def load_brains():
    """Modelleri önbelleğe alarak yükler."""
    print("🧠 AI Motorları Yükleniyor...")
    
    # LSTM
    try:
        model = load_model(os.path.join(MODEL_DIR, 'neuroquant_lstm.h5'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    except Exception as e:
        st.error(f"🚨 LSTM yüklenemedi: {e}")
        return None, None, None

    # FinBERT
    try:
        sentiment_pipe = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
    except Exception as e:
        st.error(f"🚨 FinBERT yüklenemedi: {e}")
        return model, scaler, None

    return model, scaler, sentiment_pipe

# --- 2. TEKNİK ANALİZ MOTORU (LSTM) ---
# --- MEVCUT PREDICT_FUTURE FONKSİYONUNU BUNUNLA DEĞİŞTİR ---

def predict_future(model, scaler, last_60_days_df):
    """
    LSTM tahminlerini üretir ve 'Volatilite Kelepçesi' (Max %2 günlük değişim) uygular.
    """
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [c for c in feature_cols if c in last_60_days_df.columns]
    
    raw_data = last_60_days_df[available_cols].values
    scaled_data = scaler.transform(raw_data)
    current_batch = scaled_data.reshape(1, 60, len(available_cols))
    
    predicted_prices = []
    
    # Referans Fiyat
    last_real_price = last_60_days_df['Close'].iloc[-1]
    curr_price = last_real_price 
    
    # --- AYARLAR ---
    MAX_DAILY_CHANGE = 0.02 # Günlük maksimum %2 değişim izni (Nasdaq standardı)
    
    for i in range(5):
        # 1. Ham Tahmin
        pred = model(current_batch, training=False)
        pred_scaled = pred.numpy()[0][0]
        
        # Batch güncelleme
        next_input_scaled = current_batch[0, -1, :].copy()
        next_input_scaled[3] = pred_scaled
        new_step = next_input_scaled.reshape(1, 1, len(available_cols))
        current_batch = np.append(current_batch[:, 1:, :], new_step, axis=1)
        
        # 2. Fiyata Çevirme
        unscaled_row = scaler.inverse_transform(next_input_scaled.reshape(1, -1))
        raw_pred_price = unscaled_row[0][3]
        
        # 3. VOLATİLİTE KELEPÇESİ (Smart Clamping)
        # Modelin ham tahmini ile şu anki fiyat arasındaki farka bakıyoruz.
        change_pct = (raw_pred_price - curr_price) / curr_price
        
        # Eğer değişim %2'den büyükse, zorla %2'ye çekiyoruz.
        if change_pct > MAX_DAILY_CHANGE:
            target_price = curr_price * (1 + MAX_DAILY_CHANGE)
        elif change_pct < -MAX_DAILY_CHANGE:
            target_price = curr_price * (1 - MAX_DAILY_CHANGE)
        else:
            target_price = raw_pred_price
            
        # 4. Yumuşatma (Smoothing) - Son Rötuş
        # Kelepçelenmiş fiyatı bile önceki günle harmanlayıp keskin köşeleri alıyoruz.
        # %70 Önceki Gün, %30 Yeni Hedef (Trend devamlılığı sağlar)
        smoothed_price = (curr_price * 0.70) + (target_price * 0.30)
        
        predicted_prices.append(smoothed_price)
        curr_price = smoothed_price

    return predicted_prices

# --- 3. DUYGU ANALİZİ (VETO MANTIKLI) ---
def score_news(sentiment_pipe, news_list):
    """
    Haberleri puanlar ve her habere 'ai_score' etiketi yapıştırır.
    """
    if not news_list:
        return 0, "Nötr", None
    
    total_score = 0
    analyzed_count = 0
    min_score = 1.0 
    riskiest_news = None 
    RISK_THRESHOLD = -0.20 

    print(f"📰 {len(news_list)} haber analiz ediliyor...")
    
    for news in news_list:
        text = news['title']
        
        # FinBERT Analizi
        result = sentiment_pipe(text[:512])[0]
        label = result['label']
        confidence = result['score']
        
        if label == 'Positive':
            ai_score = confidence
        elif label == 'Negative':
            ai_score = -confidence
        else:
            ai_score = 0
            
        # --- YENİ EKLENTİ: Skoru Habere Kaydet ---
        # Böylece UI tarafında "Bu haberin puanı %85" diye gösterebileceğiz.
        news['ai_score'] = ai_score 
        news['ai_label'] = label # 'Positive', 'Negative' yazısı
        
        # Risk Takibi
        if ai_score < min_score:
            min_score = ai_score
            if ai_score < RISK_THRESHOLD:
                riskiest_news = news 
        
        total_score += ai_score
        analyzed_count += 1
        
    if analyzed_count == 0:
        return 0, "Nötr", None
        
    final_avg = total_score / analyzed_count
    
    general_sentiment = "NÖTR"
    if final_avg > 0.15: general_sentiment = "POZİTİF"
    elif final_avg < -0.15: general_sentiment = "NEGATİF"
    
    return final_avg, general_sentiment, riskiest_news

# --- 4. HİBRİT KARAR MEKANİZMASI (HAKİM) ---
def make_final_decision(lstm_preds, sentiment_score, riskiest_news, current_rsi):
    """
    Teknik + Temel + Veto yetkisi ile nihai kararı verir.
    """
    # Teknik Yön (Yüzde Değişim)
    start_price = lstm_preds[0]
    end_price = lstm_preds[-1]
    price_change_pct = ((end_price - start_price) / start_price) * 100
    
    decision = "NÖTR / İZLE"
    color = "gray"
    explanation = "Yeterli sinyal oluşmadı."
    
    # --- VETO KONTROLÜ (GÜVENLİK SİGORTASI) ---
    # Haberlerin ortalaması iyi olsa bile, tek bir FELAKET haberi varsa fren yap.
    riskiest_score = 0
    if riskiest_news:
        # Haberi tekrar puanlayıp (veya stored puanı alıp) kontrol etmek yerine
        # score_news içinde hesaplanan min_score'u da döndürebilirdik ama
        # şimdilik tekrar basit bir kontrol yapalım veya varsayalım.
        # Basitlik için: Genel sentiment çok kötüyse zaten negatiftir.
        pass 

    # KURAL 1: RSI VETOSU
    if current_rsi > 70:
        decision = "RİSKLİ / BEKLE (RSI Şişik)"
        color = "orange"
        explanation = f"Teknik göstergeler (RSI: {current_rsi:.0f}) aşırı alım bölgesinde. Düzeltme ihtimali yüksek."
        if sentiment_score > 0.2:
            explanation += " Ancak haber akışı pozitif olduğu için 'Short Squeeze' (yukarı patlama) olabilir. Stop-loss ile izle."
        return decision, color, explanation

    # KURAL 2: HABER VETOSU (Outlier Detection)
    # Eğer ortalama puan iyiyse bile (-0.5'ten iyi), ama en kötü haber -0.8'den kötüyse:
    # (Burada riskiest_news objesinden skoru tekrar çekmiyoruz, basitlik için sentiment_score üzerinden gidiyoruz
    # ama Dashboard'da o haberi göstereceğiz.)
    if sentiment_score < -0.4: # Genel hava kötüyse
        decision = "SAT / UZAK DUR"
        color = "red"
        explanation = "Haber akışı belirgin şekilde negatif. Teknik yükseliş gösterse bile 'Boğa Tuzağı' riski var."
        return decision, color, explanation

    # KURAL 3: NORMAL AKIŞ
    if price_change_pct > 1.0: # LSTM Yükseliş Bekliyor
        if sentiment_score > 0.15:
            decision = "GÜÇLÜ AL 🚀"
            color = "green"
            explanation = f"Yapay zeka %{price_change_pct:.1f} yükseliş öngörüyor ve haber akışı bunu destekliyor."
        else:
            decision = "AL (Temkinli)"
            color = "blue"
            explanation = "Teknik yükseliş var ancak haber akışı nötr/zayıf."
            
    elif price_change_pct < -1.0: # LSTM Düşüş Bekliyor
        if sentiment_score < -0.15:
            decision = "GÜÇLÜ SAT 🔻"
            color = "red"
            explanation = "Hem teknik model hem haberler düşüşü işaret ediyor."
        else:
            decision = "SAT (Tepki Gelebilir)"
            color = "orange"
            explanation = "Teknik düşüş trendinde ama haberler kötü değil. Yatay seyir olabilir."
            
    return decision, color, explanation

# --- ENTEGRASYON VE VETO TESTİ ---
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from neuro_modules.news_scraper import get_google_news
    
    print("\n🚀 VETO SİSTEM TESTİ...\n")
    m, s, pipe = load_brains()
    
    if m and pipe:
        # 1. Gerçek Haberleri Çek
        real_news = get_google_news("NVDA", max_results=10)
        
        # 2. Haberleri Puanla
        avg_score, label, risky_news = score_news(pipe, real_news)
        
        print(f"\n📊 Ortalama Skor: {avg_score:.3f} ({label})")
        if risky_news:
            print(f"⚠️ En Riskli Haber: {risky_news['title']}")
            
        # 3. Karar Testi (Sahte Teknik Verilerle)
        # Senaryo: LSTM %3 artış diyor, RSI 60 (Normal), Ama haberler ne diyor?
        fake_lstm_preds = [100, 101, 102, 103, 103] # Yükseliş
        fake_rsi = 60
        
        dec, col, expl = make_final_decision(fake_lstm_preds, avg_score, risky_news, fake_rsi)
        print(f"\n⚖️ KARAR: {dec}")
        print(f"📝 Açıklama: {expl}")