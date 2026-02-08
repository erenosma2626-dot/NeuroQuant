import streamlit as st
import numpy as np
import pandas as pd
import joblib
from transformers import pipeline
import os

# --- 1. AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

@st.cache_resource
def load_brains():
    """
    Evrensel Random Forest Modelini ve FinBERT'i yükler.
    Artık Scaler yok, çünkü Random Forest buna ihtiyaç duymaz.
    """
    print("🧠 AI Motorları Yükleniyor...")
    
    # A) Random Forest (Fiyat Tahmini)
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'universal_rf.pkl'))
    except Exception as e:
        st.error(f"🚨 Model Dosyası Bulunamadı: {e}")
        st.warning("⚠️ Lütfen önce 'python training/train_universal.py' kodunu çalıştırın.")
        return None, None

    # B) FinBERT (Haber Analizi)
    try:
        sentiment_pipe = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
    except Exception as e:
        st.error(f"🚨 FinBERT yüklenemedi: {e}")
        return model, None

    return model, sentiment_pipe

# --- 2. TEKNİK ANALİZ MOTORU ---
def predict_future(model, last_60_days_df):
    """
    Random Forest ile Gelecek Tahmini.
    Fiyatları değil, % Değişimleri kullanır.
    """
    # Veri Hazırlığı
    prices = last_60_days_df['Close'].values
    
    # Fiyatı Yüzde Değişime Çevir
    pct_changes = pd.Series(prices).pct_change().fillna(0).values
    
    # Yeterli veri yoksa (Yeni halka arz vb.)
    if len(pct_changes) < 60:
        return [prices[-1]] * 5
        
    # Son 60 günün değişimini modele ver (2D Array olarak)
    input_features = pct_changes[-60:].reshape(1, -1)
    
    # Tahmin (Gelecek 5 günün % değişimi)
    # Random Forest direkt 5 çıktılı vektör verir
    pred_pcts = model.predict(input_features)[0]
    
    # Fiyatı Geri İnşa Et (Reconstruct Price)
    current_price = prices[-1]
    future_prices = []
    
    for pct in pred_pcts:
        # Güvenlik Limiti (%5) - Modelin uçmasını engeller
        if pct > 0.05: pct = 0.05
        if pct < -0.05: pct = -0.05
            
        next_price = current_price * (1 + pct)
        future_prices.append(next_price)
        current_price = next_price
        
    return future_prices

# --- 3. DUYGU ANALİZİ (VETO DESTEKLİ) ---
# Bu kısım eski kodun aynısı, çünkü UI burayı kullanıyor.
def score_news(sentiment_pipe, news_list):
    """Haberleri puanlar ve risk analizi yapar."""
    if not news_list:
        return 0, "Nötr", None
    
    total_score = 0
    analyzed_count = 0
    min_score = 1.0 
    riskiest_news = None 
    RISK_THRESHOLD = -0.20 
    
    for news in news_list:
        text = news['title']
        try:
            # FinBERT Analizi
            result = sentiment_pipe(text[:512])[0]
            label = result['label']
            confidence = result['score']
        except:
            continue
        
        if label == 'Positive':
            ai_score = confidence
        elif label == 'Negative':
            ai_score = -confidence
        else:
            ai_score = 0
            
        # UI için skoru habere yapıştır
        news['ai_score'] = ai_score 
        
        # Risk Takibi (En kötü haberi bul)
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

# --- 4. KARAR MEKANİZMASI ---
def make_final_decision(lstm_preds, sentiment_score, riskiest_news, current_rsi):
    """Yatırım Kararını Verir."""
    # Fiyat Değişimi Hesabı
    start_price = lstm_preds[0]
    end_price = lstm_preds[-1]
    price_change_pct = ((end_price - start_price) / start_price) * 100
    
    decision = "NÖTR / İZLE"
    color = "gray"
    explanation = "Yeterli sinyal oluşmadı."
    
    # 1. RSI Kontrolü
    if current_rsi > 70:
        return "RİSKLİ (RSI Şişik)", "orange", f"RSI {current_rsi:.0f} seviyesinde, düzeltme gelebilir."
        
    # 2. Haber Vetosusu
    if riskiest_news and sentiment_score < 0:
        return "SAT / UZAK DUR", "red", f"Riskli haber tespit edildi: '{riskiest_news['title']}'."

    # 3. Trend Kararı
    if price_change_pct > 0.1:
        if sentiment_score > 0: # Haber de biraz pozitifse yeter
            return "AL (Fırsat)", "green", f"Model %{price_change_pct:.2f} yükseliş öngörüyor."
        else:
            return "AL (Riskli)", "blue", "Model yükseliş bekliyor ama haberler desteklemiyor."

    elif price_change_pct < -0.1:
        return "SAT", "red", f"Model %{price_change_pct:.2f} düşüş öngörüyor."
        
    return decision, color, explanation