import os
# --- MAC DONMA ÇÖZÜMÜ (EN TEPEYE) ---
# TensorFlow'un Mac GPU'sunu görmesini engelliyoruz.
# Sadece CPU kullanarak kilitlenmeyi önleriz.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from transformers import pipeline
import google.generativeai as genai

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

@st.cache_resource
def load_brains():
    """
    LSTM Modelini, Scaler'ı ve FinBERT'i yükler.
    Artık 3 parça dönüyor: Model, Scaler, Pipe.
    """
    print("🧠 LSTM Motorları Yükleniyor...")
    
    # 1. LSTM Modeli (.h5)
    try:
        model = load_model(os.path.join(MODEL_DIR, 'universal_lstm.h5'))
    except Exception as e:
        st.error(f"🚨 Model Dosyası Bulunamadı: {e}")
        return None, None, None

    # 2. Scaler (.pkl) - LSTM için şart!
    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, 'universal_scaler.pkl'))
    except Exception as e:
        st.error(f"🚨 Scaler Bulunamadı: {e}")
        return model, None, None

    # 3. FinBERT (Haber Analizi)
    try:
        sentiment_pipe = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
    except Exception as e:
        st.warning(f"⚠️ FinBERT yüklenemedi (Haber analizi çalışmayacak): {e}")
        return model, scaler, None

    return model, scaler, sentiment_pipe

# --- TEKNİK ANALİZ MOTORU (LSTM UYUMLU) ---
def predict_future(model, scaler, df):
    """
    LSTM ile Gelecek Tahmini.
    Scaler kullanarak veriyi 0-1 arasına sıkıştırır ve 3D formatına sokar.
    """
    # 1. Veriyi Hazırla (% Değişim)
    prices = df['Close'].values
    pct_changes = df['Close'].pct_change().fillna(0).values.reshape(-1, 1)
    
    # Yeterli veri kontrolü (60 gün lazım)
    if len(pct_changes) < 60:
        return [prices[-1]] * 5
        
    # 2. Ölçeklendir (Scaling)
    # Model 0-1 arası sayılarla eğitildi, aynısını verelim.
    scaled_data = scaler.transform(pct_changes)
    
    # 3. Son 60 günü al ve Reshape yap (1, 60, 1)
    # (Batch Size, Time Steps, Features)
    current_batch = scaled_data[-60:].reshape(1, 60, 1)
    
    # 4. Tahmin Et
    predicted_scaled = model.predict(current_batch, verbose=0)[0] # Çıktı: [0.5, 0.6, ...]
    
    # 5. Ters Ölçeklendir (Inverse Transform)
    # Modelin ürettiği 0-1 arası sayıları tekrar % değişime çevir.
    predicted_pcts = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()
    
    # 6. Fiyatı İnşa Et
    current_price = prices[-1]
    future_prices = []
    
    # Dinamik Volatilite Tavanı (3 * 20 günlük oynaklık, en az %5, en fazla %25)
    rolling_vol = df['Close'].pct_change().tail(20).std()
    if np.isnan(rolling_vol) or rolling_vol == 0:
        vol_limit = 0.10
    else:
        vol_limit = float(np.clip(3 * rolling_vol, 0.05, 0.25))

    for pct in predicted_pcts:
        # Dinamik volatilite filtresi (varlığın doğal rejimine duyarlı)
        if pct > vol_limit: pct = vol_limit
        if pct < -vol_limit: pct = -vol_limit
            
        next_price = current_price * (1 + pct)
        future_prices.append(next_price)
        current_price = next_price
        
    return future_prices

# --- DUYGU ANALİZİ (AYNI KALDI) ---
def score_news(sentiment_pipe, news_list):
    if not news_list or not sentiment_pipe:
        return 0, "Nötr", None
    
    total_score = 0
    analyzed_count = 0
    min_score = 1.0 
    riskiest_news = None 
    
    for news in news_list:
        try:
            result = sentiment_pipe(news['title'][:512])[0]
            score = result['score'] if result['label'] == 'Positive' else -result['score'] if result['label'] == 'Negative' else 0
            news['ai_score'] = score
            
            if score < min_score:
                min_score = score
                if score < -0.2: riskiest_news = news 
            
            total_score += score
            analyzed_count += 1
        except: continue
        
    if analyzed_count == 0: return 0, "Nötr", None
    
    avg = total_score / analyzed_count
    label = "POZİTİF" if avg > 0.15 else "NEGATİF" if avg < -0.15 else "NÖTR"
    return avg, label, riskiest_news

# --- KARAR MEKANİZMASI ---
def make_final_decision(preds, sentiment_score, riskiest_news, current_rsi=None):
    start_p = preds[0]
    end_p = preds[-1]
    change_pct = ((end_p - start_p) / start_p) * 100
    
    # Not: Yanıltıcı aşırı alım sinyallerini önlemek için RSI filtresi kaldırıldı.
    if riskiest_news and sentiment_score < 0: return "SAT / UZAK DUR", "red", "Riskli Haber Var"

    # Eşiği LSTM / Quantile için hassas tutalım (0.1 ideal)
    if change_pct > 0.1:
        if sentiment_score > 0: return "GÜÇLÜ AL 🚀", "green", f"Model %{change_pct:.2f} Artış Bekliyor"
        else: return "AL (Teknik)", "blue", "Yükseliş Beklentisi"
    elif change_pct < -0.1:
        return "SAT", "red", "Düşüş Beklentisi"
        
    return "İZLE / NÖTR", "gray", "Yatay Seyir Beklentisi"



def ask_gemini(ticker, price, rsi=None, macd_signal="", decision="", news_list=None, sentiment_score=0.0):
    """
    Gemini Pro'ya HEM TEKNİK HEM HABER verilerini gönderip hibrit yorum ister.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                if "GEMINI_API_KEY" in st.secrets:
                    api_key = st.secrets["GEMINI_API_KEY"]
            except Exception:
                pass

        if not api_key:
            return "⚠️ Hata: 'GEMINI_API_KEY' ortam değişkeninde veya Streamlit Secrets içinde bulunamadı."

        # 2. Modeli Hazırla
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')

        # 3. Haberleri Özetle (İlk 3 başlığı alalım ki model boğulmasın)
        news_summary = "Henüz güncel haber yok."
        if news_list:
            titles = [f"- {n.get('title', '')}" for n in news_list[:3]]
            news_summary = "\n".join(titles)

        rsi_line = f"- RSI: {rsi:.2f}\n" if rsi is not None else ""

        # 4. Soruyu Hazırla (Prompt Engineering - Hibrit Analiz)
        prompt = f"""
        Sen profesyonel bir finansal stratejistsin. Aşağıdaki verileri birleştirerek {ticker} için bir analiz yaz.
        
        A) TEKNİK GÖSTERGELER:
        - Fiyat: {price}
        {rsi_line}- MACD Durumu: {macd_signal}
        - Algoritma Kararı: {decision}
        
        B) TEMEL ANALİZ (HABERLER & DUYGU):
        - Piyasa Duygusu Skoru: {sentiment_score:.2f} (-1 Negatif, +1 Pozitif)
        - Son Başlıklar:
        {news_summary}
        
        GÖREVİN:
        Teknik veriler ile haber akışını kıyasla. Örneğin teknik "AL" derken haberler "KÖTÜ" ise bu bir tuzak mı?
        Yoksa ikisi de birbirini destekliyor mu?
        Yatırım tavsiyesi vermeden, riskleri ve fırsatları 3-4 cümleyle, akıcı bir Türkçe ile anlat.
        """

        # 5. Cevabı Al
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Üzgünüm, Gemini şu an yanıt veremiyor. Hata: {str(e)}"