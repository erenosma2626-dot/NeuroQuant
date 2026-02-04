import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from transformers import pipeline
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. MODELLERİ ÖNBELLEĞE ALARAK YÜKLE
@st.cache_resource
def load_all_engines():
    # LSTM Beyni
    model = load_model('neuroquant_lstm.h5')
    # Ölçekleyici
    scaler = joblib.load('scaler.pkl')
    # Sözel Zeka (Haber Analizcisi)
    sentiment_pipe = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
    return model, scaler, sentiment_pipe

nn_model, nn_scaler, finbert = load_all_engines()


# 2. TEKNİK TAHMİN FONKSİYONU
def generate_technical_forecast(model, scaler, last_60_days_df):
    print(">>> Tahmin Motoru: Veri hazırlanıyor...")
    # Sadece gerekli sütunları alalım
    cols = ['Close', 'Open', 'High', 'Low', 'Volume']
    raw_data = last_60_days_df[cols].values
    
    # Ölçeklendir
    scaled_data = scaler.transform(raw_data)
    
    # Başlangıç paketini hazırla (1, 60, 5)
    current_batch = scaled_data.reshape(1, 60, 5)
    forecast_scaled = []
    
    print(">>> 30 Günlük Döngü Başlıyor...")
        # Mevcut döngü kısmını bununla değiştirerek test et:
    for i in range(30):
        # predict() yerine doğrudan model() çağrısı bazen daha hızlıdır
        pred = model(current_batch, training=False) 
        pred_value = pred.numpy()[0][0]
        forecast_scaled.append(pred_value)
        
        # Pencere kaydırma
        new_row = np.array([pred_value, current_batch[0, -1, 1], current_batch[0, -1, 2], 
                        current_batch[0, -1, 3], current_batch[0, -1, 4]]).reshape(1, 1, 5)
        
        current_batch = np.append(current_batch[:, 1:, :], new_row, axis=1)
        
        # Her adımda terminale bir işaret koy ki yaşadığını görelim
        print(f"DEBUG: {i+1}. gün hesaplandı...")
    print(">>> Döngü bitti, ters ölçeklendirme yapılıyor...")
    # Tahminleri gerçek dolar değerine geri çevir
    dummy_df = np.zeros((30, 5))
    dummy_df[:, 0] = forecast_scaled
    unscaled_preds = scaler.inverse_transform(dummy_df)[:, 0]
    
    return unscaled_preds

# 3. STREAMLIT ARAYÜZÜ
st.title("🧠 NeuroQuant v2.0: Technical Engine")

if st.button("Teknik Analizi ve Tahmini Başlat"):
    print(">>> Butona basıldı, veri çekme başlıyor...") # Terminalde görünecek
    with st.spinner('Canlı veri çekiliyor...'):
        ticker = yf.Ticker("NVDA")
        hist = ticker.history(period="100d")
        print(f">>> Veri çekildi. Satır sayısı: {len(hist)}")

        if len(hist) >= 60:
            print(">>> LSTM Tahmin döngüsü başladı...")
            last_60_days = hist.tail(60)
            preds = generate_technical_forecast(nn_model, nn_scaler, last_60_days)
            print(">>> Tahmin tamamlandı!")
            # 3. Tarihleri Hazırla
            future_dates = pd.date_range(start=hist.index[-1] + timedelta(days=1), periods=30, freq='B')
            
            # 4. Grafik
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index[-30:], y=hist['Close'].tail(30), name="Geçmiş (Canlı)"))
            fig.add_trace(go.Scatter(x=future_dates, y=preds, name="Teknik Projeksiyon (LSTM)", line=dict(dash='dot', color='orange')))
            
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Güncel Fiyat: {hist['Close'].iloc[-1]:.2f}$ | 30 Günlük Teknik Beklenti: {preds[-1]:.2f}$")
        else:
            st.error("Yeterli veri çekilemedi.")
            