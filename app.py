from xml.parsers.expat import model
import streamlit as st
import pandas as pd
from neuro_modules import ui  # Az önce yarattığımız görselci
from neuro_modules import market_data
from neuro_modules import news_scraper
from neuro_modules import ai_engine


# Sayfa Ayarları
st.set_page_config(page_title="NeuroQuant v2.0", page_icon="🧠", layout="wide")

def main():
    # 1. Kenar Çubuğunu Çiz ve Girdileri Al
    ticker, is_clicked = ui.render_sidebar()
    
    # is_clicked True ise (Butona basıldıysa) VEYA ticker değiştiyse çalıştırabiliriz.
    # Şimdilik sadece butona basınca çalışsın.
    if is_clicked:
        # 2. Beyinleri Yükle (Cache sayesinde hızlıdır)
        # Scaler'ı sildik, sadece 2 değişken alıyoruz
        model, sentiment_pipe = ai_engine.load_brains()
        
        if not model or not sentiment_pipe:
            st.error("Modeller yüklenemedi! Lütfen kurulumu kontrol et.")
            return

        with st.spinner(f'{ticker} için yapay zeka çalışıyor...'):
            try:
                # 3. Veri Toplama (Data Pipeline)
                df = market_data.get_rich_market_data(ticker, period="1y")
                news_list = news_scraper.get_google_news(ticker)
                
                # 4. Analiz (Intelligence Layer)
                # a) Teknik Tahmin
                last_60_days = df.tail(60)
                # Verimizin adı 'df', onu gönderiyoruz
                future_preds = ai_engine.predict_future(model, df)
                
                # b) Duygu Analizi (Veto Mekanizmalı)
                avg_sentiment, label, risky_news = ai_engine.score_news(sentiment_pipe, news_list)
                
                # c) Karar Mekanizması (Logic Layer)
                current_rsi = df['RSI'].iloc[-1]
                decision, color, explanation = ai_engine.make_final_decision(
                    future_preds, avg_sentiment, risky_news, current_rsi
                )
                
                # 5. Ekrana Basma (UI Layer)
                current_price = df['Close'].iloc[-1]
                
                ui.render_header(ticker, current_price)
                ui.render_veto_warning(risky_news) 
                
                # SEKMELİ YAPI (TABS)
                tab1, tab2, tab3 = st.tabs(["🚀 Ana Özet", "📊 Teknik Detaylar", "📰 Haber Masası"])
                
                with tab1:
                    # Eski usül temiz görünüm
                    ui.render_decision_gauge(decision, color, explanation, avg_sentiment)
                    ui.render_chart(df, future_preds)
                
                with tab2:
                    # Yeni Hacim ve RSI Grafikleri
                    ui.render_technical_charts(df)
                    
                with tab3:
                    # Yeni Haber Kartları (AI Puanlı)
                    ui.render_news_cards(news_list)
                
                # --- GÜNCELLEME BİTTİ ---

            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
                
    else:
        st.info("👈 Analizi başlatmak için soldaki butona basınız.")

if __name__ == "__main__":
    main()