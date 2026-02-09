from xml.parsers.expat import model
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from neuro_modules import ui  # Az önce yarattığımız görselci
from neuro_modules import market_data
from neuro_modules import news_scraper
from neuro_modules import ai_engine


# Sayfa Ayarları
st.set_page_config(page_title="NeuroQuant v2.0", page_icon="🧠", layout="wide")

def main():
    # 1. Kenar Çubuğunu Çiz ve Girdileri Al
    ticker, is_clicked = ui.render_sidebar()
    
    with st.expander("ℹ️ Proje Amacı ve Yasal Uyarı (Lütfen Okuyunuz)", expanded=False):
        st.markdown("""
        ### 🧠 NeuroQuant Nedir?
        Bu proje, finansal piyasaları analiz etmek için **Yapay Zeka (LSTM & FinBERT)** teknolojilerini kullanan deneysel bir analiz aracıdır. Geçmiş verilerden öğrenerek teknik analiz yapar ve haber akışlarını yorumlar.
        
        ---
        
        ### ⚠️ YASAL UYARI (YTD)
        **Burada yer alan bilgi, yorum ve tavsiyeler Yatırım Danışmanlığı kapsamında DEĞİLDİR.**
        * Bu uygulama sadece **eğitim ve analiz** amaçlı geliştirilmiştir.
        * Yapay zeka tahminleri geleceği garanti edemez ve hata payı içerir.
        * Yatırım kararlarınızı kendi araştırmanıza veya yetkili yatırım danışmanlarına dayanarak veriniz.
        """)
    # is_clicked True ise (Butona basıldıysa) VEYA ticker değiştiyse çalıştırabiliriz.
    # Şimdilik sadece butona basınca çalışsın.
    if is_clicked:
        # 2. Beyinleri Yükle (Cache sayesinde hızlıdır)
        # Scaler'ı sildik, sadece 2 değişken alıyoruz
        model, scaler, sentiment_pipe = ai_engine.load_brains()
        
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
                future_preds = ai_engine.predict_future(model, scaler, df)
                
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
                
                    # --- EKLENEN KISIM: Yeni Grafikler ---
                    with st.expander("📊 Gelişmiş Teknik Analiz (Bollinger & MACD)", expanded=True):
                        # 1. Bollinger Grafiği
                        st.caption("Bollinger Bantları (Volatilite)")
                        fig_bb = go.Figure()
                        fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='Üst Bant', line=dict(color='gray', width=1, dash='dot')))
                        fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='Alt Bant', line=dict(color='gray', width=1, dash='dot'), fill='tonexty'))
                        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Fiyat', line=dict(color='blue', width=2)))
                        fig_bb.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
                        st.plotly_chart(fig_bb, use_container_width=True)
                        
                        # 2. MACD Grafiği
                        st.caption("MACD (Trend Yönü)")
                        fig_macd = go.Figure()
                        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='green')))
                        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Sinyal', line=dict(color='red')))
                        fig_macd.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0))
                        st.plotly_chart(fig_macd, use_container_width=True)
                # -------------------------------------
                    
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