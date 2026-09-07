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
    ticker, btn_press = ui.render_sidebar()

    # 2. HAFIZA SİSTEMİ (Köprü Kuruyoruz)
    # Eğer hafıza kutusu yoksa oluştur
    if 'analiz_aktif' not in st.session_state:
        st.session_state['analiz_aktif'] = False

    # Eğer butona basıldıysa hafızayı 'AÇIK' yap
    if btn_press:
        st.session_state['analiz_aktif'] = True
    
    # Eğer kullanıcı hisseyi değiştirirse analizi kapat (Yeniden başlatabilsin)
    if 'son_hisse' not in st.session_state: st.session_state['son_hisse'] = ticker
    if ticker != st.session_state['son_hisse']:
        st.session_state['analiz_aktif'] = False
        st.session_state['son_hisse'] = ticker

    # 3. SİHİRLİ DOKUNUŞ:
    # Artık is_clicked değişkeni, anlık butona değil, HAFIZAYA bağlı.
    # Böylece sayfa yenilense bile True kalır!
    is_clicked = st.session_state['analiz_aktif']


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
                    # --- ZAMAN DİLİMİ AYARI (ui.py'ye dokunmadan ekliyoruz) ---
                
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

                # --- SİNYAL ÖZET TABLOSU (Auto-Interpreter) ---
                st.markdown("---")
                st.subheader("🤖 Algoritmik Sinyal Özeti")
                
                # En son verileri alalım
                last_rsi = df['RSI'].iloc[-1]
                last_macd = df['MACD'].iloc[-1]
                last_macd_signal = df['MACD_Signal'].iloc[-1]
                last_close = df['Close'].iloc[-1]
                last_bb_upper = df['BB_Upper'].iloc[-1]
                last_bb_lower = df['BB_Lower'].iloc[-1]
                
                # 1. RSI Yorumu
                if last_rsi < 30:
                    rsi_signal = "🟢 GÜÇLÜ AL (Aşırı Satım)"
                elif last_rsi > 70:
                    rsi_signal = "🔴 GÜÇLÜ SAT (Aşırı Alım)"
                else:
                    rsi_signal = "⚪ NÖTR"
                    
                # 2. MACD Yorumu
                if last_macd > last_macd_signal:
                    macd_signal = "🟢 AL (Pozitif Trend)"
                else:
                    macd_signal = "🔴 SAT (Negatif Trend)"
                    
                # 3. Bollinger Yorumu
                if last_close > last_bb_upper:
                    bb_signal = "🔴 SAT (Fiyat Çok Yüksek)"
                elif last_close < last_bb_lower:
                    bb_signal = "🟢 AL (Fiyat Çok Düşük)"
                else:
                    bb_signal = "⚪ NÖTR (Bant İçinde)"

                # Tabloyu Oluştur
                signal_data = {
                    "İndikatör": ["RSI (Momentum)", "MACD (Trend)", "Bollinger (Volatilite)"],
                    "Değer": [f"{last_rsi:.2f}", f"{last_macd:.2f}", f"{last_close:.2f}"],
                    "Yapay Zeka Sinyali": [rsi_signal, macd_signal, bb_signal]
                }
                st.table(pd.DataFrame(signal_data))
                

                # ----------------------------------------------
                st.markdown("---")
                st.subheader("✨ Yapay Zeka Yorumu (Teknik + Haberler)")
                
                if st.button("🤖 Piyasayı Yorumla (Gemini)"):
                    with st.spinner("Gemini teknik verileri ve haberleri sentezliyor..."):
                        # Fonksiyonu YENİ parametrelerle çağırıyoruz
                        ai_comment = ai_engine.ask_gemini(
                            ticker, 
                            last_close, 
                            last_rsi, 
                            macd_signal, 
                            decision,
                            news_list,      # <-- Yeni eklendi: Haber Listesi
                            avg_sentiment   # <-- Yeni eklendi: Duygu Skoru
                        )
                        
                        # Sonucu Göster
                        st.info(ai_comment)
                        st.caption("Not: Bu yorum Google Gemini yapay zekası tarafından oluşturulmuştur.")
                # ------------------------------------------
                
                # ----------------------------------------------

                st.markdown("---")
                st.subheader("📥 Analiz Çıktısı")
                
                    # Veriyi CSV formatına çeviriyoruz (Risk yok, sadece format değişiyor)
                csv_data = df.to_csv().encode('utf-8')
                    
                st.download_button(
                    label="💾 Tüm Verileri ve İndikatörleri İndir (Excel/CSV)",
                    data=csv_data,
                    file_name=f"{ticker}_analiz_verisi.csv",
                    mime='text/csv',
                    use_container_width=True
                )
                

            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
                
    else:
        st.info("👈 Analizi başlatmak için soldaki butona basınız.")

if __name__ == "__main__":
    main()