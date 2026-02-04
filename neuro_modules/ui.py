import streamlit as st
import plotly.graph_objects as go
import textwrap

def render_sidebar():
    """Yan menüyü çizer (GÜNCELLENDİ: Watchlist Eklendi)."""
    with st.sidebar:
        st.header("🧠 NeuroQuant v2.1")
        st.markdown("---")
        
        # 1. FAVORİLER (Hızlı Geçiş Butonları)
        st.subheader("🔥 Hızlı Erişim")
        
        # Session State kontrolü (Hafızada ticker var mı?)
        if 'ticker' not in st.session_state:
            st.session_state.ticker = "NVDA"
            
        # Butonlar için yardımcı fonksiyon
        def set_ticker(symbol):
            st.session_state.ticker = symbol
        
        # 3 Kolonlu Buton Yapısı
        col1, col2, col3 = st.columns(3)
        if col1.button("NVDA"): set_ticker("NVDA")
        if col2.button("TSLA"): set_ticker("TSLA")
        if col3.button("AAPL"): set_ticker("AAPL")
        
        col4, col5, col6 = st.columns(3)
        if col4.button("AMD"): set_ticker("AMD")
        if col5.button("AMZN"): set_ticker("AMZN")
        if col6.button("BTC"): set_ticker("BTC-USD") # Kripto desteği!

        st.markdown("---")
        
        # 2. MANUEL GİRİŞ
        # 'key="ticker"' diyerek bu kutuyu session_state'e bağladık.
        # Butona basınca burası otomatik değişecek.
        ticker = st.text_input("Hisse Kodu Girin:", key="ticker")
        st.caption("Örn: MSFT, GOOGL, ETH-USD")
        
        st.markdown("---")
        
        # 3. BAŞLAT BUTONU
        analyze_btn = st.button("Analizi Başlat ▶", type="primary", use_container_width=True)
        
        st.markdown("---")
        with st.expander("ℹ️ Sistem Durumu"):
            st.success("✅ AI Motoru: Aktif")
            st.success("✅ Veri Akışı: Online")
            st.info("v2.1 Stable")
            
        return ticker, analyze_btn

def render_header(ticker, current_price):
    """Ana sayfa başlığını ve güncel fiyatı gösterir."""
    st.title(f"📊 Analiz Raporu: {ticker.upper()}")
    st.markdown(f"**Güncel Fiyat:** `${current_price:.2f}`")
    st.markdown("---")

def render_decision_gauge(decision, color, explanation, sentiment_score):
    """Karar ibresini ve açıklamasını gösterir."""
    
    # Renk kodlarını CSS stiline çevir
    color_map = {"green": "#2ecc71", "red": "#e74c3c", "orange": "#f39c12", "gray": "#95a5a6", "blue": "#3498db"}
    css_color = color_map.get(color, "#95a5a6")
    
    # 3 Kolonlu Yapı
    k1, k2, k3 = st.columns(3)
    k1.metric("Yapay Zeka Kararı", decision)
    k1.markdown(f":color[{decision}]") # Streamlit renk desteği
    
    k2.metric("Haber Duygu Skoru", f"{sentiment_score:.2f}")
    
    # Karar Kutusu (Renkli Arka Plan)
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {css_color}20; border: 1px solid {css_color};">
        <h3 style="color: {css_color}; margin:0;">AI Gerekçesi:</h3>
        <p style="font-size: 16px; margin-top: 10px;">{explanation}</p>
    </div>
    """, unsafe_allow_html=True)

def render_veto_warning(risky_news):
    """Eğer veto yemiş bir haber varsa kırmızı alarm verir."""
    if risky_news:
        st.error(f"⚠️ **KRİTİK RİSK UYARISI:** Sistem genel olarak olumlu olsa bile şu haber risk yaratıyor:")
        st.markdown(f"👉 **[{risky_news['title']}]({risky_news['link']})**")
        st.markdown("---")

def render_chart(history_df, future_prices):
    """Geçmiş ve Gelecek grafiğini çizer (ZOOM YAPILMIŞ HALİ)."""
    st.subheader("📈 Fiyat Projeksiyonu (LSTM)")
    
    fig = go.Figure()
    
    # 1. Geçmiş Veri (SADECE SON 14 GÜN - ODAKLANMA)
    # Kullanıcı bugünü daha net görsün diye tarihi kısalttık
    lookback = 14 
    fig.add_trace(go.Scatter(
        x=history_df.index[-lookback:], 
        y=history_df['Close'].tail(lookback),
        mode='lines',
        name='Geçmiş (Gerçek)',
        line=dict(color='#3498db', width=3)
    ))
    
    # 2. Gelecek Tahmini
    last_date = history_df.index[-1]
    # İş günü (Business Day) yerine normal gün kullanalım ki hafta sonu boşluğu grafiği koparmasın
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5)
    
    chart_y = [history_df['Close'].iloc[-1]] + list(future_prices)
    chart_x = [last_date] + list(future_dates)
    
    fig.add_trace(go.Scatter(
        x=chart_x,
        y=chart_y,
        mode='lines+markers',
        name='AI Tahmini (5 Gün)',
        line=dict(color='#e67e22', width=3, dash='dot')
    ))
    
    # Grafiği biraz daha estetik yapalım
    # Grafiği estetik yap ve ZOOM'u kilitle
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        # EKSENLERİ KİLİTLEME (Kullanıcı bozamasın)
        xaxis=dict(fixedrange=True, title="Tarih"), 
        yaxis=dict(fixedrange=True, title="Fiyat ($)")
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': False, 'scrollZoom': False})
    
# Imports inside chart function to avoid circular dependency if needed, 
# but putting them at top is fine for now.
import pandas as pd

# --- MEVCUT KODLARIN ALTINA EKLE ---

def render_technical_charts(df):
    """Teknik Analiz Sekmesi: RSI ve Hacim Grafikleri."""
    
    # 1. Hacim (Volume) Grafiği
    st.subheader("📊 İşlem Hacmi (Volume)")
    
    # Renkleri belirle: Kapanış > Açılış ise Yeşil, değilse Kırmızı
    colors = ['#2ecc71' if c >= o else '#e74c3c' for c, o in zip(df['Close'], df['Open'])]
    
    fig_vol = go.Figure(data=[go.Bar(
        x=df.index[-30:], # Son 30 gün
        y=df['Volume'].tail(30),
        marker_color=colors
    )])
    
    fig_vol.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
    st.plotly_chart(fig_vol, use_container_width=True, config={'staticPlot': False, 'scrollZoom': False})
    
    # 2. RSI Grafiği
    st.subheader("📉 RSI Momentum (70=Pahalı, 30=Ucuz)")
    
    fig_rsi = go.Figure(data=[go.Scatter(
        x=df.index[-30:],
        y=df['RSI'].tail(30),
        mode='lines',
        line=dict(color='#9b59b6', width=2),
        name='RSI'
    )])
    
    # Referans Çizgileri (30 ve 70)
    fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Aşırı Alım")
    fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Aşırı Satım")
    
    fig_rsi.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10), 
                          yaxis=dict(range=[0, 100], fixedrange=True), xaxis=dict(fixedrange=True))
    st.plotly_chart(fig_rsi, use_container_width=True, config={'staticPlot': False, 'scrollZoom': False})

def render_news_cards(news_list):
    """Haberleri sıkıcı liste yerine şık kartlar olarak gösterir."""
    st.subheader("📰 Yapay Zeka Haber Analizi")
    
    if not news_list:
        st.info("Gösterilecek haber bulunamadı.")
        return

    # Haberleri 2'li kolonlar halinde dizelim
    for i in range(0, len(news_list), 2):
        col1, col2 = st.columns(2)
        
        # 1. Haber
        news1 = news_list[i]
        with col1:
            _draw_single_card(news1)
            
        # 2. Haber (Eğer varsa)
        if i + 1 < len(news_list):
            news2 = news_list[i+1]
            with col2:
                _draw_single_card(news2)

def _draw_single_card(news):
    """Tek bir haber kartını çizer (GÜNCELLENDİ: Boşluk Hatası Giderildi)."""
    score = news.get('ai_score', 0)
    
    # Skora göre renk ve ikon belirle
    if score > 0.15: 
        border_color = "#2ecc71" # Yeşil
        icon = "🐂"
        score_text = f"POZİTİF ({score:.2f})"
        bg_color = "rgba(46, 204, 113, 0.1)" 
    elif score < -0.15:
        border_color = "#e74c3c" # Kırmızı
        icon = "🐻"
        score_text = f"NEGATİF ({score:.2f})"
        bg_color = "rgba(231, 76, 60, 0.1)" 
    else:
        border_color = "#95a5a6" # Gri
        icon = "😐"
        score_text = "NÖTR" 
        bg_color = "rgba(149, 165, 166, 0.1)"
        
    # HTML KODU (textwrap.dedent kullanacağız)
    html_code = f"""
    <div style="
        border-left: 4px solid {border_color};
        padding: 12px;
        background-color: {bg_color};
        border-radius: 8px;
        margin-bottom: 12px;
    ">
        <div style="display: flex; justify-content: space-between; font-size: 12px; color: #bdc3c7; margin-bottom: 5px;">
            <span>{news['source']}</span>
            <span>{news['published']}</span>
        </div>
        <h5 style="margin: 0 0 10px 0; font-size: 16px;">
            <a href="{news['link']}" target="_blank" style="text-decoration: none; color: #ecf0f1;">{news['title']}</a>
        </h5>
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span style="font-weight: bold; color: {border_color}; font-size: 14px;">
                {icon} {score_text}
            </span>
        </div>
    </div>
    """
    
    # DEDENT FONKSİYONU İLE BOŞLUKLARI TEMİZLİYORUZ
    st.markdown(textwrap.dedent(html_code), unsafe_allow_html=True)