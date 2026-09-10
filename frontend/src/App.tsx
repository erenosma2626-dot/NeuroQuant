import React, { useState, useEffect } from 'react';
import { Navigation } from './components/Navigation';
import { DashboardPage } from './components/DashboardPage';
import { PortfolioPage } from './components/PortfolioPage';
import { MarketBanner } from './components/MarketBanner';
import { TradingViewChart } from './components/TradingViewChart';
import { QuantMatrix } from './components/QuantMatrix';
import { SimulationLab } from './components/SimulationLab';
import { FundamentalRadar } from './components/FundamentalRadar';
import { AgentTerminal } from './components/AgentTerminal';
import { NewsFeed } from './components/NewsFeed';
import type { 
  MarketData, 
  ForecastData, 
  FundamentalsData, 
  SimulationData, 
  AgentCommentData,
  ScreenerItem,
  UserPortfolio,
  MacroBarometerData,
  NewsData,
} from './types';
import { AlertCircle, RefreshCw } from 'lucide-react';

const DEFAULT_PORTFOLIO: UserPortfolio = {
  name: 'Wall Street Portföyü',
  initial_capital: 10000,
  cash: 3400,
  positions: [
    {
      ticker: 'NVDA',
      name: 'Nvidia Corporation',
      category: 'Semis',
      shares: 20,
      buy_price: 122.50,
      current_price: 128.50,
      weight_pct: 38.0,
    },
    {
      ticker: 'AAPL',
      name: 'Apple Inc.',
      category: 'Tech',
      shares: 10,
      buy_price: 218.00,
      current_price: 224.20,
      weight_pct: 33.0,
    },
    {
      ticker: 'BTC-USD',
      name: 'Bitcoin (USD)',
      category: 'Crypto',
      shares: 0.015,
      buy_price: 58500.00,
      current_price: 62450.00,
      weight_pct: 13.8,
    },
  ],
};

export const App: React.FC = () => {
  const [currentTicker, setCurrentTicker] = useState<string>('NVDA');
  const [activeTab, setActiveTab] = useState<'dashboard' | 'terminal' | 'simulation' | 'portfolio'>('dashboard');

  // Screener Universe & Portfolio
  const [screenerData, setScreenerData] = useState<ScreenerItem[]>([]);
  const [userPortfolio, setUserPortfolio] = useState<UserPortfolio>(() => {
    try {
      const saved = localStorage.getItem('neuroquant_portfolio_v3');
      return saved ? JSON.parse(saved) : DEFAULT_PORTFOLIO;
    } catch {
      return DEFAULT_PORTFOLIO;
    }
  });

  // Macro Barometer & News Feeds
  const [macroData, setMacroData] = useState<MacroBarometerData | null>(null);
  const [tickerNews, setTickerNews] = useState<NewsData | null>(null);
  const [globalNews, setGlobalNews] = useState<NewsData | null>(null);

  // Per-Ticker Data
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [forecastData, setForecastData] = useState<ForecastData | null>(null);
  const [fundamentalsData, setFundamentalsData] = useState<FundamentalsData | null>(null);
  const [simulationData, setSimulationData] = useState<SimulationData | null>(null);
  const [agentComment, setAgentComment] = useState<AgentCommentData | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Portföy güncellendiğinde localStorage'a kaydet
  const handleUpdatePortfolio = (updated: UserPortfolio) => {
    setUserPortfolio(updated);
    try {
      localStorage.setItem('neuroquant_portfolio_v3', JSON.stringify(updated));
    } catch (e) {
      console.error(e);
    }
  };

  // 1. Screener Evrenini, Makro Barometreleri ve Küresel Haberleri Çek
  const fetchScreener = async () => {
    try {
      const [scrRes, macroRes, globNewsRes] = await Promise.all([
        fetch('/api/market/screener/all').catch(() => null),
        fetch('/api/macro/barometers').catch(() => null),
        fetch('/api/news/global').catch(() => null),
      ]);

      if (scrRes && scrRes.ok) {
        const data = await scrRes.json();
        setScreenerData(data);
      }
      if (macroRes && macroRes.ok) {
        const mData = await macroRes.json();
        setMacroData(mData);
      }
      if (globNewsRes && globNewsRes.ok) {
        const gData = await globNewsRes.json();
        setGlobalNews(gData);
      }
    } catch (err) {
      console.warn('Initial data fetch failed', err);
    }
  };

  useEffect(() => {
    fetchScreener();
  }, []);

  // 2. Seçili Hisse İçin Tüm Verileri Çek
  const fetchTickerData = async (ticker: string) => {
    setIsLoading(true);
    setError(null);
    // Eski hissenin verilerini hemen temizle (stale state önleme)
    setMarketData(null);
    setForecastData(null);
    setFundamentalsData(null);
    setSimulationData(null);
    setAgentComment(null);
    setTickerNews(null);

    try {
      const [mRes, fRes, fundRes, simRes, agentRes, newsRes] = await Promise.all([
        fetch(`/api/market/${ticker}`).catch(() => null),
        fetch(`/api/forecast/${ticker}`).catch(() => null),
        fetch(`/api/fundamentals/${ticker}`).catch(() => null),
        fetch(`/api/simulation/${ticker}`).catch(() => null),
        fetch(`/api/agent/comment/${ticker}`).catch(() => null),
        fetch(`/api/news/${ticker}`).catch(() => null),
      ]);

      if (!mRes || !mRes.ok) {
        throw new Error(`'${ticker}' sembolü Yahoo Finance üzerinde bulunamadı veya veri çekilemedi.`);
      }

      const mData = await mRes.json();
      setMarketData(mData);

      // Dinamik olarak aranan hisseyi evrene dahil et veya güncelle
      setScreenerData((prev) => {
        const cleanT = ticker.toUpperCase();
        if (prev.some((item) => item.ticker.toUpperCase() === cleanT)) {
          return prev.map((item) =>
            item.ticker.toUpperCase() === cleanT
              ? { ...item, last_close: mData.current_price, change_pct: mData.change_pct }
              : item
          );
        }
        const isCrypto = cleanT.includes('-USD');
        const newItem: ScreenerItem = {
          ticker: cleanT,
          name: cleanT,
          category: isCrypto ? 'Crypto' : 'Tech',
          sector: isCrypto ? 'Kripto Varlık' : 'Wall Street / US Equity',
          last_close: mData.current_price,
          change_pct: mData.change_pct,
          dist_sma200_pct: mData.dist_sma200_pct ?? 0,
          is_golden_cross: mData.is_golden_cross ?? false,
          alpha_20d_cum: mData.alpha_20d_cum ?? 0,
          beta: mData.beta ?? 1,
          ai_signal: 'ANALİZ EDİLDİ',
          confidence_score: 75,
          volume_ratio: mData.volume_ratio ?? 1.0,
        };
        return [newItem, ...prev];
      });

      if (fRes && fRes.ok) setForecastData(await fRes.json());
      if (fundRes && fundRes.ok) setFundamentalsData(await fundRes.json());
      if (simRes && simRes.ok) setSimulationData(await simRes.json());
      if (agentRes && agentRes.ok) setAgentComment(await agentRes.json());
      if (newsRes && newsRes.ok) setTickerNews(await newsRes.json());

    } catch (err: any) {
      console.error(err);
      setError(err.message || `'${ticker}' verisi yüklenirken bir sorun oluştu.`);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchTickerData(currentTicker);
  }, [currentTicker]);

  return (
    <div className="app-container">
      {/* Üst Navigasyon Barı */}
      <Navigation
        currentTicker={currentTicker}
        onSelectTicker={(t) => setCurrentTicker(t)}
        activeTab={activeTab}
        onSelectTab={(tab) => setActiveTab(tab)}
        universe={screenerData}
        activeMarketData={marketData}
      />

      {/* Hata Bildirimi */}
      {error && (
        <div style={{
          padding: '0.75rem 2.5rem',
          background: 'var(--madder-tint)',
          borderBottom: '2px solid var(--madder-rule)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: '0.85rem', color: 'var(--madder-loss)' }}>
            <AlertCircle size={16} />
            <span style={{ fontWeight: 500 }}>{error}</span>
          </div>
          <button
            onClick={() => fetchTickerData(currentTicker)}
            className="btn btn-secondary"
            style={{ padding: '4px 12px', fontSize: '0.78rem' }}
          >
            <RefreshCw size={12} /> Yeniden Dene
          </button>
        </div>
      )}

      {/* Ana Gövde */}
      <main className="main-content">
        {/* SAYFA 1: 🏛️ DASHBOARD (Piyasa Radarı & Tarayıcı) */}
        {activeTab === 'dashboard' && (
          <DashboardPage
            screenerData={screenerData}
            userPortfolio={userPortfolio}
            onSelectTicker={(t) => setCurrentTicker(t)}
            onNavigateTab={(tab) => setActiveTab(tab)}
            macroData={macroData}
          />
        )}

        {/* SAYFA 2: 📈 KANTİTATİF TERMİNAL (TradingView, Güven Konisi & Bilanço) */}
        {activeTab === 'terminal' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            {marketData && <MarketBanner key={`banner-${marketData.ticker}`} data={marketData} />}

            {isLoading || !marketData ? (
              <div className="panel" style={{ height: 480, display: 'flex', alignItems: 'center', justifyContent: 'center', borderTop: '2px solid var(--ink-secondary)' }}>
                <div style={{ textAlign: 'center', color: 'var(--ink-muted)' }}>
                  <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: '1.25rem', marginBottom: 8, color: 'var(--ink-primary)' }}>
                    {currentTicker} Verileri Derleniyor…
                  </div>
                  <div style={{ fontSize: '0.82rem' }}>Google TimesFM 3.0 tahminleri, TradingView grafiği ve bilanço çarpanları yükleniyor</div>
                </div>
              </div>
            ) : (
              <>
                <TradingViewChart key={`chart-${marketData.ticker}`} data={marketData} />

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                  {forecastData && <QuantMatrix key={`quant-${forecastData.ticker}`} forecast={forecastData} />}
                  {fundamentalsData && <FundamentalRadar key={`fund-${fundamentalsData.ticker}`} data={fundamentalsData} />}
                </div>

                {agentComment && <AgentTerminal key={`agent-${currentTicker}`} comment={agentComment} />}

                {/* Dual-Mode Broadsheet Haber & Duygu Beslemesi */}
                <NewsFeed tickerNews={tickerNews} globalNews={globalNews} currentTicker={currentTicker} />
              </>
            )}
          </div>
        )}

        {/* SAYFA 3: 10.000$ SİMÜLASYON LABORATUVARI */}
        {activeTab === 'simulation' && (
          <div>
            {isLoading || !simulationData ? (
              <div className="panel" style={{ padding: '4rem', textAlign: 'center', borderTop: '2px solid var(--ink-secondary)' }}>
                <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: '1.25rem', color: 'var(--ink-primary)', marginBottom: 10 }}>
                  {currentTicker} Simülasyon Verisi Hesaplanıyor…
                </div>
                <p style={{ color: 'var(--ink-secondary)', fontSize: '0.88rem', lineHeight: 1.6, maxWidth: 540, margin: '0 auto' }}>
                  6 aylık geriye dönük test, 10.000$ dinamik sermaye tahsisi, histerezis filtreleri ve XAI karar gerekçeleri derleniyor.
                </p>
              </div>
            ) : (
              <SimulationLab key={simulationData.ticker} simulation={simulationData} />
            )}
          </div>
        )}

        {/* SAYFA 4: 💼 ÖZEL PORTFÖY YÖNETİCİSİ */}
        {activeTab === 'portfolio' && (
          <PortfolioPage
            screenerData={screenerData}
            userPortfolio={userPortfolio}
            onUpdatePortfolio={handleUpdatePortfolio}
            onSelectTicker={(t) => setCurrentTicker(t)}
            onNavigateTab={(tab) => setActiveTab(tab)}
          />
        )}
      </main>
    </div>
  );
};

export default App;
