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
import type { 
  MarketData, 
  ForecastData, 
  FundamentalsData, 
  SimulationData, 
  AgentCommentData,
  ScreenerItem,
  UserPortfolio
} from './types';
import { AlertCircle, RefreshCw } from 'lucide-react';

const DEFAULT_PORTFOLIO: UserPortfolio = {
  name: 'Kurumsal Ana Portföy',
  initial_capital: 10000,
  cash: 4200,
  positions: [
    {
      ticker: 'NVDA',
      name: 'Nvidia Corporation',
      category: 'Tech',
      shares: 12,
      buy_price: 215.00,
      current_price: 230.36,
      weight_pct: 27.6,
    },
    {
      ticker: 'THYAO.IS',
      name: 'Türk Hava Yolları',
      category: 'BIST',
      shares: 6,
      buy_price: 285.00,
      current_price: 296.50,
      weight_pct: 17.8,
    },
    {
      ticker: 'BTC-USD',
      name: 'Bitcoin (USD)',
      category: 'Crypto',
      shares: 0.02,
      buy_price: 61000.00,
      current_price: 64250.00,
      weight_pct: 12.8,
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

  // 1. Screener Evrenini Çek
  const fetchScreener = async () => {
    try {
      const res = await fetch('/api/market/screener/all');
      if (res.ok) {
        const data = await res.json();
        setScreenerData(data);
      }
    } catch (err) {
      console.warn('Screener fetch failed', err);
    }
  };

  useEffect(() => {
    fetchScreener();
  }, []);

  // 2. Seçili Hisse İçin Tüm Verileri Çek
  const fetchTickerData = async (ticker: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const [mRes, fRes, fundRes, simRes, agentRes] = await Promise.all([
        fetch(`/api/market/${ticker}`),
        fetch(`/api/forecast/${ticker}`),
        fetch(`/api/fundamentals/${ticker}`),
        fetch(`/api/simulation/${ticker}`),
        fetch(`/api/agent/comment/${ticker}`),
      ]);

      if (!mRes.ok) throw new Error(`${ticker} piyasa verisi alınamadı.`);

      setMarketData(await mRes.json());
      if (fRes.ok) setForecastData(await fRes.json());
      if (fundRes.ok) setFundamentalsData(await fundRes.json());
      if (simRes.ok) setSimulationData(await simRes.json());
      if (agentRes.ok) setAgentComment(await agentRes.json());

    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Veri yüklenirken bir sorun oluştu.');
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
          />
        )}

        {/* SAYFA 2: 📈 KANTİTATİF TERMİNAL (TradingView, Güven Konisi & Bilanço) */}
        {activeTab === 'terminal' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            {marketData && <MarketBanner data={marketData} />}

            {isLoading ? (
              <div className="panel" style={{ height: 480, display: 'flex', alignItems: 'center', justifyContent: 'center', borderTop: '2px solid var(--ink-secondary)' }}>
                <div style={{ textAlign: 'center', color: 'var(--ink-muted)' }}>
                  <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: '1rem', marginBottom: 8 }}>Veri yükleniyor…</div>
                  <div style={{ fontSize: '0.78rem' }}>Grafik ve göstergeler derleniyor</div>
                </div>
              </div>
            ) : (
              <>
                {marketData && <TradingViewChart data={marketData} />}

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                  {forecastData && <QuantMatrix forecast={forecastData} />}
                  {fundamentalsData && <FundamentalRadar data={fundamentalsData} />}
                </div>

                {agentComment && <AgentTerminal comment={agentComment} />}
              </>
            )}
          </div>
        )}

        {/* SAYFA 3: 10.000$ SİMÜLASYON LABORATUVARI */}
        {activeTab === 'simulation' && (
          <div>
            {simulationData ? (
              <SimulationLab simulation={simulationData} />
            ) : (
              <div className="panel" style={{ padding: '4rem', textAlign: 'center', borderTop: '2px solid var(--ink-secondary)' }}>
                <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: '1.2rem', color: 'var(--ink-primary)', marginBottom: 10 }}>
                  10k Simülasyon Verisi Hesaplanıyor…
                </div>
                <p style={{ color: 'var(--ink-secondary)', fontSize: '0.88rem', lineHeight: 1.6 }}>
                  Histerezis filtreleri, Sharpe oranı ve XAI karar gerekçeleri derleniyor.
                </p>
              </div>
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
