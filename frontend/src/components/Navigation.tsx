import React, { useState, useRef, useEffect } from 'react';
import { Search, ArrowRight, X } from 'lucide-react';
import type { ScreenerItem, MarketData } from '../types';

interface NavigationProps {
  currentTicker: string;
  onSelectTicker: (ticker: string) => void;
  activeTab: 'dashboard' | 'terminal' | 'simulation' | 'portfolio';
  onSelectTab: (tab: 'dashboard' | 'terminal' | 'simulation' | 'portfolio') => void;
  universe: ScreenerItem[];
  activeMarketData?: MarketData | null;
}

const TABS: { id: 'dashboard' | 'terminal' | 'simulation' | 'portfolio'; label: string }[] = [
  { id: 'dashboard',  label: 'Piyasa Tarayıcısı'   },
  { id: 'terminal',   label: 'Kantitatif Terminal'  },
  { id: 'simulation', label: '10k Simülasyon Lab'   },
  { id: 'portfolio',  label: 'Portföy Atölyesi'     },
];

// Popüler Wall Street ve Kripto Varlıkları
const POPULAR_SUGGESTIONS = [
  { ticker: 'NVDA', name: 'Nvidia Corp. (AI Leader)', category: 'Tech' },
  { ticker: 'AAPL', name: 'Apple Inc.', category: 'Tech' },
  { ticker: 'MSFT', name: 'Microsoft Corp.', category: 'Tech' },
  { ticker: 'AMZN', name: 'Amazon.com Inc.', category: 'Tech' },
  { ticker: 'GOOGL', name: 'Alphabet Inc.', category: 'Tech' },
  { ticker: 'META', name: 'Meta Platforms Inc.', category: 'Tech' },
  { ticker: 'TSLA', name: 'Tesla Inc.', category: 'Tech' },
  { ticker: 'AMD', name: 'Advanced Micro Devices', category: 'Semis' },
  { ticker: 'AVGO', name: 'Broadcom Inc.', category: 'Semis' },
  { ticker: 'JPM', name: 'JPMorgan Chase', category: 'Finance' },
  { ticker: 'BTC-USD', name: 'Bitcoin (USD)', category: 'Crypto' },
  { ticker: 'ETH-USD', name: 'Ethereum (USD)', category: 'Crypto' },
];

export const Navigation: React.FC<NavigationProps> = ({
  currentTicker,
  onSelectTicker,
  activeTab,
  onSelectTab,
  universe,
  activeMarketData,
}) => {
  const [query, setQuery]                 = useState('');
  const [dropOpen, setDropOpen]           = useState(false);
  const [isFocused, setIsFocused]         = useState(false);
  const [hasRecentSwap, setHasRecentSwap] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);
  const inputRef  = useRef<HTMLInputElement>(null);
  const prevTickerRef = useRef(currentTicker);

  // Sembol değiştiğinde yumuşak parıltı animasyonu tetikle
  useEffect(() => {
    if (prevTickerRef.current !== currentTicker) {
      prevTickerRef.current = currentTicker;
      setHasRecentSwap(true);
      const timer = setTimeout(() => setHasRecentSwap(false), 900);
      return () => clearTimeout(timer);
    }
  }, [currentTicker]);

  const cleanQuery = query.trim().toUpperCase();

  // Screener evrenindeki eşleşmeler
  const filtered = cleanQuery
    ? universe.filter(
        (u) =>
          u.ticker.toLowerCase().includes(query.toLowerCase()) ||
          u.name.toLowerCase().includes(query.toLowerCase()) ||
          u.sector.toLowerCase().includes(query.toLowerCase())
      ).slice(0, 8)
    : [];

  // Exact match var mı?
  const hasExactMatch = filtered.some((u) => u.ticker.toUpperCase() === cleanQuery);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setDropOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleSelect = (ticker: string) => {
    const target = ticker.trim().toUpperCase();
    if (!target) return;
    onSelectTicker(target);
    setQuery('');
    setDropOpen(false);
    // Eğer kullanıcı simülasyon veya portföy sayfasındaysa o sayfada kalsın, dashboarddaysa terminale yönlendirilsin
    if (activeTab === 'dashboard') {
      onSelectTab('terminal');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && cleanQuery) {
      handleSelect(cleanQuery);
    } else if (e.key === 'Escape') {
      setDropOpen(false);
    }
  };

  // Aktif varlığın fiyat ve değişim bilgisi (öncelik live marketData)
  const currentFromUniverse = universe.find((u) => u.ticker.toUpperCase() === currentTicker.toUpperCase());
  const displayAsset = (activeMarketData && activeMarketData.ticker.toUpperCase() === currentTicker.toUpperCase())
    ? {
        last_close: activeMarketData.current_price,
        change_pct: activeMarketData.change_pct,
      }
    : currentFromUniverse
      ? {
          last_close: currentFromUniverse.last_close,
          change_pct: currentFromUniverse.change_pct,
        }
      : null;

  // Güncel İstanbul Tarihi
  const now = new Date();
  const timeStr = now.toLocaleString('tr-TR', {
    timeZone: 'Europe/Istanbul',
    weekday: 'long',
    day: '2-digit',
    month: 'long',
    year: 'numeric',
  });

  return (
    <header className="masthead">
      {/* ── Top Bar: Brand + Date + Search + Status ── */}
      <div className="masthead-top">

        {/* Brand / Logotype */}
        <div className="brand-section" onClick={() => onSelectTab('dashboard')} style={{ cursor: 'pointer' }}>
          <div className="brand-logotype">NeuroQuant</div>
          <div className="brand-tagline">Google TimesFM 3.0 &amp; Çok-Faktörlü Quant Motoru · Est. 2024</div>
        </div>

        {/* Date line — newspaper style */}
        <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
          <div style={{
            fontFamily: 'var(--font-body)',
            fontSize: '0.68rem',
            fontWeight: 600,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            color: 'var(--ink-muted)',
          }}>
            {timeStr}
          </div>
          <div style={{
            fontFamily: 'var(--font-display)',
            fontSize: '0.75rem',
            fontStyle: 'italic',
            color: 'var(--ink-secondary)',
          }}>
            Gerçek Zamanlı Piyasa Verisi &amp; Sıfır-Atış AI Kestirimleri
          </div>
        </div>

        {/* Right: Unified Search & Ticker Capsule Bar + Status */}
        <div className="masthead-right">

          {/* Unified Command & Ticker Capsule Bar */}
          <div
            className={`unified-command-bar ${isFocused ? 'is-focused' : ''} ${hasRecentSwap ? 'has-swap' : ''}`}
            ref={searchRef}
          >
            {/* Search Input Section */}
            <div className="unified-search-section">
              <Search size={14} className="unified-search-icon" />
              <input
                ref={inputRef}
                type="text"
                className="unified-search-input"
                placeholder="Herhangi bir hisse ara (NVDA, AAPL, GARAN, BTC)..."
                value={query}
                onChange={(e) => { setQuery(e.target.value); setDropOpen(true); }}
                onFocus={() => { setIsFocused(true); setDropOpen(true); }}
                onBlur={() => setIsFocused(false)}
                onKeyDown={handleKeyDown}
              />
              {query && (
                <button
                  type="button"
                  className="unified-clear-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    setQuery('');
                    inputRef.current?.focus();
                  }}
                  title="Aramayı temizle"
                >
                  <X size={12} />
                </button>
              )}
            </div>

            {/* Seamless Hairline Divider */}
            <div className="unified-divider" />

            {/* Active Ticker Capsule (Integrated right inside) */}
            <div
              className={`unified-ticker-chip ${activeTab === 'terminal' ? 'is-terminal-active' : ''}`}
              onClick={() => onSelectTab('terminal')}
              title="Terminale Git (Aktif Varlık)"
            >
              <div key={currentTicker} className="unified-ticker-inner animate-ticker-swap">
                <span className="unified-ticker-symbol tabular">{currentTicker}</span>
                <span className="unified-ticker-sep">|</span>
                {displayAsset ? (
                  <>
                    <span className="unified-ticker-price tabular">
                      {displayAsset.last_close.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                    <span
                      className={`unified-ticker-change tabular ${displayAsset.change_pct >= 0 ? 'is-up' : 'is-down'}`}
                    >
                      {displayAsset.change_pct >= 0 ? '+' : ''}{displayAsset.change_pct.toFixed(2)}%
                    </span>
                  </>
                ) : (
                  <span className="unified-ticker-pending">Analiz Ediliyor…</span>
                )}
              </div>
            </div>

            {/* Dropdown Menu (Anchored under the unified command bar) */}
            {dropOpen && (
              <div className="search-dropdown unified-search-dropdown">
                {/* 1. Doğrudan Ticker Sorgulama Eylemi */}
                {cleanQuery && !hasExactMatch && (
                  <div
                    className="search-result-item"
                    onClick={() => handleSelect(cleanQuery)}
                    style={{
                      background: 'rgba(20, 83, 45, 0.06)',
                      borderBottom: '1px solid var(--rule-light)',
                      padding: '0.65rem 0.95rem'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <Search size={14} style={{ color: 'var(--forest-gain)' }} />
                      <div>
                        <div style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--ink-primary)' }}>
                          "{cleanQuery}" Sembolünü Analiz Et
                        </div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--ink-muted)' }}>
                          Yahoo Finance üzerinden anında çek (Enter)
                        </div>
                      </div>
                    </div>
                    <ArrowRight size={13} style={{ color: 'var(--forest-gain)' }} />
                  </div>
                )}

                {/* 3. Filtrelenmiş Screener Hisseleri */}
                {filtered.map((item) => (
                  <div
                    key={item.ticker}
                    className="search-result-item"
                    onClick={() => handleSelect(item.ticker)}
                  >
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span className="tabular" style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--ink-primary)' }}>
                          {item.ticker}
                        </span>
                        <span style={{ fontSize: '0.7rem', color: 'var(--ink-muted)', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                          {item.category}
                        </span>
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--ink-secondary)', marginTop: 1 }}>
                        {item.name}
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div className="tabular" style={{ fontWeight: 600, fontSize: '0.82rem', color: 'var(--ink-primary)' }}>
                        {item.last_close.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </div>
                      <div className={item.change_pct >= 0 ? 'change-up' : 'change-down'} style={{ fontSize: '0.75rem' }}>
                        {item.change_pct >= 0 ? '+' : ''}{item.change_pct.toFixed(2)}%
                      </div>
                    </div>
                  </div>
                ))}

                {/* 4. Arama boşken popüler öneriler */}
                {!cleanQuery && (
                  <div>
                    <div style={{
                      padding: '0.4rem 0.95rem',
                      fontSize: '0.62rem',
                      fontWeight: 700,
                      letterSpacing: '0.1em',
                      textTransform: 'uppercase',
                      color: 'var(--ink-muted)',
                      borderBottom: '1px solid var(--rule-light)',
                    }}>
                      Hızlı Varlık Seçimi
                    </div>
                    {POPULAR_SUGGESTIONS.map((s) => (
                      <div
                        key={s.ticker}
                        className="search-result-item"
                        onClick={() => handleSelect(s.ticker)}
                        style={{ padding: '0.5rem 0.95rem' }}
                      >
                        <div>
                          <span className="tabular" style={{ fontWeight: 700, fontSize: '0.82rem', color: 'var(--ink-primary)' }}>
                            {s.ticker}
                          </span>
                          <span style={{ fontSize: '0.74rem', color: 'var(--ink-secondary)', marginLeft: 8 }}>
                            {s.name}
                          </span>
                        </div>
                        <span style={{ fontSize: '0.65rem', color: 'var(--ink-muted)', textTransform: 'uppercase' }}>
                          {s.category}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* System Status */}
          <div className="status-pill">
            <span className="status-dot" />
            CANLI
          </div>
        </div>
      </div>

      {/* ── Section Navigation Tabs ── */}
      <nav className="masthead-nav">
        {TABS.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              className={`nav-tab ${isActive ? 'active' : ''}`}
              onClick={() => onSelectTab(tab.id)}
            >
              {tab.label}
              {isActive && <span className="nav-ink-underline" />}
            </button>
          );
        })}
      </nav>
    </header>
  );
};
