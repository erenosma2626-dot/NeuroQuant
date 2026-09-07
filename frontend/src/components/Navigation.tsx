import React, { useState, useRef, useEffect } from 'react';
import { Search } from 'lucide-react';
import type { ScreenerItem } from '../types';

interface NavigationProps {
  currentTicker: string;
  onSelectTicker: (ticker: string) => void;
  activeTab: 'dashboard' | 'terminal' | 'simulation' | 'portfolio';
  onSelectTab: (tab: 'dashboard' | 'terminal' | 'simulation' | 'portfolio') => void;
  universe: ScreenerItem[];
}

const TABS: { id: 'dashboard' | 'terminal' | 'simulation' | 'portfolio'; label: string }[] = [
  { id: 'dashboard',  label: 'Piyasa Tarayıcısı'   },
  { id: 'terminal',   label: 'Kantitatif Terminal'  },
  { id: 'simulation', label: '10k Simülasyon Lab'   },
  { id: 'portfolio',  label: 'Portföy Atölyesi'     },
];

export const Navigation: React.FC<NavigationProps> = ({
  currentTicker,
  onSelectTicker,
  activeTab,
  onSelectTab,
  universe,
}) => {
  const [query, setQuery]         = useState('');
  const [dropOpen, setDropOpen]   = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  const filtered = query.trim()
    ? universe.filter(
        (u) =>
          u.ticker.toLowerCase().includes(query.toLowerCase()) ||
          u.name.toLowerCase().includes(query.toLowerCase()) ||
          u.sector.toLowerCase().includes(query.toLowerCase())
      ).slice(0, 8)
    : [];

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
    onSelectTicker(ticker);
    setQuery('');
    setDropOpen(false);
    if (activeTab === 'dashboard') onSelectTab('terminal');
  };

  const currentAsset = universe.find((u) => u.ticker === currentTicker);

  // Get current time in Istanbul
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
        <div className="brand-section" onClick={() => onSelectTab('dashboard')}>
          <div className="brand-logotype">NeuroQuant</div>
          <div className="brand-tagline">Sovereign Quantitative Intelligence · Est. 2024</div>
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
            Gerçek Zamanlı Piyasa Verisi
          </div>
        </div>

        {/* Right: Search + Ticker + Status */}
        <div className="masthead-right">

          {/* Omnisearch */}
          <div className="search-wrapper" ref={searchRef}>
            <Search size={13} className="search-icon" />
            <input
              type="text"
              className="search-input"
              placeholder="Varlık ara — NVDA, BTC, THYAO..."
              value={query}
              onChange={(e) => { setQuery(e.target.value); setDropOpen(true); }}
              onFocus={() => setDropOpen(true)}
            />
            {dropOpen && filtered.length > 0 && (
              <div className="search-dropdown">
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
              </div>
            )}
          </div>

          {/* Active Ticker Capsule */}
          {currentAsset && (
            <div
              className="ticker-capsule"
              onClick={() => onSelectTab('terminal')}
              title="Terminale git"
            >
              <span className="tabular" style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--ink-primary)' }}>
                {currentAsset.ticker}
              </span>
              <span style={{ width: 1, height: 14, background: 'var(--rule-strong)' }} />
              <span className="tabular" style={{ color: 'var(--ink-secondary)', fontSize: '0.82rem' }}>
                {currentAsset.last_close.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
              <span className={currentAsset.change_pct >= 0 ? 'change-up' : 'change-down'} style={{ fontSize: '0.78rem' }}>
                {currentAsset.change_pct >= 0 ? '+' : ''}{currentAsset.change_pct.toFixed(2)}%
              </span>
            </div>
          )}

          {/* System Status */}
          <div className="status-pill">
            <span className="status-dot" />
            CANLI
          </div>
        </div>
      </div>

      {/* ── Section Navigation Tabs ── */}
      <nav className="masthead-nav">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => onSelectTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
    </header>
  );
};
