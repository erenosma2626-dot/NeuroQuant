import React, { useState, useRef, useEffect } from 'react';
import { Search, ArrowRight, TrendingUp } from 'lucide-react';
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

// Popüler BIST ve Küresel Örnekler
const POPULAR_SUGGESTIONS = [
  { ticker: 'NVDA', name: 'Nvidia Corp.', category: 'Tech' },
  { ticker: 'AAPL', name: 'Apple Inc.', category: 'Tech' },
  { ticker: 'MSFT', name: 'Microsoft Corp.', category: 'Tech' },
  { ticker: 'TSLA', name: 'Tesla Inc.', category: 'Auto' },
  { ticker: 'BTC-USD', name: 'Bitcoin (USD)', category: 'Crypto' },
  { ticker: 'ETH-USD', name: 'Ethereum (USD)', category: 'Crypto' },
  { ticker: 'THYAO.IS', name: 'Türk Hava Yolları', category: 'BIST' },
  { ticker: 'ASELS.IS', name: 'Aselsan', category: 'BIST' },
  { ticker: 'GARAN.IS', name: 'Garanti BBVA', category: 'BIST' },
  { ticker: 'TUPRS.IS', name: 'Tüpraş Petrol', category: 'BIST' },
  { ticker: 'EREGL.IS', name: 'Ereğli Demir Çelik', category: 'BIST' },
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

  // BIST akıllı önerisi: Eğer kullanıcı .IS yazmadıysa ve harflerden oluşuyorsa
  const showBistSuggestion =
    cleanQuery.length >= 3 &&
    !cleanQuery.includes('.') &&
    !cleanQuery.includes('-') &&
    !cleanQuery.endsWith('.IS');

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
    onSelectTab('terminal');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && cleanQuery) {
      // Eğer kullanıcı doğrudan Enter'a bastıysa o sembolü ara
      handleSelect(cleanQuery);
    } else if (e.key === 'Escape') {
      setDropOpen(false);
    }
  };

  const currentAsset = universe.find((u) => u.ticker.toUpperCase() === currentTicker.toUpperCase());

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

        {/* Right: Search + Ticker + Status */}
        <div className="masthead-right">

          {/* Omnisearch */}
          <div className="search-wrapper" ref={searchRef}>
            <Search size={13} className="search-icon" />
            <input
              type="text"
              className="search-input"
              placeholder="Herhangi bir hisse ara (NVDA, AAPL, GARAN, BTC)..."
              value={query}
              onChange={(e) => { setQuery(e.target.value); setDropOpen(true); }}
              onFocus={() => setDropOpen(true)}
              onKeyDown={handleKeyDown}
            />

            {dropOpen && (
              <div className="search-dropdown" style={{ minWidth: 320 }}>
                {/* 1. Doğrudan Ticker Sorgulama Eylemi */}
                {cleanQuery && !hasExactMatch && (
                  <div
                    className="search-result-item"
                    onClick={() => handleSelect(cleanQuery)}
                    style={{
                      background: 'rgba(20, 83, 45, 0.06)',
                      borderBottom: '1px solid var(--rule-light)',
                      padding: '0.6rem 0.85rem'
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

                {/* 2. BIST Akıllı Tamamlama Önerisi (.IS) */}
                {showBistSuggestion && (
                  <div
                    className="search-result-item"
                    onClick={() => handleSelect(`${cleanQuery}.IS`)}
                    style={{
                      borderBottom: '1px solid var(--rule-light)',
                      padding: '0.55rem 0.85rem'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <TrendingUp size={13} style={{ color: 'var(--cobalt)' }} />
                      <div>
                        <div style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--ink-primary)' }}>
                          {cleanQuery}.IS
                        </div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--ink-muted)' }}>
                          Borsa İstanbul (BIST) Pay Piyasası
                        </div>
                      </div>
                    </div>
                    <span style={{ fontSize: '0.68rem', fontWeight: 600, color: 'var(--cobalt)', textTransform: 'uppercase' }}>
                      BIST
                    </span>
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
                      padding: '0.4rem 0.85rem',
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
                        style={{ padding: '0.45rem 0.85rem' }}
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

          {/* Active Ticker Capsule (Always visible) */}
          <div
            className="ticker-capsule"
            onClick={() => onSelectTab('terminal')}
            title="Terminale git"
          >
            <span className="tabular" style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--ink-primary)' }}>
              {currentTicker}
            </span>
            <span style={{ width: 1, height: 14, background: 'var(--rule-strong)' }} />
            {currentAsset ? (
              <>
                <span className="tabular" style={{ color: 'var(--ink-secondary)', fontSize: '0.82rem' }}>
                  {currentAsset.last_close.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
                <span className={currentAsset.change_pct >= 0 ? 'change-up' : 'change-down'} style={{ fontSize: '0.78rem' }}>
                  {currentAsset.change_pct >= 0 ? '+' : ''}{currentAsset.change_pct.toFixed(2)}%
                </span>
              </>
            ) : (
              <span style={{ fontSize: '0.75rem', color: 'var(--cobalt)', fontStyle: 'italic' }}>
                Aktif Analiz
              </span>
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
              className={`masthead-nav-item ${isActive ? 'active' : ''}`}
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
