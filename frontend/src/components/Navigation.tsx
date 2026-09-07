import React, { useState, useRef, useEffect } from 'react';
import { 
  LayoutDashboard, 
  CandlestickChart, 
  FlaskConical, 
  Briefcase, 
  Search, 
  ShieldCheck,
  ArrowRight
} from 'lucide-react';
import type { ScreenerItem } from '../types';

interface NavigationProps {
  currentTicker: string;
  onSelectTicker: (ticker: string) => void;
  activeTab: 'dashboard' | 'terminal' | 'simulation' | 'portfolio';
  onSelectTab: (tab: 'dashboard' | 'terminal' | 'simulation' | 'portfolio') => void;
  universe: ScreenerItem[];
}

export const Navigation: React.FC<NavigationProps> = ({
  currentTicker,
  onSelectTicker,
  activeTab,
  onSelectTab,
  universe,
}) => {
  const [query, setQuery] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  // Filtered assets for dropdown
  const filtered = query.trim()
    ? universe.filter(
        (u) =>
          u.ticker.toLowerCase().includes(query.toLowerCase()) ||
          u.name.toLowerCase().includes(query.toLowerCase()) ||
          u.sector.toLowerCase().includes(query.toLowerCase())
      )
    : [];

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelectAsset = (ticker: string) => {
    onSelectTicker(ticker);
    setQuery('');
    setIsDropdownOpen(false);
    // If on dashboard, seamlessly navigate to terminal for details
    if (activeTab === 'dashboard') {
      onSelectTab('terminal');
    }
  };

  const currentAsset = universe.find((u) => u.ticker === currentTicker);

  return (
    <header className="top-nav">
      {/* Brand Section */}
      <div className="brand-section" onClick={() => onSelectTab('dashboard')}>
        <div className="brand-logo-icon">
          <CandlestickChart size={20} color="#FFFFFF" strokeWidth={2.2} />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="brand-title">NeuroQuant</span>
          <span className="brand-badge">Sovereign 3.0</span>
        </div>
      </div>

      {/* Center: Main Page Tabs */}
      <nav className="nav-tabs-group">
        <button
          className={`nav-tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => onSelectTab('dashboard')}
        >
          <LayoutDashboard size={16} />
          Piyasa Radarı & Dashboard
        </button>

        <button
          className={`nav-tab-btn ${activeTab === 'terminal' ? 'active' : ''}`}
          onClick={() => onSelectTab('terminal')}
        >
          <CandlestickChart size={16} />
          Kantitatif Terminal
        </button>

        <button
          className={`nav-tab-btn ${activeTab === 'simulation' ? 'active' : ''}`}
          onClick={() => onSelectTab('simulation')}
        >
          <FlaskConical size={16} />
          10k Simülasyon Lab
        </button>

        <button
          className={`nav-tab-btn ${activeTab === 'portfolio' ? 'active' : ''}`}
          onClick={() => onSelectTab('portfolio')}
        >
          <Briefcase size={16} />
          Portföy Yöneticisi
        </button>
      </nav>

      {/* Right: Omnisearch & Active Ticker */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
        {/* Omnisearch */}
        <div className="search-wrapper" ref={searchRef}>
          <Search size={15} className="search-icon" />
          <input
            type="text"
            className="search-input"
            placeholder="Varlık veya Sektör Ara (NVDA, THYAO)..."
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setIsDropdownOpen(true);
            }}
            onFocus={() => setIsDropdownOpen(true)}
          />

          {isDropdownOpen && filtered.length > 0 && (
            <div className="search-dropdown">
              {filtered.map((item) => (
                <div
                  key={item.ticker}
                  className="search-result-item"
                  onClick={() => handleSelectAsset(item.ticker)}
                >
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span className="tabular" style={{ fontWeight: 700, color: '#F8FAFC' }}>
                        {item.ticker}
                      </span>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                        {item.category}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      {item.name}
                    </div>
                  </div>

                  <div style={{ textAlign: 'right' }}>
                    <div className="tabular" style={{ fontWeight: 600, color: '#F8FAFC' }}>
                      ${item.last_close.toLocaleString()}
                    </div>
                    <div
                      className={`tag ${item.change_pct >= 0 ? 'tag-bull' : 'tag-bear'}`}
                      style={{ fontSize: '0.68rem', padding: '1px 5px' }}
                    >
                      {item.change_pct >= 0 ? `+${item.change_pct}%` : `${item.change_pct}%`}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Current Active Ticker Capsule */}
        {currentAsset && (
          <div
            onClick={() => onSelectTab('terminal')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '6px 14px',
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              cursor: 'pointer',
              transition: 'border-color 0.15s',
            }}
            title="Detaylı Grafiğe Git"
          >
            <span className="tabular" style={{ fontWeight: 700, color: '#FFFFFF', fontSize: '0.85rem' }}>
              {currentAsset.ticker}
            </span>
            <span className="tabular" style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
              ${currentAsset.last_close.toLocaleString()}
            </span>
            <span
              className={`tag ${currentAsset.change_pct >= 0 ? 'tag-bull' : 'tag-bear'}`}
              style={{ fontSize: '0.7rem', padding: '1px 6px' }}
            >
              {currentAsset.change_pct >= 0 ? `+${currentAsset.change_pct}%` : `${currentAsset.change_pct}%`}
            </span>
            <ArrowRight size={13} color="var(--text-muted)" />
          </div>
        )}

        {/* System Online Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.75rem', color: 'var(--bull-text)' }}>
          <ShieldCheck size={16} />
          <span className="tabular" style={{ fontWeight: 600, letterSpacing: '0.04em' }}>ONLINE</span>
        </div>
      </div>
    </header>
  );
};
