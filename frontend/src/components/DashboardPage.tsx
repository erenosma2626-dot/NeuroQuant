import React, { useState, useMemo } from 'react';
import type { ScreenerItem, UserPortfolio } from '../types';
import { 
  TrendingUp, 
  Zap, 
  ArrowUpRight, 
  ArrowDownRight, 
  BarChart3, 
  SlidersHorizontal,
  Briefcase,
  ChevronRight
} from 'lucide-react';

interface DashboardPageProps {
  screenerData: ScreenerItem[];
  userPortfolio: UserPortfolio;
  onSelectTicker: (ticker: string) => void;
  onNavigateTab: (tab: 'dashboard' | 'terminal' | 'simulation' | 'portfolio') => void;
}

type SortField = 'last_close' | 'change_pct' | 'dist_sma200_pct' | 'alpha_20d_cum' | 'confidence_score';

export const DashboardPage: React.FC<DashboardPageProps> = ({
  screenerData,
  userPortfolio,
  onSelectTicker,
  onNavigateTab,
}) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');
  const [sortField, setSortField] = useState<SortField>('confidence_score');
  const [sortAsc, setSortAsc] = useState<boolean>(false);
  const [tableSearch, setTableSearch] = useState<string>('');

  // Süzme ve Sıralama
  const filteredAndSorted = useMemo(() => {
    let result = [...screenerData];

    if (selectedCategory !== 'ALL') {
      result = result.filter((item) => item.category === selectedCategory);
    }

    if (tableSearch.trim()) {
      const q = tableSearch.toLowerCase();
      result = result.filter(
        (item) =>
          item.ticker.toLowerCase().includes(q) ||
          item.name.toLowerCase().includes(q) ||
          item.sector.toLowerCase().includes(q)
      );
    }

    result.sort((a, b) => {
      const valA = a[sortField];
      const valB = b[sortField];
      return sortAsc ? valA - valB : valB - valA;
    });

    return result;
  }, [screenerData, selectedCategory, sortField, sortAsc, tableSearch]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  // Öne Çıkanlar (Top Highlights)
  const topAlphaAsset = [...screenerData].sort((a, b) => b.alpha_20d_cum - a.alpha_20d_cum)[0];
  const topSmaAsset = [...screenerData].sort((a, b) => b.dist_sma200_pct - a.dist_sma200_pct)[0];
  const topVolumeAsset = [...screenerData].sort((a, b) => b.volume_ratio - a.volume_ratio)[0];

  // Portföy Özeti Hesaplaması
  const totalStockVal = userPortfolio.positions.reduce((acc, p) => acc + p.shares * p.current_price, 0);
  const totalPortfolioVal = userPortfolio.cash + totalStockVal;
  const portfolioReturnPct = ((totalPortfolioVal - userPortfolio.initial_capital) / userPortfolio.initial_capital) * 100;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* 1. Üst Başlık & Açıklama */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <span style={{ fontSize: '0.8rem', color: 'var(--accent-sky)', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
            Kurumsal Gözetim
          </span>
          <h1 style={{ fontSize: '2rem', color: '#FFFFFF', marginTop: 4 }}>
            Piyasa Radarı & Kantitatif Tarayıcı
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginTop: 4, maxWidth: 800 }}>
            Çok faktörlü yapay zeka modelleri, 200 günlük hareketli ortalama mesafeleri ve sektörel alfa ayrışmalarına göre sıralanmış canlı piyasa evreni.
          </p>
        </div>

        {/* Hızlı Portföy Önizleme Kartı */}
        <div 
          onClick={() => onNavigateTab('portfolio')}
          style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            padding: '1rem 1.5rem',
            cursor: 'pointer',
            transition: 'all 0.2s',
            minWidth: 260,
          }}
          className="screener-row"
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <Briefcase size={14} color="var(--accent-sky)" />
              {userPortfolio.name}
            </span>
            <ChevronRight size={14} color="var(--text-muted)" />
          </div>
          <div className="tabular" style={{ fontSize: '1.35rem', fontWeight: 700, color: '#FFFFFF' }}>
            ${totalPortfolioVal.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </div>
          <div style={{ fontSize: '0.78rem', marginTop: 2 }}>
            <span className={portfolioReturnPct >= 0 ? 'tabular' : 'tabular'} style={{ color: portfolioReturnPct >= 0 ? 'var(--bull-text)' : 'var(--bear-text)', fontWeight: 600 }}>
              %{portfolioReturnPct >= 0 ? `+${portfolioReturnPct.toFixed(2)}` : portfolioReturnPct.toFixed(2)}
            </span>
            <span style={{ color: 'var(--text-muted)', marginLeft: 6 }}>
              ({userPortfolio.positions.length} Varlık)
            </span>
          </div>
        </div>
      </div>

      {/* 2. Günün Öne Çıkan Ayrışanları (Highlights) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
        {/* Top Alpha */}
        {topAlphaAsset && (
          <div 
            className="card screener-row" 
            style={{ padding: '1.5rem' }}
            onClick={() => {
              onSelectTicker(topAlphaAsset.ticker);
              onNavigateTab('terminal');
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
              <span className="tag tag-bull">
                <Zap size={13} /> Sektörel Alfa Lideri
              </span>
              <span className="tabular" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                20 Günlük Ayrışma
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
              <div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#FFFFFF' }}>
                  {topAlphaAsset.ticker}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {topAlphaAsset.name}
                </div>
              </div>
              <div className="tabular" style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--bull-text)' }}>
                +%{topAlphaAsset.alpha_20d_cum}
              </div>
            </div>
            <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--border-divider)', display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <span>Fiyat: ${topAlphaAsset.last_close}</span>
              <span style={{ color: 'var(--accent-sky)', fontWeight: 600 }}>Grafiğe Git →</span>
            </div>
          </div>
        )}

        {/* Top 200 SMA Distance */}
        {topSmaAsset && (
          <div 
            className="card screener-row" 
            style={{ padding: '1.5rem' }}
            onClick={() => {
              onSelectTicker(topSmaAsset.ticker);
              onNavigateTab('terminal');
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
              <span className="tag tag-accent">
                <TrendingUp size={13} /> Güçlü Trend Momentumu
              </span>
              <span className="tabular" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                200 SMA Mesafesi
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
              <div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#FFFFFF' }}>
                  {topSmaAsset.ticker}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {topSmaAsset.name}
                </div>
              </div>
              <div className="tabular" style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-sky)' }}>
                +%{topSmaAsset.dist_sma200_pct}
              </div>
            </div>
            <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--border-divider)', display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <span>Sinyal: {topSmaAsset.ai_signal}</span>
              <span style={{ color: 'var(--accent-sky)', fontWeight: 600 }}>Grafiğe Git →</span>
            </div>
          </div>
        )}

        {/* Top Volume Outlier */}
        {topVolumeAsset && (
          <div 
            className="card screener-row" 
            style={{ padding: '1.5rem' }}
            onClick={() => {
              onSelectTicker(topVolumeAsset.ticker);
              onNavigateTab('terminal');
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
              <span className="tag" style={{ background: 'var(--accent-amber-bg)', color: 'var(--accent-amber)', border: '1px solid rgba(245, 158, 11, 0.3)' }}>
                <BarChart3 size={13} /> Para Akışı & Hacim Anomalisi
              </span>
              <span className="tabular" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                20G Ortalamaya Oran
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
              <div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#FFFFFF' }}>
                  {topVolumeAsset.ticker}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {topVolumeAsset.name}
                </div>
              </div>
              <div className="tabular" style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-amber)' }}>
                {topVolumeAsset.volume_ratio}x
              </div>
            </div>
            <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--border-divider)', display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <span>Beta: {topVolumeAsset.beta}</span>
              <span style={{ color: 'var(--accent-sky)', fontWeight: 600 }}>Grafiğe Git →</span>
            </div>
          </div>
        )}
      </div>

      {/* 3. Dinamik Hisse Tarayıcı Tablosu (Screener) */}
      <div className="card" style={{ padding: '2rem' }}>
        <div className="card-header">
          <div className="card-title-group">
            <h2 className="card-title">
              <SlidersHorizontal size={20} color="var(--accent-sky)" />
              Kantitatif Varlık Evreni & Filtreleme
            </h2>
            <p className="card-subtitle">
              Sütun başlıklarına tıklayarak fiyata, değişime, alfaya veya yapay zeka güven skoruna göre sıralayabilirsiniz.
            </p>
          </div>

          {/* Sektör Filtreleri ve Tablo İçi Arama */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <input
              type="text"
              placeholder="Tabloda filtrele..."
              value={tableSearch}
              onChange={(e) => setTableSearch(e.target.value)}
              className="search-input"
              style={{ width: 180, padding: '6px 12px' }}
            />

            <div className="filter-tabs">
              {[
                { id: 'ALL', label: 'Tümü (12)' },
                { id: 'BIST', label: 'BIST 100' },
                { id: 'Tech', label: 'ABD Teknoloji' },
                { id: 'Crypto', label: 'Kripto' },
              ].map((tab) => (
                <button
                  key={tab.id}
                  className={`filter-btn ${selectedCategory === tab.id ? 'active' : ''}`}
                  onClick={() => setSelectedCategory(tab.id)}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Tablo */}
        <div className="screener-table-container">
          <table className="screener-table">
            <thead>
              <tr>
                <th>Varlık / İsim</th>
                <th>Sektör / Kategori</th>
                <th onClick={() => handleSort('last_close')}>
                  Son Fiyat {sortField === 'last_close' && (sortAsc ? '▲' : '▼')}
                </th>
                <th onClick={() => handleSort('change_pct')}>
                  24s Değişim {sortField === 'change_pct' && (sortAsc ? '▲' : '▼')}
                </th>
                <th onClick={() => handleSort('dist_sma200_pct')}>
                  200 SMA Mesafesi {sortField === 'dist_sma200_pct' && (sortAsc ? '▲' : '▼')}
                </th>
                <th onClick={() => handleSort('alpha_20d_cum')}>
                  Sektörel Alfa (20G) {sortField === 'alpha_20d_cum' && (sortAsc ? '▲' : '▼')}
                </th>
                <th>Beta</th>
                <th>Hacim Hızı</th>
                <th onClick={() => handleSort('confidence_score')}>
                  AI Kararı & Güven {sortField === 'confidence_score' && (sortAsc ? '▲' : '▼')}
                </th>
                <th style={{ textAlign: 'right' }}>İşlem</th>
              </tr>
            </thead>
            <tbody>
              {filteredAndSorted.map((item) => {
                const isBull = item.change_pct >= 0;
                return (
                  <tr
                    key={item.ticker}
                    className="screener-row"
                    onClick={() => {
                      onSelectTicker(item.ticker);
                      onNavigateTab('terminal');
                    }}
                  >
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <span className="tabular" style={{ fontWeight: 800, fontSize: '0.95rem', color: '#FFFFFF' }}>
                          {item.ticker}
                        </span>
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                          {item.name}
                        </span>
                      </div>
                    </td>
                    <td>
                      <span className="tag tag-neutral" style={{ fontSize: '0.72rem' }}>
                        {item.sector}
                      </span>
                    </td>
                    <td>
                      <span className="tabular" style={{ fontWeight: 700, fontSize: '0.95rem', color: '#FFFFFF' }}>
                        ${item.last_close.toLocaleString()}
                      </span>
                    </td>
                    <td>
                      <span
                        className={`tag ${isBull ? 'tag-bull' : 'tag-bear'} tabular`}
                        style={{ fontWeight: 700 }}
                      >
                        {isBull ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}
                        {isBull ? `+${item.change_pct}%` : `${item.change_pct}%`}
                      </span>
                    </td>
                    <td>
                      <span
                        className="tabular"
                        style={{
                          fontWeight: 600,
                          color: item.dist_sma200_pct >= 0 ? 'var(--bull-text)' : 'var(--bear-text)',
                        }}
                      >
                        %{item.dist_sma200_pct >= 0 ? `+${item.dist_sma200_pct}` : item.dist_sma200_pct}
                      </span>
                    </td>
                    <td>
                      <span
                        className="tabular"
                        style={{
                          fontWeight: 600,
                          color: item.alpha_20d_cum >= 0 ? 'var(--bull-text)' : 'var(--text-muted)',
                        }}
                      >
                        %{item.alpha_20d_cum >= 0 ? `+${item.alpha_20d_cum}` : item.alpha_20d_cum}
                      </span>
                    </td>
                    <td>
                      <span className="tabular" style={{ color: 'var(--text-secondary)' }}>
                        {item.beta}
                      </span>
                    </td>
                    <td>
                      <span
                        className="tabular"
                        style={{
                          color: item.volume_ratio > 1.2 ? 'var(--accent-amber)' : 'var(--text-secondary)',
                          fontWeight: item.volume_ratio > 1.2 ? 700 : 400,
                        }}
                      >
                        {item.volume_ratio}x
                      </span>
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span
                          className={`tag ${
                            item.ai_signal.includes('AL')
                              ? 'tag-bull'
                              : item.ai_signal.includes('AZALT')
                              ? 'tag-bear'
                              : 'tag-neutral'
                          }`}
                          style={{ fontWeight: 700 }}
                        >
                          {item.ai_signal}
                        </span>
                        <span className="tabular" style={{ fontSize: '0.78rem', color: 'var(--accent-sky)' }}>
                          %{item.confidence_score}
                        </span>
                      </div>
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', gap: 6 }}>
                        <button
                          className="btn-secondary"
                          style={{ padding: '4px 10px', fontSize: '0.72rem' }}
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectTicker(item.ticker);
                            onNavigateTab('terminal');
                          }}
                        >
                          Terminal
                        </button>
                        <button
                          className="btn-primary"
                          style={{ padding: '4px 10px', fontSize: '0.72rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-hover)' }}
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectTicker(item.ticker);
                            onNavigateTab('simulation');
                          }}
                        >
                          10k Lab
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
