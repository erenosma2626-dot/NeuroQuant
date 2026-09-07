import React, { useState, useMemo } from 'react';
import type { ScreenerItem, UserPortfolio } from '../types';

interface DashboardPageProps {
  screenerData: ScreenerItem[];
  userPortfolio: UserPortfolio;
  onSelectTicker: (ticker: string) => void;
  onNavigateTab: (tab: 'dashboard' | 'terminal' | 'simulation' | 'portfolio') => void;
}

type SortField = 'last_close' | 'change_pct' | 'dist_sma200_pct' | 'alpha_20d_cum' | 'confidence_score';

const CATEGORIES = [
  { id: 'ALL',    label: 'Tüm Evren' },
  { id: 'BIST',   label: 'BIST 100'  },
  { id: 'Tech',   label: 'ABD Tekno' },
  { id: 'Crypto', label: 'Kripto'    },
];

function signalClass(signal: string): string {
  if (signal.includes('AL')) return 'signal-buy';
  if (signal.includes('SAT') || signal.includes('AZALT')) return 'signal-sell';
  return 'signal-neutral';
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  screenerData,
  userPortfolio,
  onSelectTicker,
  onNavigateTab,
}) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');
  const [sortField, setSortField]               = useState<SortField>('confidence_score');
  const [sortAsc, setSortAsc]                   = useState<boolean>(false);
  const [tableSearch, setTableSearch]           = useState<string>('');

  /* ── Filter + Sort ───────────────────────────────────────────── */
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
    if (sortField === field) setSortAsc(!sortAsc);
    else { setSortField(field); setSortAsc(false); }
  };

  /* ── Highlights ──────────────────────────────────────────────── */
  const topAlpha  = [...screenerData].sort((a, b) => b.alpha_20d_cum - a.alpha_20d_cum)[0];
  const topSma    = [...screenerData].sort((a, b) => b.dist_sma200_pct - a.dist_sma200_pct)[0];
  const topVolume = [...screenerData].sort((a, b) => b.volume_ratio - a.volume_ratio)[0];

  /* ── Portfolio Stats ─────────────────────────────────────────── */
  const stockVal     = userPortfolio.positions.reduce((s, p) => s + p.shares * p.current_price, 0);
  const totalVal     = userPortfolio.cash + stockVal;
  const returnPct    = ((totalVal - userPortfolio.initial_capital) / userPortfolio.initial_capital) * 100;
  const returnPositive = returnPct >= 0;

  /* ── Sort Indicator ──────────────────────────────────────────── */
  const sortArrow = (field: SortField) => sortField === field ? (sortAsc ? ' ▲' : ' ▼') : '';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem', animation: 'fadeUp 0.35s ease' }}>

      {/* ══════════════════════════════════════════════════════════
          SECTION 1 — PORTFOLIO OVERVIEW BAR
         ══════════════════════════════════════════════════════════ */}
      <div
        className="panel"
        style={{ borderTop: '3px solid var(--ink-primary)' }}
      >
        {/* Panel header */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '1rem 2rem',
          borderBottom: '2px solid var(--ink-primary)',
        }}>
          <div>
            <div className="section-label">Portföy Durumu</div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', fontWeight: 700, color: 'var(--ink-primary)' }}>
              {userPortfolio.name}
            </div>
          </div>
          <button
            className="btn btn-secondary"
            onClick={() => onNavigateTab('portfolio')}
            style={{ fontSize: '0.78rem' }}
          >
            Portföy Atölyesi →
          </button>
        </div>

        {/* Metrics row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)' }}>
          <div className="metric-block">
            <div className="metric-label">Toplam Değer</div>
            <div className="metric-value tabular">
              {totalVal.toLocaleString('tr-TR', { minimumFractionDigits: 0, maximumFractionDigits: 0 })} ₺
            </div>
            <div className="metric-sub">Nakit + Hisse</div>
          </div>
          <div className="metric-block">
            <div className="metric-label">Net Getiri</div>
            <div className="metric-value tabular" style={{ color: returnPositive ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
              {returnPositive ? '+' : ''}{returnPct.toFixed(2)}%
            </div>
            <div className="metric-sub">Başlangıçtan beri</div>
          </div>
          <div className="metric-block">
            <div className="metric-label">Nakit Rezervi</div>
            <div className="metric-value tabular">
              {userPortfolio.cash.toLocaleString('tr-TR')} ₺
            </div>
            <div className="metric-sub">Kullanılabilir</div>
          </div>
          <div className="metric-block">
            <div className="metric-label">Pozisyon Sayısı</div>
            <div className="metric-value tabular">{userPortfolio.positions.length}</div>
            <div className="metric-sub">Aktif varlık</div>
          </div>
          <div className="metric-block">
            <div className="metric-label">Hisse Değeri</div>
            <div className="metric-value tabular">
              {stockVal.toLocaleString('tr-TR', { minimumFractionDigits: 0, maximumFractionDigits: 0 })} ₺
            </div>
            <div className="metric-sub">Piyasa değeri</div>
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════
          SECTION 2 — LEAD STORY: THREE COLUMN HIGHLIGHTS
         ══════════════════════════════════════════════════════════ */}
      <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)' }}>
        {/* Masthead row */}
        <div style={{
          padding: '0.75rem 2rem',
          borderBottom: '1px solid var(--rule-strong)',
          display: 'flex',
          alignItems: 'baseline',
          justifyContent: 'space-between',
        }}>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', fontWeight: 700, fontStyle: 'italic', color: 'var(--ink-primary)' }}>
            Günün Ayrışan Hareketleri
          </div>
          <div style={{ fontSize: '0.65rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)' }}>
            Çok-Faktörlü Kantitatif Tarama
          </div>
        </div>

        {/* Three-column editorial grid */}
        <div className="lead-grid">
          {/* Alpha Leader */}
          {topAlpha && (
            <div
              className="lead-cell"
              onClick={() => { onSelectTicker(topAlpha.ticker); onNavigateTab('terminal'); }}
            >
              <div className="lead-rank">Sektörel Alfa Lideri — 20G Kümülatif</div>
              <div className="lead-headline">{topAlpha.ticker}</div>
              <div style={{ marginBottom: 8 }}>
                <span style={{ fontFamily: 'var(--font-display)', fontSize: '1.8rem', fontWeight: 700, color: 'var(--forest-gain)' }}>
                  +{topAlpha.alpha_20d_cum}%
                </span>
              </div>
              <div className="lead-sub">{topAlpha.name}</div>
              <div style={{ marginTop: 10, display: 'flex', gap: 8, alignItems: 'center' }}>
                <span className="signal signal-buy">{topAlpha.ai_signal}</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--ink-muted)' }}>Güven: %{topAlpha.confidence_score}</span>
              </div>
            </div>
          )}

          {/* 200 SMA Trend Leader */}
          {topSma && (
            <div
              className="lead-cell"
              onClick={() => { onSelectTicker(topSma.ticker); onNavigateTab('terminal'); }}
            >
              <div className="lead-rank">200 Günlük Ortalama — En Güçlü Trend</div>
              <div className="lead-headline">{topSma.ticker}</div>
              <div style={{ marginBottom: 8 }}>
                <span style={{ fontFamily: 'var(--font-display)', fontSize: '1.8rem', fontWeight: 700, color: 'var(--cobalt)' }}>
                  +{topSma.dist_sma200_pct}%
                </span>
              </div>
              <div className="lead-sub">{topSma.name}</div>
              <div style={{ marginTop: 10, display: 'flex', gap: 8, alignItems: 'center' }}>
                <span className="signal signal-cobalt">200 SMA</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--ink-muted)' }}>
                  Fiyat: {topSma.last_close.toLocaleString('tr-TR', { maximumFractionDigits: 2 })}
                </span>
              </div>
            </div>
          )}

          {/* Volume Anomaly */}
          {topVolume && (
            <div
              className="lead-cell"
              onClick={() => { onSelectTicker(topVolume.ticker); onNavigateTab('terminal'); }}
            >
              <div className="lead-rank">Hacim Anomalisi — Para Akışı Yoğunluğu</div>
              <div className="lead-headline">{topVolume.ticker}</div>
              <div style={{ marginBottom: 8 }}>
                <span style={{ fontFamily: 'var(--font-display)', fontSize: '1.8rem', fontWeight: 700, color: 'var(--amber-warm)' }}>
                  {topVolume.volume_ratio}×
                </span>
              </div>
              <div className="lead-sub">{topVolume.name}</div>
              <div style={{ marginTop: 10, display: 'flex', gap: 8, alignItems: 'center' }}>
                <span className="signal signal-neutral">Beta {topVolume.beta}</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--ink-muted)' }}>20G Hacim Ortalaması</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════
          SECTION 3 — THE FINANCIAL COMPASS: SCREENER TABLE
         ══════════════════════════════════════════════════════════ */}
      <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)' }}>

        {/* Table Header */}
        <div style={{
          padding: '1rem 2rem',
          borderBottom: '2px solid var(--ink-primary)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '1rem',
          flexWrap: 'wrap',
        }}>
          <div>
            <div style={{
              fontFamily: 'var(--font-display)',
              fontSize: '1rem',
              fontWeight: 700,
              fontStyle: 'italic',
              color: 'var(--ink-primary)',
            }}>
              Kantitatif Varlık Evreni
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--ink-secondary)', marginTop: 2 }}>
              Yapay zeka sinyalleri, 200 SMA, alfa ve güven skoruna göre sıralanabilir — sütun başlığına tıklayın
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <input
              type="text"
              className="table-search"
              placeholder="Tablo içi filtrele..."
              value={tableSearch}
              onChange={(e) => setTableSearch(e.target.value)}
            />
            <div className="filter-bar">
              {CATEGORIES.map((cat) => (
                <button
                  key={cat.id}
                  className={`filter-btn ${selectedCategory === cat.id ? 'active' : ''}`}
                  onClick={() => setSelectedCategory(cat.id)}
                >
                  {cat.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Screener Table */}
        <div className="screener-wrap">
          <table className="screener-table">
            <thead>
              <tr>
                <th style={{ paddingLeft: '2rem' }}>Varlık</th>
                <th>Kategori</th>
                <th onClick={() => handleSort('last_close')} style={{ cursor: 'pointer' }}>
                  Son Fiyat{sortArrow('last_close')}
                </th>
                <th onClick={() => handleSort('change_pct')} style={{ cursor: 'pointer' }}>
                  24s Değişim{sortArrow('change_pct')}
                </th>
                <th onClick={() => handleSort('dist_sma200_pct')} style={{ cursor: 'pointer' }}>
                  200 SMA{sortArrow('dist_sma200_pct')}
                </th>
                <th onClick={() => handleSort('alpha_20d_cum')} style={{ cursor: 'pointer' }}>
                  Alfa (20G){sortArrow('alpha_20d_cum')}
                </th>
                <th>Beta</th>
                <th>Hacim ×</th>
                <th onClick={() => handleSort('confidence_score')} style={{ cursor: 'pointer' }}>
                  AI Kararı{sortArrow('confidence_score')}
                </th>
                <th style={{ textAlign: 'right', paddingRight: '2rem' }}>Hızlı Erişim</th>
              </tr>
            </thead>
            <tbody>
              {filteredAndSorted.length === 0 ? (
                <tr>
                  <td colSpan={10} style={{ textAlign: 'center', padding: '3rem', color: 'var(--ink-muted)', fontStyle: 'italic' }}>
                    Bu filtrelerle eşleşen varlık bulunamadı.
                  </td>
                </tr>
              ) : filteredAndSorted.map((item) => {
                const isUp = item.change_pct >= 0;
                const smaUp = item.dist_sma200_pct >= 0;
                const alphaUp = item.alpha_20d_cum >= 0;
                return (
                  <tr
                    key={item.ticker}
                    className="screener-row"
                    onClick={() => { onSelectTicker(item.ticker); onNavigateTab('terminal'); }}
                  >
                    {/* Asset name */}
                    <td style={{ paddingLeft: '2rem' }}>
                      <div>
                        <span className="tabular" style={{ fontWeight: 700, fontSize: '0.92rem', color: 'var(--ink-primary)' }}>
                          {item.ticker}
                        </span>
                        <div style={{ fontSize: '0.76rem', color: 'var(--ink-muted)', marginTop: 1 }}>
                          {item.name}
                        </div>
                      </div>
                    </td>

                    {/* Category */}
                    <td>
                      <span style={{
                        fontSize: '0.68rem',
                        fontWeight: 600,
                        letterSpacing: '0.08em',
                        textTransform: 'uppercase',
                        color: 'var(--ink-muted)',
                        fontFamily: 'var(--font-body)',
                      }}>
                        {item.sector}
                      </span>
                    </td>

                    {/* Price */}
                    <td>
                      <span className="tabular" style={{ fontWeight: 700, fontSize: '0.9rem', color: 'var(--ink-primary)' }}>
                        {item.last_close.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </span>
                    </td>

                    {/* 24h change */}
                    <td>
                      <span className={`tabular ${isUp ? 'change-up' : 'change-down'}`} style={{ fontWeight: 700 }}>
                        {isUp ? '+' : ''}{item.change_pct.toFixed(2)}%
                      </span>
                    </td>

                    {/* SMA 200 distance */}
                    <td>
                      <span className="tabular" style={{ fontWeight: 600, color: smaUp ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
                        {smaUp ? '+' : ''}{item.dist_sma200_pct.toFixed(2)}%
                      </span>
                    </td>

                    {/* Alpha 20d */}
                    <td>
                      <span className="tabular" style={{ fontWeight: 600, color: alphaUp ? 'var(--forest-gain)' : 'var(--ink-secondary)' }}>
                        {alphaUp ? '+' : ''}{item.alpha_20d_cum.toFixed(2)}%
                      </span>
                    </td>

                    {/* Beta */}
                    <td>
                      <span className="tabular" style={{ color: 'var(--ink-secondary)', fontSize: '0.85rem' }}>
                        {item.beta.toFixed(2)}
                      </span>
                    </td>

                    {/* Volume ratio */}
                    <td>
                      <span className="tabular" style={{
                        fontWeight: item.volume_ratio > 1.3 ? 700 : 400,
                        color: item.volume_ratio > 1.3 ? 'var(--amber-warm)' : 'var(--ink-secondary)',
                        fontSize: '0.85rem',
                      }}>
                        {item.volume_ratio.toFixed(2)}×
                      </span>
                    </td>

                    {/* AI Signal */}
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span className={`signal ${signalClass(item.ai_signal)}`}>
                          {item.ai_signal}
                        </span>
                        <span className="tabular" style={{ fontSize: '0.72rem', color: 'var(--cobalt)', fontWeight: 600 }}>
                          %{item.confidence_score}
                        </span>
                      </div>
                    </td>

                    {/* Actions */}
                    <td style={{ textAlign: 'right', paddingRight: '2rem' }}>
                      <div style={{ display: 'inline-flex', gap: 6 }}>
                        <button
                          className="btn btn-secondary"
                          style={{ padding: '3px 10px', fontSize: '0.72rem' }}
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectTicker(item.ticker);
                            onNavigateTab('terminal');
                          }}
                        >
                          Terminal
                        </button>
                        <button
                          className="btn btn-primary"
                          style={{ padding: '3px 10px', fontSize: '0.72rem' }}
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

        {/* Table footer */}
        <div style={{
          padding: '0.75rem 2rem',
          borderTop: '1px solid var(--rule-hairline)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '0.72rem',
          color: 'var(--ink-muted)',
        }}>
          <span>{filteredAndSorted.length} varlık gösteriliyor</span>
          <span style={{ fontStyle: 'italic' }}>
            Veriler yfinance üzerinden gerçek zamanlı alınmaktadır · Google TimesFM 3.0 Foundation Model &amp; Çok-Faktörlü Quant Motoru
          </span>
        </div>
      </div>
    </div>
  );
};
