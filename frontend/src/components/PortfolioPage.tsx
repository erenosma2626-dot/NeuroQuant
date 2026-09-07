import React, { useState } from 'react';
import type { ScreenerItem, UserPortfolio } from '../types';

interface PortfolioPageProps {
  screenerData: ScreenerItem[];
  userPortfolio: UserPortfolio;
  onUpdatePortfolio: (updated: UserPortfolio) => void;
  onSelectTicker: (ticker: string) => void;
  onNavigateTab: (tab: 'dashboard' | 'terminal' | 'simulation' | 'portfolio') => void;
}

// A palette of editorial / restrained colours for allocation bar
const ALLOC_COLORS = ['#14532D', '#1E3A8A', '#881337', '#92400E', '#4B5563', '#166534'];

export const PortfolioPage: React.FC<PortfolioPageProps> = ({
  screenerData,
  userPortfolio,
  onUpdatePortfolio,
  onSelectTicker,
  onNavigateTab,
}) => {
  const [selectedTickerToAdd, setSelectedTickerToAdd] = useState<string>(screenerData[0]?.ticker || 'NVDA');
  const [sharesToAdd, setSharesToAdd]                 = useState<number>(10);

  /* ── Portfolio Stats ──────────────────────────────────────── */
  const totalStockVal    = userPortfolio.positions.reduce((s, p) => s + p.shares * p.current_price, 0);
  const totalPortfolioVal = userPortfolio.cash + totalStockVal;
  const initialCap       = userPortfolio.initial_capital;
  const totalGain        = totalPortfolioVal - initialCap;
  const totalReturnPct   = (totalGain / initialCap) * 100;
  const isPositive       = totalReturnPct >= 0;

  const weightedBeta = totalStockVal > 0
    ? userPortfolio.positions.reduce((s, p) => {
        const item = screenerData.find((x) => x.ticker === p.ticker);
        return s + (p.shares * p.current_price / totalStockVal) * (item?.beta ?? 1);
      }, 0)
    : 0;

  const weightedAlpha = totalStockVal > 0
    ? userPortfolio.positions.reduce((s, p) => {
        const item = screenerData.find((x) => x.ticker === p.ticker);
        return s + (p.shares * p.current_price / totalStockVal) * (item?.alpha_20d_cum ?? 0);
      }, 0)
    : 0;

  /* ── Actions ──────────────────────────────────────────────── */
  const handleAddPosition = (e: React.FormEvent) => {
    e.preventDefault();
    const asset = screenerData.find((s) => s.ticker === selectedTickerToAdd);
    if (!asset) return;
    const cost = sharesToAdd * asset.last_close;
    if (cost > userPortfolio.cash) {
      alert(`Yetersiz nakit! Gerekli: ${cost.toFixed(2)}, Mevcut: ${userPortfolio.cash.toFixed(2)}`);
      return;
    }
    const existingIdx = userPortfolio.positions.findIndex((p) => p.ticker === selectedTickerToAdd);
    let newPositions = [...userPortfolio.positions];
    if (existingIdx >= 0) {
      const ex = newPositions[existingIdx];
      const newShares = ex.shares + sharesToAdd;
      newPositions[existingIdx] = {
        ...ex,
        shares: newShares,
        buy_price: (ex.shares * ex.buy_price + cost) / newShares,
        current_price: asset.last_close,
        weight_pct: 0,
      };
    } else {
      newPositions.push({
        ticker: asset.ticker, name: asset.name, category: asset.category,
        shares: sharesToAdd, buy_price: asset.last_close,
        current_price: asset.last_close, weight_pct: 0,
      });
    }
    onUpdatePortfolio({ ...userPortfolio, cash: userPortfolio.cash - cost, positions: newPositions });
  };

  const handleRemovePosition = (ticker: string) => {
    const pos = userPortfolio.positions.find((p) => p.ticker === ticker);
    if (!pos) return;
    onUpdatePortfolio({
      ...userPortfolio,
      cash: userPortfolio.cash + pos.shares * pos.current_price,
      positions: userPortfolio.positions.filter((p) => p.ticker !== ticker),
    });
  };

  const handleResetPortfolio = () => {
    if (confirm('Portföy 10.000 başlangıç nakdine sıfırlansın mı?')) {
      onUpdatePortfolio({ name: 'Kurumsal Ana Portföy', initial_capital: 10000, cash: 10000, positions: [] });
    }
  };

  const cashPct = totalPortfolioVal > 0 ? (userPortfolio.cash / totalPortfolioVal) * 100 : 100;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', animation: 'fadeUp 0.35s ease' }}>

      {/* ── PAGE HEADER ────────────────────────────────────────────── */}
      <div style={{ borderTop: '3px solid var(--ink-primary)', paddingTop: '1.5rem' }}>
        <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 6 }}>
          Portföy Laboratuvarı
        </div>
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between' }}>
          <div>
            <h1 style={{ fontFamily: 'var(--font-display)', fontSize: '2rem', fontWeight: 700, color: 'var(--ink-primary)', letterSpacing: '-0.02em', lineHeight: 1 }}>
              Portföy Yöneticisi &amp; Risk Radarı
            </h1>
            <p style={{ color: 'var(--ink-secondary)', fontSize: '0.88rem', marginTop: 8, lineHeight: 1.6, maxWidth: 800 }}>
              Kendi hisse ve varlık sepetinizi oluşturun; yapay zeka konsensüsünü, ağırlıklı beta ve sektörel alfasını tek ekranda izleyin.
            </p>
          </div>
          <button className="btn btn-secondary" onClick={handleResetPortfolio} style={{ flexShrink: 0, fontSize: '0.78rem' }}>
            ↺ Portföyü Sıfırla
          </button>
        </div>
      </div>

      {/* ── METRICS STRIP ──────────────────────────────────────────── */}
      <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)' }}>
        <div className="metric-block">
          <div className="metric-label">Toplam Portföy</div>
          <div className="metric-value tabular">{totalPortfolioVal.toLocaleString('tr-TR', { minimumFractionDigits: 0 })}</div>
          <div className="metric-sub">Başlangıç: {initialCap.toLocaleString('tr-TR')}</div>
        </div>
        <div className="metric-block">
          <div className="metric-label">Net Getiri</div>
          <div className="metric-value tabular" style={{ color: isPositive ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
            {isPositive ? '+' : ''}{totalReturnPct.toFixed(2)}%
          </div>
          <div className="metric-sub" style={{ color: isPositive ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
            {isPositive ? '+' : ''}{totalGain.toLocaleString('tr-TR', { minimumFractionDigits: 0 })}
          </div>
        </div>
        <div className="metric-block">
          <div className="metric-label">Ağırlıklı Beta</div>
          <div className="metric-value tabular" style={{ color: 'var(--cobalt)' }}>{weightedBeta.toFixed(2)}</div>
          <div className="metric-sub">{weightedBeta < 1 ? 'Defansif' : 'Agresif / Piyasa Üstü'}</div>
        </div>
        <div className="metric-block">
          <div className="metric-label">Sektörel Alfa (20G)</div>
          <div className="metric-value tabular" style={{ color: weightedAlpha >= 0 ? 'var(--forest-gain)' : 'var(--ink-muted)' }}>
            {weightedAlpha >= 0 ? '+' : ''}{weightedAlpha.toFixed(2)}%
          </div>
          <div className="metric-sub">Endekse karşı göreceli güç</div>
        </div>
      </div>

      {/* ── ALLOCATION BAR ─────────────────────────────────────────── */}
      <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)', padding: '1.25rem 2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
          <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)' }}>
            Varlık Tahsis Dağılımı
          </div>
          <div className="tabular" style={{ fontSize: '0.75rem', color: 'var(--ink-muted)' }}>
            Nakit: {userPortfolio.cash.toLocaleString('tr-TR', { maximumFractionDigits: 0 })} ({cashPct.toFixed(1)}%)
          </div>
        </div>

        {/* Stacked bar */}
        <div style={{ height: 10, width: '100%', background: 'var(--rule-hairline)', borderRadius: 2, overflow: 'hidden', display: 'flex' }}>
          {userPortfolio.positions.map((pos, idx) => {
            const pct = totalPortfolioVal > 0 ? ((pos.shares * pos.current_price) / totalPortfolioVal) * 100 : 0;
            return (
              <div
                key={pos.ticker}
                style={{ width: `${pct}%`, background: ALLOC_COLORS[idx % ALLOC_COLORS.length], transition: 'width 0.3s ease' }}
                title={`${pos.ticker}: ${pct.toFixed(1)}%`}
              />
            );
          })}
          <div
            style={{ width: `${cashPct}%`, background: 'var(--rule-strong)' }}
            title={`Nakit: ${cashPct.toFixed(1)}%`}
          />
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: 10 }}>
          {userPortfolio.positions.map((pos, idx) => {
            const pct = totalPortfolioVal > 0 ? ((pos.shares * pos.current_price) / totalPortfolioVal) * 100 : 0;
            return (
              <div key={pos.ticker} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.76rem' }}>
                <span style={{ width: 10, height: 3, borderRadius: 1, background: ALLOC_COLORS[idx % ALLOC_COLORS.length] }} />
                <span className="tabular" style={{ fontWeight: 600, color: 'var(--ink-primary)' }}>{pos.ticker}</span>
                <span className="tabular" style={{ color: 'var(--ink-muted)' }}>{pct.toFixed(1)}%</span>
              </div>
            );
          })}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.76rem' }}>
            <span style={{ width: 10, height: 3, borderRadius: 1, background: 'var(--rule-strong)' }} />
            <span style={{ fontWeight: 600, color: 'var(--ink-primary)' }}>Nakit</span>
            <span className="tabular" style={{ color: 'var(--ink-muted)' }}>{cashPct.toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {/* ── POSITIONS TABLE + ADD PANEL ─────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '1.5rem', alignItems: 'start' }}>

        {/* Positions table */}
        <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)' }}>
          <div style={{
            padding: '1rem 2rem',
            borderBottom: '2px solid var(--ink-primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', fontWeight: 700, fontStyle: 'italic', color: 'var(--ink-primary)' }}>
              Aktif Pozisyonlar ({userPortfolio.positions.length})
            </div>
            <div className="tabular" style={{ fontSize: '0.72rem', color: 'var(--ink-muted)' }}>
              Nakit Rezervi: {userPortfolio.cash.toLocaleString('tr-TR', { maximumFractionDigits: 0 })}
            </div>
          </div>

          {userPortfolio.positions.length === 0 ? (
            <div style={{ padding: '3rem 2rem', textAlign: 'center', color: 'var(--ink-muted)', fontFamily: 'var(--font-display)', fontStyle: 'italic' }}>
              Henüz bir varlık eklemediniz. Sağdaki panelden hisse seçerek portföy oluşturun.
            </div>
          ) : (
            <table className="screener-table" style={{ width: '100%' }}>
              <thead>
                <tr>
                  <th style={{ paddingLeft: '2rem' }}>Varlık</th>
                  <th>Adet</th>
                  <th>Alış</th>
                  <th>Güncel</th>
                  <th>Piyasa Değeri</th>
                  <th>K/Z</th>
                  <th style={{ paddingRight: '2rem', textAlign: 'right' }}>İşlem</th>
                </tr>
              </thead>
              <tbody>
                {userPortfolio.positions.map((pos) => {
                  const marketVal  = pos.shares * pos.current_price;
                  const gain       = marketVal - pos.shares * pos.buy_price;
                  const gainPct    = (gain / (pos.shares * pos.buy_price)) * 100;
                  const isProfitable = gain >= 0;
                  return (
                    <tr key={pos.ticker} className="screener-row">
                      <td style={{ paddingLeft: '2rem' }}>
                        <div
                          onClick={() => { onSelectTicker(pos.ticker); onNavigateTab('terminal'); }}
                          style={{ cursor: 'pointer' }}
                        >
                          <span className="tabular" style={{ fontWeight: 700, fontSize: '0.88rem', color: 'var(--ink-primary)' }}>
                            {pos.ticker}
                          </span>
                          <div style={{ fontSize: '0.72rem', color: 'var(--ink-muted)', marginTop: 1 }}>{pos.name}</div>
                        </div>
                      </td>
                      <td className="tabular" style={{ fontWeight: 600, color: 'var(--ink-primary)' }}>{pos.shares}</td>
                      <td className="tabular" style={{ color: 'var(--ink-secondary)' }}>{pos.buy_price.toFixed(2)}</td>
                      <td className="tabular" style={{ fontWeight: 700, color: 'var(--ink-primary)' }}>{pos.current_price.toFixed(2)}</td>
                      <td className="tabular" style={{ fontWeight: 600, color: 'var(--cobalt)' }}>{marketVal.toFixed(0)}</td>
                      <td>
                        <span className={`signal ${isProfitable ? 'signal-buy' : 'signal-sell'} tabular`}>
                          {isProfitable ? '+' : ''}{gainPct.toFixed(2)}%
                        </span>
                      </td>
                      <td style={{ textAlign: 'right', paddingRight: '2rem' }}>
                        <button
                          onClick={() => handleRemovePosition(pos.ticker)}
                          className="btn btn-secondary"
                          style={{ padding: '3px 8px', fontSize: '0.72rem', color: 'var(--madder-loss)', borderColor: 'var(--madder-rule)' }}
                          title="Pozisyonu Kapat"
                        >
                          Kapat ✕
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* Add position form */}
        <div className="panel" style={{ borderTop: '3px solid var(--forest-gain)' }}>
          <div style={{ padding: '1rem 1.5rem', borderBottom: '2px solid var(--ink-primary)' }}>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', fontWeight: 700, fontStyle: 'italic', color: 'var(--ink-primary)' }}>
              Portföye Varlık Ekle
            </div>
          </div>

          <form onSubmit={handleAddPosition} style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div>
              <label style={{ fontSize: '0.65rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', display: 'block', marginBottom: 6 }}>
                Varlık Seçin
              </label>
              <select
                value={selectedTickerToAdd}
                onChange={(e) => setSelectedTickerToAdd(e.target.value)}
                style={{
                  width: '100%',
                  padding: '8px 10px',
                  background: 'var(--paper-card)',
                  border: '1px solid var(--rule-strong)',
                  borderRadius: 'var(--radius-sm)',
                  color: 'var(--ink-primary)',
                  fontFamily: 'var(--font-body)',
                  fontSize: '0.82rem',
                  outline: 'none',
                }}
              >
                {screenerData.map((item) => (
                  <option key={item.ticker} value={item.ticker} style={{ background: '#FAF8F3' }}>
                    {item.ticker} — {item.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ fontSize: '0.65rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', display: 'block', marginBottom: 6 }}>
                Adet / Miktar
              </label>
              <input
                type="number"
                min={1}
                value={sharesToAdd}
                onChange={(e) => setSharesToAdd(Math.max(1, Number(e.target.value)))}
                className="search-input"
                style={{ width: '100%', padding: '8px 10px' }}
              />
            </div>

            {/* Cost estimate */}
            {(() => {
              const item = screenerData.find((s) => s.ticker === selectedTickerToAdd);
              const cost = item ? item.last_close * sharesToAdd : 0;
              const canAfford = cost <= userPortfolio.cash;
              return (
                <div style={{
                  padding: '10px 12px',
                  background: 'var(--paper-elevated)',
                  border: `1px solid ${canAfford ? 'var(--rule-hairline)' : 'var(--madder-rule)'}`,
                  borderRadius: 'var(--radius-xs)',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.76rem', marginBottom: 4 }}>
                    <span style={{ color: 'var(--ink-muted)' }}>Tahmini Maliyet</span>
                    <span className="tabular" style={{ fontWeight: 700, color: canAfford ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
                      {cost.toFixed(2)}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem' }}>
                    <span style={{ color: 'var(--ink-muted)' }}>Mevcut Nakit</span>
                    <span className="tabular" style={{ color: 'var(--ink-secondary)' }}>{userPortfolio.cash.toFixed(2)}</span>
                  </div>
                </div>
              );
            })()}

            <button type="submit" className="btn btn-primary" style={{ justifyContent: 'center' }}>
              + Varlığı Ekle
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};
