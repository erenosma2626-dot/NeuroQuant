import React, { useState } from 'react';
import type { ScreenerItem, UserPortfolio } from '../types';
import { 
  Briefcase, 
  Plus, 
  Trash2, 
  RotateCcw,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react';

interface PortfolioPageProps {
  screenerData: ScreenerItem[];
  userPortfolio: UserPortfolio;
  onUpdatePortfolio: (updated: UserPortfolio) => void;
  onSelectTicker: (ticker: string) => void;
  onNavigateTab: (tab: 'dashboard' | 'terminal' | 'simulation' | 'portfolio') => void;
}

export const PortfolioPage: React.FC<PortfolioPageProps> = ({
  screenerData,
  userPortfolio,
  onUpdatePortfolio,
  onSelectTicker,
  onNavigateTab,
}) => {
  const [selectedTickerToAdd, setSelectedTickerToAdd] = useState<string>(screenerData[0]?.ticker || 'NVDA');
  const [sharesToAdd, setSharesToAdd] = useState<number>(10);

  // Portföy İstatistikleri
  const totalStockVal = userPortfolio.positions.reduce((acc, p) => acc + p.shares * p.current_price, 0);
  const totalPortfolioVal = userPortfolio.cash + totalStockVal;
  const initialCap = userPortfolio.initial_capital;
  const totalGain = totalPortfolioVal - initialCap;
  const totalReturnPct = (totalGain / initialCap) * 100;

  // Ağırlıklı Beta & Alfa
  const weightedBeta = totalStockVal > 0
    ? userPortfolio.positions.reduce((acc, p) => {
        const item = screenerData.find((s) => s.ticker === p.ticker);
        const beta = item ? item.beta : 1.0;
        return acc + (p.shares * p.current_price / totalStockVal) * beta;
      }, 0)
    : 0;

  const weightedAlpha = totalStockVal > 0
    ? userPortfolio.positions.reduce((acc, p) => {
        const item = screenerData.find((s) => s.ticker === p.ticker);
        const alpha = item ? item.alpha_20d_cum : 0;
        return acc + (p.shares * p.current_price / totalStockVal) * alpha;
      }, 0)
    : 0;

  // Yeni Varlık Ekleme
  const handleAddPosition = (e: React.FormEvent) => {
    e.preventDefault();
    const asset = screenerData.find((s) => s.ticker === selectedTickerToAdd);
    if (!asset) return;

    const cost = sharesToAdd * asset.last_close;
    if (cost > userPortfolio.cash) {
      alert(`Yetersiz nakit! Gerekli: $${cost.toFixed(2)}, Mevcut Nakit: $${userPortfolio.cash.toFixed(2)}`);
      return;
    }

    const existingIndex = userPortfolio.positions.findIndex((p) => p.ticker === selectedTickerToAdd);
    let newPositions = [...userPortfolio.positions];

    if (existingIndex >= 0) {
      const existing = newPositions[existingIndex];
      const newShares = existing.shares + sharesToAdd;
      const avgPrice = (existing.shares * existing.buy_price + cost) / newShares;
      newPositions[existingIndex] = {
        ...existing,
        shares: newShares,
        buy_price: avgPrice,
        current_price: asset.last_close,
        weight_pct: 0,
      };
    } else {
      newPositions.push({
        ticker: asset.ticker,
        name: asset.name,
        category: asset.category,
        shares: sharesToAdd,
        buy_price: asset.last_close,
        current_price: asset.last_close,
        weight_pct: 0,
      });
    }

    const newCash = userPortfolio.cash - cost;
    onUpdatePortfolio({
      ...userPortfolio,
      cash: Math.max(0, newCash),
      positions: newPositions,
    });
  };

  // Pozisyon Çıkarma / Satma
  const handleRemovePosition = (ticker: string) => {
    const pos = userPortfolio.positions.find((p) => p.ticker === ticker);
    if (!pos) return;

    const cashBack = pos.shares * pos.current_price;
    const updatedPositions = userPortfolio.positions.filter((p) => p.ticker !== ticker);

    onUpdatePortfolio({
      ...userPortfolio,
      cash: userPortfolio.cash + cashBack,
      positions: updatedPositions,
    });
  };

  // Portföyü Sıfırlama
  const handleResetPortfolio = () => {
    if (confirm('Portföy 10.000$ başlangıç nakdine sıfırlansın mı?')) {
      onUpdatePortfolio({
        name: 'Kurumsal Ana Portföy',
        initial_capital: 10000,
        cash: 10000,
        positions: [],
      });
    }
  };

  const COLORS = ['#10B981', '#0EA5E9', '#8B5CF6', '#F59E0B', '#EC4899', '#3B82F6'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* 1. Üst Başlık */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <span style={{ fontSize: '0.8rem', color: 'var(--accent-sky)', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
            Portföy Laboratuvarı
          </span>
          <h1 style={{ fontSize: '2rem', color: '#FFFFFF', marginTop: 4 }}>
            Özel Portföy Yöneticisi & Risk Radarı
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginTop: 4, maxWidth: 800 }}>
            Kendi hisse ve varlık sepetinizi oluşturun; yapay zeka konsensüsünü, ağırlıklı beta ve sektörel alfasını tek bir ekranda canlı izleyin.
          </p>
        </div>

        <button className="btn-secondary" onClick={handleResetPortfolio}>
          <RotateCcw size={14} /> Portföyü Sıfırla
        </button>
      </div>

      {/* 2. Portföy Temel Metrik Kartları */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem' }}>
        <div className="metric-pill">
          <div className="metric-pill-label">Toplam Portföy Büyüklüğü</div>
          <div className="metric-pill-val tabular" style={{ color: '#FFFFFF' }}>
            ${totalPortfolioVal.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 4 }}>
            Başlangıç: ${initialCap.toLocaleString()}
          </div>
        </div>

        <div className="metric-pill">
          <div className="metric-pill-label">Net Kâr / Zarar</div>
          <div
            className="metric-pill-val tabular"
            style={{ color: totalReturnPct >= 0 ? 'var(--bull-text)' : 'var(--bear-text)' }}
          >
            {totalReturnPct >= 0 ? `+$${totalGain.toLocaleString('en-US', { minimumFractionDigits: 2 })}` : `-$${Math.abs(totalGain).toLocaleString('en-US', { minimumFractionDigits: 2 })}`}
          </div>
          <div style={{ fontSize: '0.78rem', fontWeight: 600, color: totalReturnPct >= 0 ? 'var(--bull-text)' : 'var(--bear-text)', marginTop: 4 }}>
            %{totalReturnPct >= 0 ? `+${totalReturnPct.toFixed(2)}` : totalReturnPct.toFixed(2)}
          </div>
        </div>

        <div className="metric-pill">
          <div className="metric-pill-label">Ağırlıklı Portföy Betası</div>
          <div className="metric-pill-val tabular" style={{ color: 'var(--accent-sky)' }}>
            {weightedBeta.toFixed(2)}
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 4 }}>
            {weightedBeta < 1 ? 'Defansif / Düşük Volatilite' : 'Agresif / Piyasa Üstü'}
          </div>
        </div>

        <div className="metric-pill">
          <div className="metric-pill-label">Bileşik Sektörel Alfa (20G)</div>
          <div
            className="metric-pill-val tabular"
            style={{ color: weightedAlpha >= 0 ? 'var(--bull-text)' : 'var(--text-muted)' }}
          >
            %{weightedAlpha >= 0 ? `+${weightedAlpha.toFixed(2)}` : weightedAlpha.toFixed(2)}
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 4 }}>
            Endekslere Karşı Göreceli Güç
          </div>
        </div>
      </div>

      {/* 3. Varlık Dağılım Çubuğu (Asset Allocation Bar) */}
      <div className="card" style={{ padding: '1.5rem 2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            Varlık Tahsis Dağılımı:
          </span>
          <span className="tabular" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Hisse: %{totalPortfolioVal > 0 ? ((totalStockVal / totalPortfolioVal) * 100).toFixed(1) : 0} | Nakit: ${userPortfolio.cash.toFixed(2)} (%{totalPortfolioVal > 0 ? ((userPortfolio.cash / totalPortfolioVal) * 100).toFixed(1) : 100})
          </span>
        </div>

        <div style={{ height: 14, width: '100%', background: 'var(--bg-surface)', borderRadius: 7, overflow: 'hidden', display: 'flex' }}>
          {userPortfolio.positions.map((pos, idx) => {
            const pct = totalPortfolioVal > 0 ? ((pos.shares * pos.current_price) / totalPortfolioVal) * 100 : 0;
            return (
              <div
                key={pos.ticker}
                style={{
                  width: `${pct}%`,
                  background: COLORS[idx % COLORS.length],
                  transition: 'width 0.3s ease',
                }}
                title={`${pos.ticker}: %${pct.toFixed(1)}`}
              />
            );
          })}
          {/* Nakit dilimi */}
          <div
            style={{
              width: `${totalPortfolioVal > 0 ? (userPortfolio.cash / totalPortfolioVal) * 100 : 100}%`,
              background: '#334155',
            }}
            title={`Nakit: $${userPortfolio.cash.toFixed(2)}`}
          />
        </div>

        {/* Dağılım Rozetleri */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginTop: 14 }}>
          {userPortfolio.positions.map((pos, idx) => {
            const pct = totalPortfolioVal > 0 ? ((pos.shares * pos.current_price) / totalPortfolioVal) * 100 : 0;
            return (
              <div key={pos.ticker} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.78rem' }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: COLORS[idx % COLORS.length] }} />
                <span className="tabular" style={{ fontWeight: 600, color: '#FFFFFF' }}>{pos.ticker}</span>
                <span className="tabular" style={{ color: 'var(--text-muted)' }}>%{pct.toFixed(1)}</span>
              </div>
            );
          })}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.78rem' }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#334155' }} />
            <span style={{ fontWeight: 600, color: '#FFFFFF' }}>Nakit</span>
            <span className="tabular" style={{ color: 'var(--text-muted)' }}>
              %{totalPortfolioVal > 0 ? ((userPortfolio.cash / totalPortfolioVal) * 100).toFixed(1) : 100}
            </span>
          </div>
        </div>
      </div>

      {/* 4. Portföy Pozisyonları Tablosu & Yeni Varlık Ekleme */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '1.5rem' }}>
        {/* Pozisyonlar Tablosu */}
        <div className="card" style={{ padding: '1.75rem' }}>
          <div className="card-header">
            <h2 className="card-title">
              <Briefcase size={18} color="var(--accent-sky)" />
              Aktif Varlık Pozisyonları ({userPortfolio.positions.length})
            </h2>
          </div>

          {userPortfolio.positions.length === 0 ? (
            <div style={{ padding: '3rem 1rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              Henüz bir varlık eklemediniz. Sağdaki panelden hisse seçerek portföy oluşturmaya başlayabilirsiniz.
            </div>
          ) : (
            <div className="screener-table-container">
              <table className="screener-table">
                <thead>
                  <tr>
                    <th>Varlık</th>
                    <th>Adet / Pay</th>
                    <th>Alış Fiyatı</th>
                    <th>Güncel Fiyat</th>
                    <th>Piyasa Değeri</th>
                    <th>Kâr / Zarar</th>
                    <th style={{ textAlign: 'right' }}>İşlem</th>
                  </tr>
                </thead>
                <tbody>
                  {userPortfolio.positions.map((pos) => {
                    const marketVal = pos.shares * pos.current_price;
                    const gain = marketVal - (pos.shares * pos.buy_price);
                    const gainPct = (gain / (pos.shares * pos.buy_price)) * 100;
                    const isProfitable = gain >= 0;

                    return (
                      <tr key={pos.ticker} className="screener-row">
                        <td>
                          <div 
                            onClick={() => {
                              onSelectTicker(pos.ticker);
                              onNavigateTab('terminal');
                            }}
                            style={{ cursor: 'pointer' }}
                          >
                            <span className="tabular" style={{ fontWeight: 700, color: '#FFFFFF' }}>
                              {pos.ticker}
                            </span>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                              {pos.name}
                            </div>
                          </div>
                        </td>
                        <td className="tabular" style={{ fontWeight: 600 }}>{pos.shares}</td>
                        <td className="tabular">${pos.buy_price.toFixed(2)}</td>
                        <td className="tabular" style={{ fontWeight: 700, color: '#FFFFFF' }}>
                          ${pos.current_price.toFixed(2)}
                        </td>
                        <td className="tabular" style={{ fontWeight: 700, color: 'var(--accent-sky)' }}>
                          ${marketVal.toFixed(2)}
                        </td>
                        <td>
                          <span className={`tag ${isProfitable ? 'tag-bull' : 'tag-bear'} tabular`}>
                            {isProfitable ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                            %{isProfitable ? `+${gainPct.toFixed(2)}` : gainPct.toFixed(2)}
                          </span>
                        </td>
                        <td style={{ textAlign: 'right' }}>
                          <button
                            onClick={() => handleRemovePosition(pos.ticker)}
                            className="btn-secondary"
                            style={{ padding: '4px 8px', color: 'var(--bear-text)', borderColor: 'var(--bear-border)' }}
                            title="Pozisyonu Kapat / Sat"
                          >
                            <Trash2 size={13} />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Yeni Varlık Ekleme Paneli */}
        <div className="card" style={{ padding: '1.75rem', height: 'fit-content' }}>
          <h3 style={{ fontSize: '1.05rem', color: '#FFFFFF', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Plus size={16} color="var(--accent-sky)" />
            Portföye Varlık Ekle
          </h3>

          <form onSubmit={handleAddPosition} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div>
              <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
                Varlık Seçiniz:
              </label>
              <select
                value={selectedTickerToAdd}
                onChange={(e) => setSelectedTickerToAdd(e.target.value)}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  background: 'var(--bg-surface)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 8,
                  color: '#FFFFFF',
                  fontFamily: 'inherit',
                  outline: 'none',
                }}
              >
                {screenerData.map((item) => (
                  <option key={item.ticker} value={item.ticker} style={{ background: '#0F141C' }}>
                    {item.ticker} - {item.name} (${item.last_close})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
                Adet / Miktar:
              </label>
              <input
                type="number"
                min={1}
                value={sharesToAdd}
                onChange={(e) => setSharesToAdd(Math.max(1, Number(e.target.value)))}
                className="search-input"
                style={{ width: '100%', padding: '8px 12px' }}
              />
            </div>

            {/* Tahmini Maliyet */}
            {(() => {
              const item = screenerData.find((s) => s.ticker === selectedTickerToAdd);
              const cost = item ? item.last_close * sharesToAdd : 0;
              return (
                <div style={{ padding: '10px 12px', background: 'var(--bg-surface)', borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    <span>Tahmini Maliyet:</span>
                    <span className="tabular" style={{ fontWeight: 700, color: '#FFFFFF' }}>${cost.toFixed(2)}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>
                    <span>Mevcut Nakit:</span>
                    <span className="tabular">${userPortfolio.cash.toFixed(2)}</span>
                  </div>
                </div>
              );
            })()}

            <button type="submit" className="btn-primary" style={{ justifyContent: 'center', marginTop: 6 }}>
              <Plus size={16} /> Varlığı Ekle
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};
