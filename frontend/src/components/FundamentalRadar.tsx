import React from 'react';
import type { FundamentalsData } from '../types';
import { PieChart, AlertCircle, Calendar } from 'lucide-react';

interface FundamentalRadarProps {
  data: FundamentalsData;
}

export const FundamentalRadar: React.FC<FundamentalRadarProps> = ({ data }) => {
  if (!data.is_equity) {
    return (
      <div className="card" style={{ padding: '1.75rem 2rem' }}>
        <div className="card-header">
          <div className="card-title-group">
            <h2 className="card-title">
              <PieChart size={19} color="var(--accent-sky)" />
              Temel Değerleme & Çarpan Radarı
            </h2>
            <p className="card-subtitle">Varlık sınıfına göre değerleme yaklaşımı.</p>
          </div>
        </div>
        <div style={{ padding: '2.5rem 1.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <AlertCircle size={32} color="var(--accent-amber)" style={{ margin: '0 auto 12px' }} />
          <h3 style={{ color: '#FFFFFF', marginBottom: 6, fontSize: '1.1rem' }}>Kripto / Emtia Rejimi</h3>
          <p style={{ fontSize: '0.88rem', maxWidth: 480, margin: '0 auto', color: 'var(--text-secondary)' }}>
            {data.ticker} için geleneksel hisse senedi çarpanları (F/K, F/DD) uygulanmaz. Model bunun yerine on-chain akış, volatilite ve sektörel korelasyona odaklanır.
          </p>
        </div>
      </div>
    );
  }

  const isCheap = data.valuation_score >= 65;
  const isExpensive = data.valuation_score <= 40;

  return (
    <div className="card" style={{ padding: '1.75rem 2rem' }}>
      <div className="card-header">
        <div className="card-title-group">
          <h2 className="card-title">
            <PieChart size={19} color="var(--accent-sky)" />
            Temel Değerleme & Bilanço Radarı
          </h2>
          <p className="card-subtitle">
            Hissenin tarihsel çarpanları ve akran grubuna göre ucuzluk/pahallılık skoru.
          </p>
        </div>

        <span
          className={`tag ${isCheap ? 'tag-bull' : isExpensive ? 'tag-bear' : 'tag-accent'}`}
          style={{ fontSize: '0.8rem', padding: '3px 10px', fontWeight: 700 }}
        >
          {data.valuation_status}
        </span>
      </div>

      {/* 4 Ana Çarpan Kutusu */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
        <div className="metric-pill">
          <div className="metric-pill-label">F/K (Trailing P/E)</div>
          <div className="metric-pill-val tabular" style={{ color: '#FFFFFF' }}>
            {data.trailing_pe ? data.trailing_pe.toFixed(2) : '—'}
          </div>
        </div>

        <div className="metric-pill">
          <div className="metric-pill-label">İleri F/K (Forward P/E)</div>
          <div className="metric-pill-val tabular" style={{ color: '#FFFFFF' }}>
            {data.forward_pe ? data.forward_pe.toFixed(2) : '—'}
          </div>
        </div>

        <div className="metric-pill">
          <div className="metric-pill-label">F/DD (Price/Book)</div>
          <div className="metric-pill-val tabular" style={{ color: '#FFFFFF' }}>
            {data.price_to_book ? data.price_to_book.toFixed(2) : '—'}
          </div>
        </div>

        <div className="metric-pill">
          <div className="metric-pill-label">PEG Oranı</div>
          <div className="metric-pill-val tabular" style={{ color: '#FFFFFF' }}>
            {data.peg_ratio ? data.peg_ratio.toFixed(2) : '—'}
          </div>
        </div>
      </div>

      {/* Bilanço Rejimi İbresi */}
      <div
        style={{
          padding: '1.25rem 1.5rem',
          borderRadius: 'var(--radius-md)',
          background: 'var(--bg-surface)',
          border: '1px solid var(--border-subtle)',
          marginBottom: '1.5rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.9rem', fontWeight: 600, color: '#FFFFFF' }}>
            <Calendar size={17} color="var(--accent-amber)" />
            <span>Bilanço Koruma Kalkanı (De-risking)</span>
          </div>
          <span className="tabular" style={{ fontSize: '0.85rem', color: 'var(--accent-amber)', fontWeight: 700 }}>
            {data.days_to_earnings !== null ? `${data.days_to_earnings} Gün Kaldı` : 'Takvim Bekleniyor'}
          </span>
        </div>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          {data.earnings_regime}
        </p>
      </div>

      {/* Tarihsel Bilanço Sürprizleri */}
      {data.earnings_history && data.earnings_history.length > 0 && (
        <div>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'block', marginBottom: 10 }}>
            Tarihsel Bilanço & EPS Sürprizleri
          </span>
          <div className="screener-table-container">
            <table className="screener-table">
              <thead>
                <tr>
                  <th>Açıklanma Tarihi</th>
                  <th>Beklenen EPS</th>
                  <th>Açıklanan EPS</th>
                  <th>Sürpriz Sapması</th>
                </tr>
              </thead>
              <tbody>
                {data.earnings_history.map((rec, i) => (
                  <tr key={i} className="screener-row">
                    <td className="tabular">{rec.date}</td>
                    <td className="tabular">{rec.eps_estimate !== null ? `$${rec.eps_estimate.toFixed(2)}` : '—'}</td>
                    <td className="tabular" style={{ fontWeight: 600, color: '#FFFFFF' }}>
                      {rec.reported_eps !== null ? `$${rec.reported_eps.toFixed(2)}` : '—'}
                    </td>
                    <td>
                      <span className={`tag ${(rec.surprise_pct || 0) >= 0 ? 'tag-bull' : 'tag-bear'} tabular`}>
                        {rec.surprise_pct !== null ? `${rec.surprise_pct > 0 ? '+' : ''}%${rec.surprise_pct.toFixed(2)}` : '—'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
