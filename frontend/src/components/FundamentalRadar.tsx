import React from 'react';
import type { FundamentalsData } from '../types';

interface FundamentalRadarProps {
  data: FundamentalsData;
}

export const FundamentalRadar: React.FC<FundamentalRadarProps> = ({ data }) => {
  if (!data.is_equity) {
    return (
      <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)' }}>
        <div style={{ padding: '1rem 2rem', borderBottom: '2px solid var(--ink-primary)' }}>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', fontWeight: 700, fontStyle: 'italic', color: 'var(--ink-primary)' }}>
            Temel Değerleme &amp; Bilanço Radarı
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--ink-secondary)', marginTop: 2 }}>Varlık sınıfına göre değerleme yaklaşımı</div>
        </div>
        <div style={{ padding: '3rem 2rem', textAlign: 'center', color: 'var(--ink-secondary)' }}>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', fontWeight: 700, fontStyle: 'italic', color: 'var(--ink-primary)', marginBottom: 8 }}>
            Kripto / Emtia Rejimi
          </div>
          <p style={{ fontSize: '0.85rem', maxWidth: 460, margin: '0 auto', lineHeight: 1.65 }}>
            {data.ticker} için geleneksel hisse senedi çarpanları (F/K, F/DD) uygulanmaz.
            Model bunun yerine on-chain akış, volatilite ve sektörel korelasyona odaklanır.
          </p>
        </div>
      </div>
    );
  }

  const isCheap    = data.valuation_score >= 65;
  const isExpensive = data.valuation_score <= 40;

  const MultiplePill = ({ label, value }: { label: string; value: string }) => (
    <div style={{
      padding: '1rem 1.25rem',
      background: 'var(--paper-elevated)',
      borderBottom: '1px solid var(--rule-hairline)',
    }}>
      <div style={{ fontSize: '0.6rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 6 }}>
        {label}
      </div>
      <div className="tabular" style={{ fontFamily: 'var(--font-display)', fontSize: '1.35rem', fontWeight: 700, color: 'var(--ink-primary)' }}>
        {value}
      </div>
    </div>
  );

  return (
    <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)' }}>
      {/* Header */}
      <div style={{
        padding: '1rem 2rem',
        borderBottom: '2px solid var(--ink-primary)',
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
      }}>
        <div>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', fontWeight: 700, fontStyle: 'italic', color: 'var(--ink-primary)' }}>
            Temel Değerleme &amp; Bilanço Radarı
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--ink-secondary)', marginTop: 2 }}>
            Tarihsel çarpanlar ve akran grubuna göre ucuzluk/pahalılık skoru
          </div>
        </div>
        <span className={`signal ${isCheap ? 'signal-buy' : isExpensive ? 'signal-sell' : 'signal-cobalt'}`}>
          {data.valuation_status}
        </span>
      </div>

      {/* 4 Multiples in 2×2 grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1px', background: 'var(--rule-hairline)' }}>
        <MultiplePill label="F/K Trailing (P/E)" value={data.trailing_pe ? data.trailing_pe.toFixed(2) : '—'} />
        <MultiplePill label="İleri F/K (Forward P/E)" value={data.forward_pe ? data.forward_pe.toFixed(2) : '—'} />
        <MultiplePill label="F/DD (Price / Book)" value={data.price_to_book ? data.price_to_book.toFixed(2) : '—'} />
        <MultiplePill label="PEG Oranı" value={data.peg_ratio ? data.peg_ratio.toFixed(2) : '—'} />
      </div>

      {/* Earnings De-Risking Block */}
      <div style={{
        padding: '1.25rem 2rem',
        borderTop: '1px solid var(--rule-strong)',
        background: 'rgba(146, 64, 14, 0.04)',
        borderLeft: '3px solid var(--amber-warm)',
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        gap: '1rem',
      }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--amber-warm)', marginBottom: 6 }}>
            Bilanço Koruma Kalkanı
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--ink-secondary)', lineHeight: 1.6 }}>
            {data.earnings_regime}
          </p>
        </div>
        <div style={{ textAlign: 'right', flexShrink: 0 }}>
          <div style={{ fontSize: '0.6rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 4 }}>
            Sonraki Bilanço
          </div>
          <div className="tabular" style={{ fontWeight: 700, color: 'var(--amber-warm)', fontSize: '0.9rem' }}>
            {data.days_to_earnings !== null ? `${data.days_to_earnings} Gün Kaldı` : 'Bekleniyor'}
          </div>
        </div>
      </div>

      {/* Earnings History Table */}
      {data.earnings_history && data.earnings_history.length > 0 && (
        <div>
          <div style={{
            padding: '0.6rem 2rem',
            borderTop: '1px solid var(--rule-hairline)',
            borderBottom: '1px solid var(--rule-hairline)',
            fontSize: '0.62rem',
            fontWeight: 600,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            color: 'var(--ink-muted)',
          }}>
            Tarihsel EPS Sürprizleri
          </div>
          <table className="screener-table" style={{ width: '100%' }}>
            <thead>
              <tr>
                <th style={{ paddingLeft: '2rem' }}>Açıklanma</th>
                <th>Beklenen EPS</th>
                <th>Açıklanan EPS</th>
                <th style={{ paddingRight: '2rem' }}>Sürpriz</th>
              </tr>
            </thead>
            <tbody>
              {data.earnings_history.map((rec, i) => (
                <tr key={i}>
                  <td className="tabular" style={{ paddingLeft: '2rem' }}>{rec.date}</td>
                  <td className="tabular" style={{ color: 'var(--ink-secondary)' }}>
                    {rec.eps_estimate !== null ? rec.eps_estimate.toFixed(2) : '—'}
                  </td>
                  <td className="tabular" style={{ fontWeight: 600, color: 'var(--ink-primary)' }}>
                    {rec.reported_eps !== null ? rec.reported_eps.toFixed(2) : '—'}
                  </td>
                  <td style={{ paddingRight: '2rem' }}>
                    <span className={`signal ${(rec.surprise_pct || 0) >= 0 ? 'signal-buy' : 'signal-sell'} tabular`}>
                      {rec.surprise_pct !== null
                        ? `${rec.surprise_pct > 0 ? '+' : ''}${rec.surprise_pct.toFixed(2)}%`
                        : '—'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
