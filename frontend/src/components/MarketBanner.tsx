import React from 'react';
import type { MarketData } from '../types';
import { ArrowUpRight, ArrowDownRight, Compass } from 'lucide-react';

interface MarketBannerProps {
  data: MarketData;
}

export const MarketBanner: React.FC<MarketBannerProps> = ({ data }) => {
  const isPositive = data.change_pct >= 0;

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '1.5rem 2rem',
        background: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-lg)',
        boxShadow: 'var(--shadow-card)',
      }}
    >
      {/* Varlık Başlığı ve Fiyat */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '2.5rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <h2 style={{ fontSize: '2rem', fontWeight: 800, color: '#FFFFFF', letterSpacing: '-0.02em' }}>
              {data.ticker}
            </h2>
            <span
              style={{
                fontSize: '0.72rem',
                color: 'var(--text-muted)',
                background: 'var(--bg-surface)',
                padding: '3px 8px',
                borderRadius: 6,
                border: '1px solid var(--border-subtle)',
              }}
            >
              Ref: {data.benchmark}
            </span>
          </div>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Kurumsal Kantitatif Görünüm
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
          <span className="tabular" style={{ fontSize: '2.2rem', fontWeight: 800, color: '#FFFFFF', letterSpacing: '-0.02em' }}>
            ${data.current_price.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </span>
          <span
            className={`tag ${isPositive ? 'tag-bull' : 'tag-bear'} tabular`}
            style={{ fontSize: '0.9rem', padding: '4px 10px', fontWeight: 700 }}
          >
            {isPositive ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
            %{Math.abs(data.change_pct).toFixed(2)}
          </span>
        </div>
      </div>

      {/* Kantitatif Göstergeler Şeridi */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
        <div>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'block' }}>
            200 SMA Mesafesi
          </span>
          <span
            className="tabular"
            style={{
              fontSize: '1.1rem',
              fontWeight: 700,
              color: data.is_above_sma200 ? 'var(--bull-text)' : 'var(--bear-text)',
            }}
          >
            {data.dist_sma200_pct >= 0 ? `+${data.dist_sma200_pct}%` : `${data.dist_sma200_pct}%`}
          </span>
        </div>

        <div>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'block' }}>
            Trend Rejimi
          </span>
          <span
            style={{
              fontSize: '1.05rem',
              fontWeight: 600,
              color: data.is_golden_cross ? 'var(--bull-text)' : 'var(--accent-amber)',
              display: 'flex',
              alignItems: 'center',
              gap: 5,
            }}
          >
            <Compass size={15} />
            {data.is_golden_cross ? 'Golden Cross' : 'Death Cross'}
          </span>
        </div>

        <div>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'block' }}>
            Sektörel Alfa (20G)
          </span>
          <span
            className="tabular"
            style={{
              fontSize: '1.1rem',
              fontWeight: 700,
              color: data.alpha_20d_cum >= 0 ? 'var(--bull-text)' : 'var(--bear-text)',
            }}
          >
            {data.alpha_20d_cum >= 0 ? `+${data.alpha_20d_cum}%` : `${data.alpha_20d_cum}%`}
          </span>
        </div>

        <div>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'block' }}>
            Piyasa Betası
          </span>
          <span className="tabular" style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            {data.beta}
          </span>
        </div>

        <div>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'block' }}>
            Hacim Anomalisi
          </span>
          <span
            className="tabular"
            style={{
              fontSize: '1.1rem',
              fontWeight: 700,
              color: data.volume_ratio > 1.2 ? 'var(--accent-amber)' : 'var(--text-secondary)',
            }}
          >
            {data.volume_ratio}x
          </span>
        </div>
      </div>
    </div>
  );
};
