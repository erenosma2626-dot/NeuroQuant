import React from 'react';
import type { MarketData } from '../types';

interface MarketBannerProps {
  data: MarketData;
}

export const MarketBanner: React.FC<MarketBannerProps> = ({ data }) => {
  const isPositive = data.change_pct >= 0;

  const Metric = ({ label, value, color }: { label: string; value: string; color?: string }) => (
    <div style={{ textAlign: 'center' }}>
      <div style={{
        fontSize: '0.6rem',
        fontWeight: 600,
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        color: 'var(--ink-muted)',
        marginBottom: 4,
      }}>
        {label}
      </div>
      <div className="tabular" style={{
        fontSize: '0.95rem',
        fontWeight: 700,
        color: color || 'var(--ink-primary)',
      }}>
        {value}
      </div>
    </div>
  );

  return (
    <div
      className="panel"
      style={{
        borderTop: '3px solid var(--ink-primary)',
        display: 'flex',
        alignItems: 'stretch',
      }}
    >
      {/* Asset Title Block */}
      <div style={{
        padding: '1.25rem 2rem',
        borderRight: '1px solid var(--rule-strong)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        minWidth: 200,
      }}>
        <div style={{
          fontFamily: 'var(--font-display)',
          fontSize: '1.75rem',
          fontWeight: 700,
          color: 'var(--ink-primary)',
          letterSpacing: '-0.01em',
          lineHeight: 1,
        }}>
          {data.ticker}
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--ink-secondary)', marginTop: 4 }}>
          Kantitatif Terminal · Ref: {data.benchmark}
        </div>
      </div>

      {/* Price Block */}
      <div style={{
        padding: '1.25rem 2rem',
        borderRight: '1px solid var(--rule-hairline)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        minWidth: 200,
      }}>
        <div className="tabular" style={{
          fontFamily: 'var(--font-display)',
          fontSize: '2rem',
          fontWeight: 700,
          color: 'var(--ink-primary)',
          lineHeight: 1,
        }}>
          {data.current_price.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </div>
        <div style={{ marginTop: 6 }}>
          <span className="tabular" style={{
            fontSize: '0.9rem',
            fontWeight: 700,
            color: isPositive ? 'var(--forest-gain)' : 'var(--madder-loss)',
          }}>
            {isPositive ? '▲' : '▼'} {Math.abs(data.change_pct).toFixed(2)}%
          </span>
          <span style={{ fontSize: '0.72rem', color: 'var(--ink-muted)', marginLeft: 8 }}>24 saatlik değişim</span>
        </div>
      </div>

      {/* Metrics Strip */}
      <div style={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-evenly',
        padding: '1.25rem 2rem',
        gap: '1rem',
      }}>
        <Metric
          label="200 SMA Mesafesi"
          value={`${data.dist_sma200_pct >= 0 ? '+' : ''}${data.dist_sma200_pct}%`}
          color={data.is_above_sma200 ? 'var(--forest-gain)' : 'var(--madder-loss)'}
        />
        <div style={{ width: 1, height: 32, background: 'var(--rule-hairline)' }} />
        <Metric
          label="Trend Rejimi"
          value={data.is_golden_cross ? 'Golden Cross' : 'Death Cross'}
          color={data.is_golden_cross ? 'var(--forest-gain)' : 'var(--madder-loss)'}
        />
        <div style={{ width: 1, height: 32, background: 'var(--rule-hairline)' }} />
        <Metric
          label="Sektörel Alfa (20G)"
          value={`${data.alpha_20d_cum >= 0 ? '+' : ''}${data.alpha_20d_cum}%`}
          color={data.alpha_20d_cum >= 0 ? 'var(--forest-gain)' : 'var(--madder-loss)'}
        />
        <div style={{ width: 1, height: 32, background: 'var(--rule-hairline)' }} />
        <Metric
          label="Piyasa Betası"
          value={String(data.beta)}
        />
        <div style={{ width: 1, height: 32, background: 'var(--rule-hairline)' }} />
        <Metric
          label="Hacim Anomalisi"
          value={`${data.volume_ratio}×`}
          color={data.volume_ratio > 1.2 ? 'var(--amber-warm)' : 'var(--ink-secondary)'}
        />
      </div>
    </div>
  );
};
