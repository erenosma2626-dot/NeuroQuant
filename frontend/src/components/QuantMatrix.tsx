import React from 'react';
import type { ForecastData } from '../types';

interface QuantMatrixProps {
  forecast: ForecastData;
}

export const QuantMatrix: React.FC<QuantMatrixProps> = ({ forecast }) => {
  const isUp = forecast.median_5d_return_pct >= 0;

  const BandCell = ({
    label, value, color, leftBorder,
  }: { label: string; value: string; color: string; leftBorder: string }) => (
    <div style={{
      padding: '1rem 1.25rem',
      borderLeft: `3px solid ${leftBorder}`,
      background: 'var(--paper-elevated)',
    }}>
      <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 6 }}>
        {label}
      </div>
      <div className="tabular" style={{ fontFamily: 'var(--font-display)', fontSize: '1.4rem', fontWeight: 700, color }}>
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
            5-Günlük Olasılıksal Güven Konisi
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--ink-secondary)', marginTop: 2 }}>
            Quantile LightGBM · %80 güven bantları (q10, q50, q90)
          </div>
        </div>
        <span className="tabular" style={{ fontSize: '0.72rem', color: 'var(--ink-muted)', marginTop: 2 }}>
          {forecast.as_of_date}
        </span>
      </div>

      {/* Consensus + Probability */}
      <div style={{
        padding: '1.25rem 2rem',
        borderBottom: '1px solid var(--rule-hairline)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div>
          <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 4 }}>
            Model Konsensüs Kararı
          </div>
          <div style={{
            fontFamily: 'var(--font-display)',
            fontSize: '1.5rem',
            fontWeight: 700,
            color: isUp ? 'var(--forest-gain)' : 'var(--madder-loss)',
            letterSpacing: '-0.01em',
          }}>
            {isUp ? '▲' : '▼'} {forecast.decision}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 4 }}>
            Yükseliş Olasılığı
          </div>
          <div className="tabular" style={{
            fontFamily: 'var(--font-display)',
            fontSize: '2rem',
            fontWeight: 700,
            color: forecast.up_probability >= 50 ? 'var(--forest-gain)' : 'var(--madder-loss)',
          }}>
            %{forecast.up_probability.toFixed(1)}
          </div>
        </div>
      </div>

      {/* 3-band strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1px', background: 'var(--rule-hairline)', margin: '0 0' }}>
        <BandCell
          label="Medyan Beklenti (q50)"
          value={`${forecast.median_5d_return_pct >= 0 ? '+' : ''}${forecast.median_5d_return_pct}%`}
          color={isUp ? 'var(--forest-gain)' : 'var(--madder-loss)'}
          leftBorder="var(--cobalt)"
        />
        <BandCell
          label="Üst %80 Bant (q90)"
          value={`+${forecast.upper_80_return_pct}%`}
          color="var(--forest-gain)"
          leftBorder="var(--forest-gain)"
        />
        <BandCell
          label="Alt %80 Bant (q10)"
          value={`${forecast.lower_80_return_pct}%`}
          color="var(--madder-loss)"
          leftBorder="var(--madder-loss)"
        />
      </div>

      {/* Projection table */}
      <div style={{ padding: '0 0 0 0' }}>
        <table className="screener-table" style={{ width: '100%' }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '2rem' }}>Adım</th>
              <th>Tarih</th>
              <th>Medyan Hedef Fiyat</th>
              <th style={{ paddingRight: '2rem' }}>%80 Güven Aralığı</th>
            </tr>
          </thead>
          <tbody>
            {forecast.cone_series.map((step) => (
              <tr key={step.step}>
                <td className="tabular" style={{ color: 'var(--ink-muted)', paddingLeft: '2rem' }}>+{step.step}G</td>
                <td className="tabular" style={{ color: 'var(--ink-secondary)' }}>{step.date}</td>
                <td className="tabular" style={{ fontWeight: 700, color: isUp ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
                  {step.median_price.toFixed(2)}
                </td>
                <td className="tabular" style={{ color: 'var(--ink-muted)', paddingRight: '2rem' }}>
                  {step.lower_80_price.toFixed(2)} — {step.upper_80_price.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
