import React from 'react';
import type { ForecastData } from '../types';
import { Target, CheckCircle2, AlertTriangle } from 'lucide-react';

interface QuantMatrixProps {
  forecast: ForecastData;
}

export const QuantMatrix: React.FC<QuantMatrixProps> = ({ forecast }) => {
  const isUp = forecast.median_5d_return_pct >= 0;

  return (
    <div className="card" style={{ padding: '1.75rem 2rem' }}>
      <div className="card-header">
        <div className="card-title-group">
          <h2 className="card-title">
            <Target size={19} color="var(--bull-text)" />
            5-Günlük Olasılıksal Güven Konisi (Quantile LightGBM)
          </h2>
          <p className="card-subtitle">
            %80 olasılık bantları ($q_{10}, q_{50}, q_{90}$) ile tekil yanıltıcı tahminler yerine belirsizlik aralığı.
          </p>
        </div>
        <span className="tabular" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          {forecast.as_of_date}
        </span>
      </div>

      {/* Konsensüs ve Olasılık Rozeti */}
      <div
        style={{
          padding: '1.25rem 1.5rem',
          borderRadius: 'var(--radius-md)',
          background: 'var(--bg-surface)',
          border: '1px solid var(--border-subtle)',
          marginBottom: '1.5rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Model Konsensüs Kararı
          </span>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: forecast.decision_color, display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
            {isUp ? <CheckCircle2 size={20} /> : <AlertTriangle size={20} />}
            {forecast.decision}
          </div>
        </div>

        <div style={{ textAlign: 'right' }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Yükseliş Olasılığı
          </span>
          <div className="tabular" style={{ fontSize: '1.6rem', fontWeight: 800, color: forecast.up_probability >= 50 ? 'var(--bull-text)' : 'var(--bear-text)' }}>
            %{forecast.up_probability.toFixed(1)}
          </div>
        </div>
      </div>

      {/* 3 Kutulu Bant Özeti */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
        <div className="metric-pill" style={{ borderLeft: '3px solid var(--accent-sky)' }}>
          <div className="metric-pill-label">Medyan Beklenti (q50)</div>
          <div className="metric-pill-val tabular" style={{ color: isUp ? 'var(--bull-text)' : 'var(--bear-text)' }}>
            {forecast.median_5d_return_pct >= 0 ? `+${forecast.median_5d_return_pct}%` : `${forecast.median_5d_return_pct}%`}
          </div>
        </div>

        <div className="metric-pill" style={{ borderLeft: '3px solid var(--bull-text)' }}>
          <div className="metric-pill-label">Üst %80 Bant (q90)</div>
          <div className="metric-pill-val tabular" style={{ color: 'var(--bull-text)' }}>
            +{forecast.upper_80_return_pct}%
          </div>
        </div>

        <div className="metric-pill" style={{ borderLeft: '3px solid var(--bear-text)' }}>
          <div className="metric-pill-label">Alt %80 Bant (q10)</div>
          <div className="metric-pill-val tabular" style={{ color: 'var(--bear-text)' }}>
            {forecast.lower_80_return_pct}%
          </div>
        </div>
      </div>

      {/* 5 Günlük Projeksiyon Tablosu */}
      <div className="screener-table-container">
        <table className="screener-table">
          <thead>
            <tr>
              <th>Adım</th>
              <th>Tarih</th>
              <th>Medyan Hedef Fiyat</th>
              <th>%80 Konik Güven Aralığı</th>
            </tr>
          </thead>
          <tbody>
            {forecast.cone_series.map((step) => (
              <tr key={step.step} className="screener-row">
                <td className="tabular" style={{ color: 'var(--text-muted)' }}>+{step.step}G</td>
                <td className="tabular">{step.date}</td>
                <td className="tabular" style={{ color: isUp ? 'var(--bull-text)' : 'var(--bear-text)', fontWeight: 700 }}>
                  ${step.median_price.toFixed(2)}
                </td>
                <td className="tabular" style={{ color: 'var(--text-secondary)' }}>
                  ${step.lower_80_price.toFixed(2)} — ${step.upper_80_price.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
