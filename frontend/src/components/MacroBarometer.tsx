import React from 'react';
import type { MacroBarometerData } from '../types';

interface MacroBarometerProps {
  data: MacroBarometerData;
}

export const MacroBarometer: React.FC<MacroBarometerProps> = ({ data }) => {
  // Normalize composite score (-100 to +100) to gauge percentage (0% to 100%)
  const gaugePct = Math.min(100, Math.max(0, ((data.composite_score + 100) / 200) * 100));

  return (
    <div className="panel" style={{ borderTop: '3px solid var(--ink-primary)', animation: 'fadeUp 0.35s ease' }}>
      {/* ── HEADER ── */}
      <div style={{
        padding: '1.25rem 2rem',
        borderBottom: '2px solid var(--ink-primary)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 12,
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="section-label">Wall Street Barometresi</span>
            <span style={{
              fontSize: '0.65rem',
              fontWeight: 700,
              padding: '2px 8px',
              borderRadius: 3,
              background: 'rgba(30, 58, 138, 0.08)',
              color: 'var(--cobalt)',
              border: '1px solid rgba(30, 58, 138, 0.2)',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
            }}>
              Kantitatif RORO Modeli
            </span>
          </div>
          <div style={{
            fontFamily: 'var(--font-display)',
            fontSize: '1.25rem',
            fontWeight: 700,
            color: 'var(--ink-primary)',
            marginTop: 3,
          }}>
            Küresel Makro Rejim &amp; Risk İştahı Motoru
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span className="tabular" style={{ fontSize: '0.72rem', color: 'var(--ink-muted)', fontFamily: 'var(--font-mono)' }}>
            Son Güncelleme: {data.as_of}
          </span>
          <div className="status-pill">
            <span className="status-dot" />
            CANLI SENTEZ
          </div>
        </div>
      </div>

      {/* ── COMPOSITE REGIME SUMMARY STRIP ── */}
      <div style={{
        padding: '1.5rem 2rem',
        background: 'var(--paper-elevated)',
        borderBottom: '1px solid var(--rule-strong)',
        display: 'grid',
        gridTemplateColumns: '260px 1fr',
        gap: '2.5rem',
        alignItems: 'center',
      }}>
        {/* Score & Gauge Column */}
        <div style={{
          paddingRight: '2rem',
          borderRight: '1px solid var(--rule-hairline)',
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
        }}>
          <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--ink-muted)' }}>
            Bileşik Makro Skoru
          </div>

          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
            <div className="tabular" style={{
              fontFamily: 'var(--font-display)',
              fontSize: '2.4rem',
              fontWeight: 700,
              lineHeight: 1,
              color: data.regime_color,
            }}>
              {data.composite_score > 0 ? `+${data.composite_score}` : data.composite_score}
            </div>
            <span style={{ fontSize: '0.85rem', color: 'var(--ink-muted)', fontWeight: 600 }}>/ 100</span>
          </div>

          {/* RORO Slider Bar */}
          <div style={{ marginTop: 4 }}>
            <div style={{
              height: 7,
              width: '100%',
              background: 'linear-gradient(to right, #881337 0%, #D97706 40%, #57534E 50%, #1E3A8A 65%, #14532D 100%)',
              borderRadius: 4,
              position: 'relative',
              overflow: 'visible',
            }}>
              <div style={{
                position: 'absolute',
                top: -3,
                left: `${gaugePct}%`,
                transform: 'translateX(-50%)',
                width: 13,
                height: 13,
                background: '#FAF8F3',
                border: `3px solid ${data.regime_color}`,
                borderRadius: '50%',
                boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
                transition: 'left 0.4s ease',
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.62rem', color: 'var(--ink-muted)', marginTop: 6 }}>
              <span>-100 (Risk-Off)</span>
              <span>0 (Nötr)</span>
              <span>+100 (Risk-On)</span>
            </div>
          </div>
        </div>

        {/* Narrative & Regime Badge */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{
              padding: '3px 10px',
              borderRadius: 'var(--radius-xs)',
              background: data.regime_color,
              color: '#FAF8F3',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.75rem',
              fontWeight: 700,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
            }}>
              {data.regime_badge}
            </span>
            <span style={{ fontFamily: 'var(--font-display)', fontSize: '1.05rem', fontWeight: 700, color: 'var(--ink-primary)' }}>
              {data.regime_label}
            </span>
          </div>

          <p style={{ fontSize: '0.86rem', color: 'var(--ink-secondary)', lineHeight: 1.65, margin: 0 }}>
            {data.investor_note}
          </p>

          {/* Mathematical breakdown tags */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 4 }}>
            <span className="macro-chip">
              VIX Katkısı: <strong>{data.breakdown.vix_contribution > 0 ? '+' : ''}{data.breakdown.vix_contribution}</strong>
            </span>
            <span className="macro-chip">
              10Y Tahvil: <strong>{data.breakdown.tnx_contribution > 0 ? '+' : ''}{data.breakdown.tnx_contribution}</strong>
            </span>
            <span className="macro-chip">
              Petrol: <strong>{data.breakdown.oil_contribution > 0 ? '+' : ''}{data.breakdown.oil_contribution}</strong>
            </span>
            <span className="macro-chip">
              Altın/Sığınak: <strong>{data.breakdown.safe_haven_contribution > 0 ? '+' : ''}{data.breakdown.safe_haven_contribution}</strong>
            </span>
            <span className="macro-chip">
              Haber Duygusu: <strong>{data.breakdown.news_sentiment_contribution > 0 ? '+' : ''}{data.breakdown.news_sentiment_contribution}</strong>
            </span>
          </div>
        </div>
      </div>

      {/* ── 5 MACRO BAROMETER CARDS ── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(5, 1fr)',
        borderBottom: '1px solid var(--rule-strong)',
      }}>
        {data.barometers.map((b, idx) => {
          const isUp = b.change_pct >= 0;
          return (
            <div
              key={b.key}
              style={{
                padding: '1.25rem 1.5rem',
                borderRight: idx < data.barometers.length - 1 ? '1px solid var(--rule-hairline)' : 'none',
                background: 'var(--paper-card)',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                gap: 8,
                transition: 'background 0.15s ease',
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span className="tabular" style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.72rem',
                    fontWeight: 700,
                    color: 'var(--ink-primary)',
                    letterSpacing: '0.04em',
                  }}>
                    {b.key}
                  </span>
                  <span style={{
                    fontSize: '0.62rem',
                    fontWeight: 600,
                    padding: '1px 5px',
                    borderRadius: 2,
                    background: b.is_risk_on ? 'rgba(20, 83, 45, 0.08)' : 'rgba(136, 19, 55, 0.08)',
                    color: b.is_risk_on ? 'var(--forest-gain)' : 'var(--madder-loss)',
                  }}>
                    {b.is_risk_on ? 'BÜYÜME' : 'RİSK'}
                  </span>
                </div>

                <div style={{ fontSize: '0.72rem', color: 'var(--ink-secondary)', lineHeight: 1.3 }}>
                  {b.name}
                </div>
              </div>

              <div>
                <div className="tabular" style={{
                  fontFamily: 'var(--font-display)',
                  fontSize: '1.4rem',
                  fontWeight: 700,
                  color: 'var(--ink-primary)',
                  lineHeight: 1.1,
                }}>
                  {b.unit === '$' ? '$' : ''}{b.value.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}{b.unit !== '$' ? ` ${b.unit}` : ''}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
                  <span className={`tabular ${isUp ? 'change-up' : 'change-down'}`} style={{ fontSize: '0.76rem' }}>
                    {isUp ? '+' : ''}{b.change_pct.toFixed(2)}% (24s)
                  </span>
                  <span style={{ color: 'var(--rule-strong)' }}>·</span>
                  <span className="tabular" style={{ fontSize: '0.7rem', color: 'var(--ink-muted)' }}>
                    5G: {b.change_5d >= 0 ? '+' : ''}{b.change_5d.toFixed(1)}%
                  </span>
                </div>

                <div style={{
                  fontSize: '0.68rem',
                  color: 'var(--ink-secondary)',
                  marginTop: 6,
                  lineHeight: 1.3,
                  fontStyle: 'italic',
                }}>
                  {b.status}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* ── FOOTER PROTOCOL NOTE ── */}
      <div style={{
        padding: '0.65rem 2rem',
        fontSize: '0.68rem',
        color: 'var(--ink-muted)',
        fontFamily: 'var(--font-mono)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'var(--paper-card)',
      }}>
        <span>
          Ağırlıklar: VIX (%30), 10Y ABD Tahvil Getirisi (%20), WTI Petrol (%15), Ons Altın (%15), Üstel Küresel Haber Duygusu (%20).
        </span>
        <span style={{ color: 'var(--cobalt)', fontWeight: 600 }}>
          Wall Street Kurumsal Risk Sinyali
        </span>
      </div>
    </div>
  );
};
