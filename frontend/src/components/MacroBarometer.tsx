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
      {/* ── HEADER & REJİM BANNER (FERAH & MİNİMALİST) ── */}
      <div style={{
        padding: '1.75rem 2.5rem',
        borderBottom: '1px solid var(--rule-hairline)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '2rem',
      }}>
        {/* Sol: Başlık & Rejim */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
          <div>
            <div className="section-label" style={{ marginBottom: 4 }}>
              Wall Street Makro Barometresi &middot; {data.as_of}
            </div>
            <div style={{
              fontFamily: 'var(--font-display)',
              fontSize: '1.45rem',
              fontWeight: 700,
              color: 'var(--ink-primary)',
              letterSpacing: '-0.01em',
            }}>
              Küresel Risk-On Rejim Göstergesi
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{
              padding: '6px 14px',
              borderRadius: 'var(--radius-xs)',
              background: data.regime_color,
              color: '#FAF8F3',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.8rem',
              fontWeight: 700,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
            }}>
              {data.regime_badge}
            </span>
            <span style={{
              fontFamily: 'var(--font-display)',
              fontSize: '1.15rem',
              fontWeight: 600,
              fontStyle: 'italic',
              color: 'var(--ink-secondary)',
            }}>
              {data.regime_label}
            </span>
          </div>
        </div>

        {/* Sağ: RORO Göstergesi */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <div style={{ textAlign: 'right' }}>
            <div className="section-label" style={{ marginBottom: 2 }}>RORO Endeksi</div>
            <div className="tabular" style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '1.5rem',
              fontWeight: 700,
              color: data.regime_color,
            }}>
              {data.composite_score > 0 ? `+${data.composite_score}` : data.composite_score}
              <span style={{ fontSize: '0.85rem', color: 'var(--ink-muted)', fontWeight: 400 }}> / 100</span>
            </div>
          </div>

          {/* Minimalist Gauge Bar */}
          <div style={{ width: 140 }}>
            <div style={{
              height: 6,
              width: '100%',
              background: 'linear-gradient(to right, #881337 0%, #D97706 40%, #57534E 50%, #1E3A8A 65%, #14532D 100%)',
              borderRadius: 3,
              position: 'relative',
            }}>
              <div style={{
                position: 'absolute',
                top: -3,
                left: `${gaugePct}%`,
                transform: 'translateX(-50%)',
                width: 12,
                height: 12,
                background: '#FAF8F3',
                border: `2.5px solid ${data.regime_color}`,
                borderRadius: '50%',
                boxShadow: '0 1px 3px rgba(0,0,0,0.25)',
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.62rem', color: 'var(--ink-muted)', marginTop: 4 }}>
              <span>Defansif</span>
              <span>Büyüme</span>
            </div>
          </div>
        </div>
      </div>

      {/* ── 5 FERAH MAKRO BAROMETRE KARTI ── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(5, 1fr)',
        background: 'var(--paper-card)',
      }}>
        {data.barometers.map((b, idx) => {
          const isUp = b.change_pct >= 0;
          return (
            <div
              key={b.key}
              style={{
                padding: '1.75rem 1.75rem',
                borderRight: idx < data.barometers.length - 1 ? '1px solid var(--rule-hairline)' : 'none',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                minHeight: 160,
                transition: 'background 0.15s ease',
              }}
            >
              {/* Üst Kısım: Sembol & Kategori Rozeti */}
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span className="tabular" style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    color: 'var(--ink-primary)',
                    letterSpacing: '0.04em',
                  }}>
                    {b.symbol || b.key}
                  </span>
                  <span style={{
                    fontSize: '0.62rem',
                    fontWeight: 600,
                    padding: '2px 6px',
                    borderRadius: 'var(--radius-xs)',
                    background: b.is_risk_on ? 'var(--forest-tint)' : 'var(--madder-tint)',
                    color: b.is_risk_on ? 'var(--forest-gain)' : 'var(--madder-loss)',
                    letterSpacing: '0.04em',
                  }}>
                    {b.tag}
                  </span>
                </div>

                <div style={{ fontSize: '0.78rem', color: 'var(--ink-secondary)', fontWeight: 500 }}>
                  {b.name}
                </div>
              </div>

              {/* Alt Kısım: Sayısal Değer & Yapısal Metrik */}
              <div style={{ marginTop: '1rem' }}>
                <div className="tabular" style={{
                  fontFamily: 'var(--font-display)',
                  fontSize: '1.65rem',
                  fontWeight: 700,
                  color: 'var(--ink-primary)',
                  lineHeight: 1.1,
                }}>
                  {b.unit === '$' ? '$' : ''}{b.value.toLocaleString('en-US', { minimumFractionDigits: b.unit === '%' ? 3 : 2, maximumFractionDigits: 3 })}{b.unit !== '$' ? ` ${b.unit}` : ''}
                </div>

                {/* Dinamik Alt İndikatör (Gereksiz kalabalık olmadan) */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6 }}>
                  {b.key === 'TNX' ? (
                    <span className="tabular" style={{ fontSize: '0.72rem', color: b.trend_60d && b.trend_60d > 0 ? 'var(--madder-loss)' : 'var(--forest-gain)', fontWeight: 600 }}>
                      60G İvme: {b.trend_60d && b.trend_60d > 0 ? '+' : ''}{b.trend_60d}% &middot; {b.sma200 ? `SMA200: ${b.sma200}` : ''}
                    </span>
                  ) : b.key === 'FED' ? (
                    <span className="tabular" style={{ fontSize: '0.72rem', color: 'var(--cobalt)', fontWeight: 600 }}>
                      3M Bono: %{b.irx_rate?.toFixed(2)} &middot; {b.fed_badge}
                    </span>
                  ) : b.key === 'FEAR_GREED' ? (
                    <span className="tabular" style={{ fontSize: '0.72rem', color: isUp ? 'var(--forest-gain)' : 'var(--madder-loss)', fontWeight: 600 }}>
                      Haftalık: {isUp ? '+' : ''}{b.change_pct}% &middot; {b.rating}
                    </span>
                  ) : (
                    <span className={`tabular ${isUp ? 'change-up' : 'change-down'}`} style={{ fontSize: '0.74rem', fontWeight: 600 }}>
                      {isUp ? '+' : ''}{b.change_pct.toFixed(2)}% (24s)
                    </span>
                  )}
                </div>

                <div style={{
                  fontSize: '0.72rem',
                  color: 'var(--ink-muted)',
                  marginTop: 4,
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
    </div>
  );
};

export default MacroBarometer;
