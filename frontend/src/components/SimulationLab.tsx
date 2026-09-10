import React, { useState, useEffect } from 'react';
import type { SimulationData, TradeEvent, SimulationStep } from '../types';

interface SimulationLabProps {
  simulation: SimulationData;
}

export const SimulationLab: React.FC<SimulationLabProps> = ({ simulation }) => {
  const timeline    = simulation.timeline || [];
  const totalSteps  = timeline.length;

  const svgRef = React.useRef<SVGSVGElement>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Otomatik 2x oynatma: Simülasyon başladığında 0'dan başlar ve sona doğru 2x hızla akar
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [selectedTrade, setSelectedTrade] = useState<TradeEvent | null>(null);

  // Hisse veya zaman çizgisi değiştiğinde baştan otomatik başlat
  useEffect(() => {
    setSelectedTrade(null);
    setHoveredIndex(null);
    setCurrentStepIndex(0);
    setIsPlaying(true);
  }, [simulation.ticker, simulation.timeline?.length]);

  // Otomatik oynatma döngüsü (2x hız)
  useEffect(() => {
    let timer: any = null;
    if (isPlaying && totalSteps > 0) {
      const ms = 25; // Hızlı ve akıcı 2x geçiş
      timer = setInterval(() => {
        setCurrentStepIndex((prev) => {
          if (prev >= totalSteps - 1) {
            setIsPlaying(false);
            return totalSteps - 1;
          }
          return prev + 1;
        });
      }, ms);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isPlaying, totalSteps]);

  // Mouse grafik üzerindeyken hoveredIndex gösterilir, yoksa animasyon adımı
  const displayStepIndex = hoveredIndex !== null ? hoveredIndex : currentStepIndex;
  const activeStep: SimulationStep = timeline[displayStepIndex] || timeline[0] || {
    date: simulation.start_date || '',
    price: 0,
    ai_equity: simulation.initial_capital || 10000,
    buy_hold_equity: simulation.initial_capital || 10000,
    ai_cash_value: simulation.initial_capital || 10000,
    ai_stock_value: 0,
    weight_pct: 0,
    confidence_score: 50,
  };

  const aiReturn = ((activeStep.ai_equity - simulation.initial_capital) / simulation.initial_capital) * 100;
  const bhReturn = ((activeStep.buy_hold_equity - simulation.initial_capital) / simulation.initial_capital) * 100;

  // SVG geometry
  const svgW = 1080;
  const svgH = 360;
  const pad  = { top: 68, right: 40, bottom: 44, left: 75 };

  const activeTimeline = (isPlaying && hoveredIndex === null) ? timeline.slice(0, currentStepIndex + 1) : timeline;
  const allEquities    = timeline.flatMap(t => [t.ai_equity, t.buy_hold_equity]);
  const minEq = allEquities.length > 0 ? Math.min(...allEquities, 9200) * 0.975 : 9000;
  const maxEq = allEquities.length > 0 ? Math.max(...allEquities, 10800) * 1.025 : 11000;
  const span  = maxEq - minEq || 1;

  const getX = (i: number) => totalSteps <= 1 ? pad.left : pad.left + (i / (totalSteps - 1)) * (svgW - pad.left - pad.right);
  const getY = (v: number) => svgH - pad.bottom - ((v - minEq) / span) * (svgH - pad.top - pad.bottom);

  const aiPath = activeTimeline.reduce((acc, c, i) => {
    const x = getX(i); const y = getY(c.ai_equity);
    return i === 0 ? `M ${x},${y}` : `${acc} L ${x},${y}`;
  }, '');

  const bhPath = activeTimeline.reduce((acc, c, i) => {
    const x = getX(i); const y = getY(c.buy_hold_equity);
    return i === 0 ? `M ${x},${y}` : `${acc} L ${x},${y}`;
  }, '');

  const y10k = getY(10000);

  // Y-axis grid lines
  const yTicks = Array.from({ length: 5 }, (_, i) => {
    const val = minEq + ((maxEq - minEq) * i) / 4;
    return { val, y: getY(val) };
  });

  // Mouse ile grafik üzerinde gezinme
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current || totalSteps <= 1) return;
    const rect = svgRef.current.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const scaleX = svgW / rect.width;
    const svgX = clientX * scaleX;

    const plotLeft = pad.left;
    const plotRight = svgW - pad.right;
    const plotW = plotRight - plotLeft;

    const clampedX = Math.max(plotLeft, Math.min(plotRight, svgX));
    const ratio = (clampedX - plotLeft) / plotW;
    const stepIdx = Math.round(ratio * (totalSteps - 1));
    setHoveredIndex(Math.max(0, Math.min(totalSteps - 1, stepIdx)));
  };

  const handleMouseLeave = () => {
    setHoveredIndex(null);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', animation: 'fadeUp 0.35s ease' }}>

      {/* ── CHART PANEL (Başlık ve çubuk kaldırıldı, doğrudan grafikle başlıyor) ─────────────── */}
      <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)' }}>

        {/* SVG Chart Wrapper with Floating Stats in Top-Left */}
        <div style={{ position: 'relative', width: '100%', background: 'var(--paper-card)', borderBottom: '1px solid var(--rule-hairline)', overflow: 'hidden' }}>

          {/* Floating Return Badges & Date (Sol Üst Köşe) */}
          <div style={{
            position: 'absolute',
            top: '1rem',
            left: '1.25rem',
            display: 'flex',
            alignItems: 'center',
            gap: '1.25rem',
            background: 'rgba(250, 248, 243, 0.95)',
            backdropFilter: 'blur(8px)',
            padding: '8px 16px',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--rule-strong)',
            boxShadow: '0 2px 10px rgba(26,21,18,0.05)',
            zIndex: 10,
            pointerEvents: 'none',
          }}>
            {/* NeuroQuant AI */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                <span style={{ width: 12, height: 3, background: '#14532D', borderRadius: 1.5 }} />
                <span style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink-muted)' }}>
                  NeuroQuant AI
                </span>
              </div>
              <div className="tabular" style={{ fontWeight: 700, fontSize: '1.1rem', color: aiReturn >= 0 ? 'var(--forest-gain)' : 'var(--madder-loss)', lineHeight: 1.1 }}>
                ${activeStep.ai_equity.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                <span style={{ fontSize: '0.78rem', marginLeft: 5, fontWeight: 600 }}>
                  ({aiReturn >= 0 ? '+' : ''}{aiReturn.toFixed(2)}%)
                </span>
              </div>
            </div>

            <div style={{ width: 1, height: 30, background: 'var(--rule-strong)' }} />

            {/* Al-Tut (Referans) */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                <span style={{ width: 12, height: 2, borderTop: '2px dashed #8C827A' }} />
                <span style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink-muted)' }}>
                  Al-Tut (Referans)
                </span>
              </div>
              <div className="tabular" style={{ fontWeight: 600, fontSize: '1.1rem', color: 'var(--ink-secondary)', lineHeight: 1.1 }}>
                ${activeStep.buy_hold_equity.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                <span style={{ fontSize: '0.78rem', marginLeft: 5, fontWeight: 500 }}>
                  ({bhReturn >= 0 ? '+' : ''}{bhReturn.toFixed(2)}%)
                </span>
              </div>
            </div>

            <div style={{ width: 1, height: 30, background: 'var(--rule-strong)' }} />

            {/* Active Date */}
            <div>
              <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 2 }}>
                Tarih {hoveredIndex !== null ? '· İnceleme' : ''}
              </div>
              <div className="tabular" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem', fontWeight: 600, color: 'var(--ink-primary)', lineHeight: 1.1 }}>
                {activeStep.date}
              </div>
            </div>
          </div>

          {/* SVG Chart */}
          <svg
            ref={svgRef}
            viewBox={`0 0 ${svgW} ${svgH}`}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            style={{ width: '100%', height: 'auto', display: 'block', cursor: 'crosshair' }}
          >
            {/* Y-axis grid lines */}
            {yTicks.map(({ val, y }, i) => (
              <g key={i}>
                <line x1={pad.left} y1={y} x2={svgW - pad.right} y2={y}
                  stroke="rgba(26,21,18,0.06)" strokeWidth="1" strokeDasharray="3,3" />
                <text x={pad.left - 8} y={y + 4}
                  fill="#8C827A" fontSize="10" textAnchor="end"
                  fontFamily="'JetBrains Mono', monospace">
                  ${Math.round(val).toLocaleString('en-US')}
                </text>
              </g>
            ))}

            {/* 10k reference line */}
            <line x1={pad.left} y1={y10k} x2={svgW - pad.right} y2={y10k}
              stroke="rgba(26,21,18,0.2)" strokeDasharray="5,4" strokeWidth="1" />
            <text x={pad.left - 8} y={y10k - 4}
              fill="#57534E" fontSize="9.5" textAnchor="end"
              fontFamily="'JetBrains Mono', monospace" fontWeight="600">
              $10,000
            </text>

            {/* X-axis date labels */}
            {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => {
              const stepIdx = Math.min(totalSteps - 1, Math.round(pct * (totalSteps - 1)));
              if (stepIdx < 0 || !timeline[stepIdx]) return null;
              const xPos = getX(stepIdx);
              const anchor = idx === 0 ? 'start' : idx === 4 ? 'end' : 'middle';
              return (
                <text
                  key={idx}
                  x={xPos}
                  y={svgH - pad.bottom + 20}
                  fill="#8C827A"
                  fontSize="9"
                  textAnchor={anchor}
                  fontFamily="'JetBrains Mono', monospace"
                >
                  {timeline[stepIdx].date}
                </text>
              );
            })}

            {/* Buy & Hold line — stone dashed */}
            <path d={bhPath} fill="none" stroke="#8C827A" strokeWidth="1.8"
              strokeDasharray="5,4" opacity="0.7" />

            {/* AI line — Forest Green */}
            <path d={aiPath} fill="none" stroke="#14532D" strokeWidth="3" />

            {/* Trade markers */}
            {simulation.trades.map((tr) => {
              if (tr.day_index > displayStepIndex && isPlaying && hoveredIndex === null) return null;
              const xPos = getX(tr.day_index);
              const yPos = getY(timeline[tr.day_index]?.ai_equity ?? 10000);
              const isBuy = tr.action === 'ALIM' || tr.action.includes('ALIM');
              return (
                <g key={tr.day_index} style={{ cursor: 'pointer' }} onClick={() => setSelectedTrade(tr)}>
                  <title>{`${tr.date} · ${tr.action} (${tr.badge}) @ $${tr.price.toFixed(2)} — Tıklayarak gerekçeyi açın`}</title>
                  <circle cx={xPos} cy={yPos} r={4.5}
                    fill={isBuy ? '#14532D' : '#881337'}
                    stroke="#FAF8F3" strokeWidth="1.5" />
                </g>
              );
            })}

            {/* Hover Crosshair & Dots */}
            {hoveredIndex !== null && (
              <g pointerEvents="none">
                <line
                  x1={getX(displayStepIndex)}
                  y1={pad.top}
                  x2={getX(displayStepIndex)}
                  y2={svgH - pad.bottom}
                  stroke="#1E3A8A"
                  strokeWidth="1.2"
                  strokeDasharray="4,3"
                  opacity="0.65"
                />
                <circle
                  cx={getX(displayStepIndex)}
                  cy={getY(activeStep.ai_equity)}
                  r={5}
                  fill="#14532D"
                  stroke="#FAF8F3"
                  strokeWidth="2"
                />
                <circle
                  cx={getX(displayStepIndex)}
                  cy={getY(activeStep.buy_hold_equity)}
                  r={4}
                  fill="#8C827A"
                  stroke="#FAF8F3"
                  strokeWidth="1.5"
                />
              </g>
            )}

            {/* Live Playback cursor (yalnızca hover yokken ve oynatılırken) */}
            {hoveredIndex === null && isPlaying && activeTimeline.length > 0 && (
              <circle cx={getX(currentStepIndex)} cy={getY(activeStep.ai_equity)}
                r={5} fill="#1E3A8A" stroke="#FAF8F3" strokeWidth="2" />
            )}
          </svg>
        </div>

        {/* Allocation strip */}
        <div style={{
          padding: '0.85rem 2rem',
          borderBottom: '1px solid var(--rule-hairline)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'var(--paper-elevated)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
            <span style={{ fontSize: '0.7rem', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--ink-muted)' }}>
              Anlık Sermaye Dağılımı
            </span>
            <span className="tabular" style={{ fontSize: '0.85rem', color: 'var(--forest-gain)', fontWeight: 600 }}>
              Hisse: ${activeStep.ai_stock_value.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })} ({activeStep.weight_pct}%)
            </span>
            <span className="tabular" style={{ fontSize: '0.85rem', color: 'var(--ink-secondary)' }}>
              Nakit: ${activeStep.ai_cash_value.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })} ({100 - activeStep.weight_pct}%)
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--ink-muted)' }}>Model Güveni:</span>
            <span className="tabular" style={{ fontWeight: 700, color: 'var(--cobalt)', fontSize: '0.9rem' }}>
              %{activeStep.confidence_score.toFixed(1)}
            </span>
          </div>
        </div>
      </div>

      {/* ── PERFORMANCE SCORECARD ───────────────────────────────────────── */}
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
              Performans Özeti
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--ink-secondary)', marginTop: 2 }}>
              6 aylık simülasyon dönemi
            </div>
          </div>
          <span className="tabular" style={{
            fontSize: '0.7rem',
            color: 'var(--ink-muted)',
            fontFamily: 'var(--font-mono)',
            border: '1px solid var(--rule-strong)',
            padding: '3px 8px',
            borderRadius: 'var(--radius-xs)',
          }}>
            {simulation.start_date} → {simulation.end_date}
          </span>
        </div>

        {/* 4-column metrics */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)' }}>
          {[
            {
              label: 'Model Net Getiri',
              value: `+%${simulation.performance.ai_total_return_pct}`,
              sub: `Bakiye: $${simulation.performance.ai_final_equity.toLocaleString('en-US')}`,
              color: 'var(--forest-gain)',
              left: 'var(--forest-gain)',
            },
            {
              label: 'Al-Tut Getiri',
              value: `+%${simulation.performance.buy_hold_total_return_pct}`,
              sub: `Bakiye: $${simulation.performance.buy_hold_final_equity.toLocaleString('en-US')}`,
              color: 'var(--ink-secondary)',
              left: 'var(--ink-secondary)',
            },
            {
              label: 'Sharpe / Sortino',
              value: String(simulation.performance.ai_sharpe),
              sub: `Sortino: ${simulation.performance.ai_sortino}`,
              color: 'var(--cobalt)',
              left: 'var(--cobalt)',
            },
            {
              label: 'Maks. Çekilme (AI)',
              value: `%${simulation.performance.ai_max_drawdown_pct}`,
              sub: `Piyasa: %${simulation.performance.buy_hold_max_drawdown_pct}`,
              color: 'var(--madder-loss)',
              left: 'var(--madder-loss)',
            },
          ].map((m, i) => (
            <div
              key={i}
              className="metric-block"
              style={{ borderLeft: `3px solid ${m.left}`, borderRight: '1px solid var(--rule-hairline)' }}
            >
              <div className="metric-label">{m.label}</div>
              <div className="metric-value tabular" style={{ color: m.color, fontSize: '1.6rem' }}>{m.value}</div>
              <div className="metric-sub tabular">{m.sub}</div>
            </div>
          ))}
        </div>

        {/* Trade log */}
        <div style={{ padding: '0.75rem 2rem', borderTop: '1px solid var(--rule-strong)', borderBottom: '1px solid var(--rule-hairline)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: '0.9rem', fontWeight: 700, fontStyle: 'italic', color: 'var(--ink-primary)' }}>
            İşlem Geçmişi
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--ink-muted)' }}>
            {simulation.trades.length} işlem
          </div>
        </div>

        <table className="screener-table" style={{ width: '100%' }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '2rem' }}>Tarih</th>
              <th>İşlem</th>
              <th>Fiyat</th>
              <th>Ağırlık Değişimi</th>
              <th>Güven</th>
              <th>Portföy Değeri</th>
              <th style={{ paddingRight: '2rem', textAlign: 'right' }}>Gerekçe</th>
            </tr>
          </thead>
          <tbody>
            {simulation.trades.map((tr, idx) => (
              <tr
                key={idx}
                className="screener-row"
                onClick={() => setSelectedTrade(tr)}
              >
                <td className="tabular" style={{ paddingLeft: '2rem' }}>{tr.date}</td>
                <td>
                  <span className={`signal ${tr.action === 'ALIM' ? 'signal-buy' : 'signal-sell'}`}>
                    {tr.action} ({tr.badge})
                  </span>
                </td>
                <td className="tabular">${tr.price.toFixed(2)}</td>
                <td className="tabular" style={{ color: 'var(--ink-secondary)' }}>
                  %{tr.prev_weight_pct} → %{tr.new_weight_pct}
                </td>
                <td className="tabular" style={{ color: 'var(--cobalt)', fontWeight: 700 }}>
                  %{tr.confidence_score.toFixed(1)}
                </td>
                <td className="tabular" style={{ color: 'var(--forest-gain)', fontWeight: 600 }}>
                  ${tr.total_portfolio.toLocaleString('en-US')}
                </td>
                <td style={{ textAlign: 'right', paddingRight: '2rem' }}>
                  <button className="btn btn-secondary" style={{ padding: '3px 10px', fontSize: '0.72rem' }}>
                    Gerekçe →
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Table footer */}
        <div style={{ padding: '0.75rem 2rem', borderTop: '1px solid var(--rule-hairline)', fontSize: '0.7rem', color: 'var(--ink-muted)', fontStyle: 'italic' }}>
          Minimum %25 ağırlık farkı eşiği, 3 günlük bekleme süresi histerezis filtresi ve %0.10 komisyon + kayma maliyeti uygulanmıştır.
        </div>
      </div>

      {/* ── SLIDE-OVER DRAWER — XAI Gerekçe Paneli ────────────────────── */}
      {selectedTrade && (
        <div className="drawer-overlay" onClick={() => setSelectedTrade(null)}>
          <div className="drawer-content" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-header">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span className={`signal ${selectedTrade.action === 'ALIM' ? 'signal-buy' : 'signal-sell'}`} style={{ fontSize: '0.8rem', padding: '3px 10px' }}>
                    {selectedTrade.action} ({selectedTrade.badge})
                  </span>
                  <span className="tabular" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.88rem', fontWeight: 600, color: 'var(--ink-primary)' }}>
                    {selectedTrade.date}
                  </span>
                </div>
                <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', fontWeight: 700, fontStyle: 'italic', color: 'var(--ink-primary)' }}>
                  Karar Gerekçesi — XAI Analizi
                </div>
              </div>
              <button className="drawer-close-btn" onClick={() => setSelectedTrade(null)}>
                ✕ Kapat
              </button>
            </div>

            <div className="drawer-body">
              {/* Summary grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1px', background: 'var(--rule-hairline)', border: '1px solid var(--rule-hairline)' }}>
                {[
                  { label: 'İcra Fiyatı', value: selectedTrade.price.toFixed(2), color: 'var(--ink-primary)' },
                  { label: 'Model Güveni', value: `%${selectedTrade.confidence_score.toFixed(1)}`, color: 'var(--cobalt)' },
                  { label: 'Önceki Pozisyon', value: `%${selectedTrade.prev_weight_pct}`, color: 'var(--ink-secondary)' },
                  { label: 'Yeni Pozisyon', value: `%${selectedTrade.new_weight_pct}`, color: selectedTrade.action === 'ALIM' ? 'var(--forest-gain)' : 'var(--madder-loss)' },
                ].map((m, i) => (
                  <div key={i} style={{ padding: '1rem 1.25rem', background: 'var(--paper-card)' }}>
                    <div style={{ fontSize: '0.6rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 4 }}>
                      {m.label}
                    </div>
                    <div className="tabular" style={{ fontSize: '1.15rem', fontWeight: 700, color: m.color }}>
                      {m.value}
                    </div>
                  </div>
                ))}
              </div>

              {/* XAI reasons */}
              <div>
                <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 10 }}>
                  Karara Dayanak Faktörler
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  {selectedTrade.reasons.map((reason, i) => (
                    <div
                      key={i}
                      style={{
                        padding: '0.85rem 1rem 0.85rem 1.25rem',
                        borderBottom: '1px solid var(--rule-hairline)',
                        borderLeft: '3px solid var(--cobalt)',
                        background: i % 2 === 0 ? 'var(--paper-card)' : 'var(--paper-elevated)',
                        fontSize: '0.85rem',
                        color: 'var(--ink-secondary)',
                        lineHeight: 1.6,
                      }}
                    >
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--ink-muted)', marginRight: 8 }}>
                        [{String(i + 1).padStart(2, '0')}]
                      </span>
                      {reason}
                    </div>
                  ))}
                </div>
              </div>

              {/* Footnote */}
              <div style={{
                padding: '1rem',
                background: 'var(--paper-elevated)',
                border: '1px solid var(--rule-strong)',
                borderLeft: '3px solid var(--amber-warm)',
                fontSize: '0.78rem',
                color: 'var(--ink-muted)',
                lineHeight: 1.6,
              }}>
                <strong style={{ color: 'var(--amber-warm)' }}>Histerezis &amp; Overtrading Koruması:</strong>&nbsp;
                Bu işlem minimum %25 ağırlık farkı eşiği ve 3 günlük asgari bekleme süresi filtresi
                onayladıktan sonra %0.10 komisyon ve kayma maliyeti kesilerek gerçekleştirilmiştir.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
