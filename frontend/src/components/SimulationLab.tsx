import React, { useState, useEffect } from 'react';
import type { SimulationData, TradeEvent, SimulationStep } from '../types';

interface SimulationLabProps {
  simulation: SimulationData;
}

export const SimulationLab: React.FC<SimulationLabProps> = ({ simulation }) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isPlaying, setIsPlaying]               = useState(false);
  const [playbackSpeed, setPlaybackSpeed]       = useState<number>(2);
  const [selectedTrade, setSelectedTrade]       = useState<TradeEvent | null>(null);

  const timeline    = simulation.timeline;
  const totalSteps  = timeline.length;
  const currentStep: SimulationStep = timeline[currentStepIndex] || timeline[0];

  // Playback timer
  useEffect(() => {
    let timer: any = null;
    if (isPlaying) {
      const ms = Math.max(20, 380 / playbackSpeed);
      timer = setInterval(() => {
        setCurrentStepIndex((prev) => {
          if (prev >= totalSteps - 1) { setIsPlaying(false); return totalSteps - 1; }
          return prev + 1;
        });
      }, ms);
    }
    return () => { if (timer) clearInterval(timer); };
  }, [isPlaying, playbackSpeed, totalSteps]);

  const handleRestart = () => { setIsPlaying(false); setCurrentStepIndex(0); };

  const aiReturn  = ((currentStep.ai_equity    - simulation.initial_capital) / simulation.initial_capital) * 100;
  const bhReturn  = ((currentStep.buy_hold_equity - simulation.initial_capital) / simulation.initial_capital) * 100;

  // SVG geometry
  const svgW = 1080;
  const svgH = 360;
  const pad  = { top: 30, right: 40, bottom: 48, left: 80 };

  const activeTimeline = timeline.slice(0, currentStepIndex + 1);
  const allEquities    = timeline.flatMap(t => [t.ai_equity, t.buy_hold_equity]);
  const minEq = Math.min(...allEquities, 9200) * 0.975;
  const maxEq = Math.max(...allEquities, 10800) * 1.025;

  const getX = (i: number) => pad.left + (i / (totalSteps - 1)) * (svgW - pad.left - pad.right);
  const getY = (v: number) => svgH - pad.bottom - ((v - minEq) / (maxEq - minEq)) * (svgH - pad.top - pad.bottom);

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

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', animation: 'fadeUp 0.35s ease' }}>

      {/* ── PAGE HEADER ─────────────────────────────────────────────────── */}
      <div style={{
        padding: '1.5rem 0 0',
        borderTop: '3px solid var(--ink-primary)',
      }}>
        <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 6 }}>
          Kantitatif Araştırma Laboratuvarı
        </div>
        <h1 style={{ fontFamily: 'var(--font-display)', fontSize: '2rem', fontWeight: 700, color: 'var(--ink-primary)', letterSpacing: '-0.02em', lineHeight: 1 }}>
          10.000$ Dinamik Sermaye Simülasyonu
        </h1>
        <div style={{ fontSize: '0.85rem', color: 'var(--ink-secondary)', marginTop: 8, lineHeight: 1.6, maxWidth: 860 }}>
          {simulation.ticker} hissesi üzerinde 6 aylık körleme senaryosu. Yapay zeka modeli sermaye ağırlığını
          (0k / 2.5k / 5k / 7.5k / 10k) kantitatif güven skoruna göre dinamik belirlerken,
          karşı taraf tüm sermayeyi piyasada tutan bir al-tut yatırımcısı rolündedir.
        </div>
      </div>

      {/* ── CHART PANEL ─────────────────────────────────────────────────── */}
      <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)' }}>

        {/* Controls bar */}
        <div className="sim-controls">
          {/* Play / Pause */}
          <button className="sim-play-btn" onClick={() => setIsPlaying(!isPlaying)}>
            {isPlaying ? '⏸ Duraklat' : '▶ Oynat'}
          </button>

          {/* Restart */}
          <button className="btn btn-secondary" onClick={handleRestart} style={{ fontSize: '0.78rem', padding: '6px 14px' }}>
            ↺ Başa Al
          </button>

          {/* Speed selector */}
          <div style={{ display: 'flex', gap: 2, background: 'var(--paper-card)', padding: 2, borderRadius: 4, border: '1px solid var(--rule-strong)' }}>
            {[1, 2, 5, 10].map((s) => (
              <button
                key={s}
                onClick={() => setPlaybackSpeed(s)}
                style={{
                  padding: '4px 10px',
                  border: 'none',
                  borderRadius: 3,
                  background: playbackSpeed === s ? 'var(--ink-primary)' : 'transparent',
                  color: playbackSpeed === s ? 'var(--paper-card)' : 'var(--ink-muted)',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: 600,
                  fontSize: '0.72rem',
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
              >
                {s}×
              </button>
            ))}
          </div>

          {/* Date / scrubber */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flex: 1, marginLeft: '0.5rem' }}>
            <span className="tabular" style={{ fontSize: '0.8rem', color: 'var(--ink-secondary)', minWidth: 90, fontFamily: 'var(--font-mono)' }}>
              {currentStep.date}
            </span>
            <input
              type="range"
              min={0}
              max={totalSteps - 1}
              value={currentStepIndex}
              onChange={(e) => { setIsPlaying(false); setCurrentStepIndex(Number(e.target.value)); }}
              className="sim-scrubber"
              style={{ flex: 1 }}
            />
            <span className="tabular" style={{ fontSize: '0.75rem', color: 'var(--ink-muted)', minWidth: 65, fontFamily: 'var(--font-mono)' }}>
              {currentStepIndex + 1}/{totalSteps}G
            </span>
          </div>

          {/* Live equity readout */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', flexShrink: 0 }}>
            <div>
              <div style={{ fontSize: '0.58rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 2 }}>NeuroQuant AI</div>
              <div className="tabular" style={{ fontWeight: 700, fontSize: '0.95rem', color: aiReturn >= 0 ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
                {currentStep.ai_equity.toLocaleString('tr-TR', { minimumFractionDigits: 0 })}
                <span style={{ fontSize: '0.72rem', marginLeft: 5, opacity: 0.85 }}>
                  ({aiReturn >= 0 ? '+' : ''}{aiReturn.toFixed(2)}%)
                </span>
              </div>
            </div>
            <div style={{ width: 1, height: 28, background: 'var(--rule-strong)' }} />
            <div>
              <div style={{ fontSize: '0.58rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 2 }}>Al-Tut (Referans)</div>
              <div className="tabular" style={{ fontWeight: 600, fontSize: '0.95rem', color: 'var(--ink-secondary)' }}>
                {currentStep.buy_hold_equity.toLocaleString('tr-TR', { minimumFractionDigits: 0 })}
                <span style={{ fontSize: '0.72rem', marginLeft: 5, opacity: 0.85 }}>
                  ({bhReturn >= 0 ? '+' : ''}{bhReturn.toFixed(2)}%)
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* SVG Chart — paper/ivory background */}
        <div style={{ width: '100%', background: 'var(--paper-card)', borderBottom: '1px solid var(--rule-hairline)', overflow: 'hidden' }}>
          <svg viewBox={`0 0 ${svgW} ${svgH}`} style={{ width: '100%', height: 'auto', display: 'block' }}>

            {/* Y-axis grid lines */}
            {yTicks.map(({ val, y }, i) => (
              <g key={i}>
                <line x1={pad.left} y1={y} x2={svgW - pad.right} y2={y}
                  stroke="rgba(26,21,18,0.06)" strokeWidth="1" strokeDasharray="3,3" />
                <text x={pad.left - 8} y={y + 4}
                  fill="#8C827A" fontSize="10" textAnchor="end"
                  fontFamily="'JetBrains Mono', monospace">
                  {Math.round(val).toLocaleString('tr-TR')}
                </text>
              </g>
            ))}

            {/* 10k reference line */}
            <line x1={pad.left} y1={y10k} x2={svgW - pad.right} y2={y10k}
              stroke="rgba(26,21,18,0.2)" strokeDasharray="5,4" strokeWidth="1" />
            <text x={pad.left - 8} y={y10k - 4}
              fill="#57534E" fontSize="9.5" textAnchor="end"
              fontFamily="'JetBrains Mono', monospace" fontWeight="600">
              10.000 ₺
            </text>

            {/* Buy & Hold line — stone dashed */}
            <path d={bhPath} fill="none" stroke="#8C827A" strokeWidth="1.8"
              strokeDasharray="5,4" opacity="0.7" />

            {/* AI line — Forest Green */}
            <path d={aiPath} fill="none" stroke="#14532D" strokeWidth="3" />

            {/* Trade markers */}
            {simulation.trades.map((tr) => {
              if (tr.day_index > currentStepIndex) return null;
              const xPos = getX(tr.day_index);
              const yPos = getY(timeline[tr.day_index].ai_equity);
              const isBuy = tr.action === 'ALIM';
              return (
                <g key={tr.day_index} style={{ cursor: 'pointer' }} onClick={() => setSelectedTrade(tr)}>
                  <circle cx={xPos} cy={yPos} r={5.5}
                    fill={isBuy ? '#14532D' : '#881337'}
                    stroke="#FAF8F3" strokeWidth="1.5" />
                  {/* label */}
                  <rect
                    x={xPos - 18} y={isBuy ? yPos - 22 : yPos + 8}
                    width={36} height={14} rx={2}
                    fill={isBuy ? '#14532D' : '#881337'}
                    opacity="0.85"
                  />
                  <text
                    x={xPos} y={isBuy ? yPos - 12 : yPos + 19}
                    fill="#FAF8F3" fontSize="8" fontWeight="bold"
                    textAnchor="middle" fontFamily="'JetBrains Mono', monospace">
                    {tr.badge}
                  </text>
                </g>
              );
            })}

            {/* Live cursor */}
            {activeTimeline.length > 0 && (
              <circle cx={getX(currentStepIndex)} cy={getY(currentStep.ai_equity)}
                r={5} fill="#1E3A8A" stroke="#FAF8F3" strokeWidth="2" />
            )}

            {/* Legend */}
            <g transform={`translate(${svgW - pad.right - 180}, ${pad.top})`}>
              <rect x={0} y={0} width={160} height={44} rx={3}
                fill="#FAF8F3" stroke="#DFD7C8" strokeWidth="1" />
              <line x1={10} y1={14} x2={30} y2={14} stroke="#14532D" strokeWidth="3" />
              <text x={36} y={18} fill="#1A1512" fontSize="10"
                fontFamily="'Inter', sans-serif" fontWeight="600">NeuroQuant AI</text>
              <line x1={10} y1={32} x2={30} y2={32} stroke="#8C827A"
                strokeWidth="2" strokeDasharray="4,3" />
              <text x={36} y={36} fill="#57534E" fontSize="10"
                fontFamily="'Inter', sans-serif">Al-Tut (Referans)</text>
            </g>
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
              Hisse: {currentStep.ai_stock_value.toFixed(0)} ({currentStep.weight_pct}%)
            </span>
            <span className="tabular" style={{ fontSize: '0.85rem', color: 'var(--ink-secondary)' }}>
              Nakit: {currentStep.ai_cash_value.toFixed(0)} ({100 - currentStep.weight_pct}%)
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--ink-muted)' }}>Model Güveni:</span>
            <span className="tabular" style={{ fontWeight: 700, color: 'var(--cobalt)', fontSize: '0.9rem' }}>
              %{currentStep.confidence_score.toFixed(1)}
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
              Kurumsal Performans Tear-Sheet
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--ink-secondary)', marginTop: 2 }}>
              6 aylık simülasyon · Komisyon ve kayma maliyeti dahil
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
              label: 'AI Net Getiri',
              value: `+%${simulation.performance.ai_total_return_pct}`,
              sub: `Final: ${simulation.performance.ai_final_equity.toLocaleString('tr-TR')}`,
              color: 'var(--forest-gain)',
              left: 'var(--forest-gain)',
            },
            {
              label: 'Al-Tut Getiri',
              value: `+%${simulation.performance.buy_hold_total_return_pct}`,
              sub: `Final: ${simulation.performance.buy_hold_final_equity.toLocaleString('tr-TR')}`,
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
            Taktiksel Alım / Satım Kütüğü
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--ink-muted)' }}>
            {simulation.trades.length} işlem · Satıra tıklayın → Karar gerekçesi
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
                <td className="tabular">{tr.price.toFixed(2)}</td>
                <td className="tabular" style={{ color: 'var(--ink-secondary)' }}>
                  %{tr.prev_weight_pct} → %{tr.new_weight_pct}
                </td>
                <td className="tabular" style={{ color: 'var(--cobalt)', fontWeight: 700 }}>
                  %{tr.confidence_score.toFixed(1)}
                </td>
                <td className="tabular" style={{ color: 'var(--forest-gain)', fontWeight: 600 }}>
                  {tr.total_portfolio.toLocaleString('tr-TR')}
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
