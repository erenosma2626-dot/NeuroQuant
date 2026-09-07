import React, { useState, useEffect } from 'react';
import type { SimulationData, TradeEvent, SimulationStep } from '../types';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  X, 
  Shield, 
  Award, 
  ChevronRight
} from 'lucide-react';

interface SimulationLabProps {
  simulation: SimulationData;
}

export const SimulationLab: React.FC<SimulationLabProps> = ({ simulation }) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(2); // 1x, 2x, 5x, 10x
  const [selectedTrade, setSelectedTrade] = useState<TradeEvent | null>(null);

  const timeline = simulation.timeline;
  const totalSteps = timeline.length;
  const currentStep: SimulationStep = timeline[currentStepIndex] || timeline[0];

  // Animasyon Zamanlayıcısı
  useEffect(() => {
    let timer: any = null;
    if (isPlaying) {
      const intervalMs = Math.max(25, 400 / playbackSpeed);
      timer = setInterval(() => {
        setCurrentStepIndex((prev) => {
          if (prev >= totalSteps - 1) {
            setIsPlaying(false);
            return totalSteps - 1;
          }
          return prev + 1;
        });
      }, intervalMs);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isPlaying, playbackSpeed, totalSteps]);

  const handleRestart = () => {
    setIsPlaying(false);
    setCurrentStepIndex(0);
  };

  const aiReturnPct = (((currentStep.ai_equity - simulation.initial_capital) / simulation.initial_capital) * 100);
  const bhReturnPct = (((currentStep.buy_hold_equity - simulation.initial_capital) / simulation.initial_capital) * 100);

  // SVG Çizimi İçin Koordinat Hesaplamaları (Genişletilmiş Kanvas)
  const svgWidth = 1080;
  const svgHeight = 360;
  const padding = { top: 25, right: 40, bottom: 45, left: 75 };

  const activeTimeline = timeline.slice(0, currentStepIndex + 1);

  // Min ve Max değerler
  const allEquities = timeline.flatMap(t => [t.ai_equity, t.buy_hold_equity]);
  const minEquity = Math.min(...allEquities, 9200) * 0.98;
  const maxEquity = Math.max(...allEquities, 10800) * 1.02;

  const getX = (index: number) => {
    const usableWidth = svgWidth - padding.left - padding.right;
    return padding.left + (index / (totalSteps - 1)) * usableWidth;
  };

  const getY = (val: number) => {
    const usableHeight = svgHeight - padding.top - padding.bottom;
    return svgHeight - padding.bottom - ((val - minEquity) / (maxEquity - minEquity)) * usableHeight;
  };

  // Çizgi Yolları (SVG Paths)
  const aiPath = activeTimeline.reduce((acc, curr, idx) => {
    const x = getX(idx);
    const y = getY(curr.ai_equity);
    return idx === 0 ? `M ${x},${y}` : `${acc} L ${x},${y}`;
  }, '');

  const bhPath = activeTimeline.reduce((acc, curr, idx) => {
    const x = getX(idx);
    const y = getY(curr.buy_hold_equity);
    return idx === 0 ? `M ${x},${y}` : `${acc} L ${x},${y}`;
  }, '');

  // 10.000$ Başlangıç Çizgisi
  const y10k = getY(10000);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* 1. Üst Başlık */}
      <div>
        <span style={{ fontSize: '0.8rem', color: 'var(--bull-text)', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
          Kantitatif Simülasyon Deneyi
        </span>
        <h1 style={{ fontSize: '2rem', color: '#FFFFFF', marginTop: 4 }}>
          10.000$ Dinamik Sermaye Laboratuvarı ({simulation.ticker})
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginTop: 4, maxWidth: 850 }}>
          Yapay zekanın hisse ve nakit ağırlığını risk rejimine göre dinamik belirlediği ($0k, 2.5k, 5k, 7.5k, 10k$) 6 aylık körleme piyasa simülasyonu.
        </p>
      </div>

      {/* 2. Oynatıcı Kontrolleri ve Canlı Metrik Şeridi */}
      <div className="card" style={{ padding: '1.75rem 2rem' }}>
        <div className="simulation-hero-player">
          <div className="player-controls">
            <button className="play-toggle-btn" onClick={() => setIsPlaying(!isPlaying)}>
              {isPlaying ? <Pause size={17} /> : <Play size={17} fill="#090C10" />}
              {isPlaying ? 'DURAKLAT' : 'OYNAT'}
            </button>

            <button className="btn-secondary" onClick={handleRestart} title="Simülasyonu Başa Al">
              <RotateCcw size={15} /> Başa Sar
            </button>

            {/* Hız Seçici */}
            <div style={{ display: 'flex', gap: 4, background: 'var(--bg-card)', padding: 3, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
              {[1, 2, 5, 10].map((s) => (
                <button
                  key={s}
                  onClick={() => setPlaybackSpeed(s)}
                  style={{
                    padding: '4px 10px',
                    borderRadius: 6,
                    border: 'none',
                    background: playbackSpeed === s ? 'var(--bg-elevated)' : 'transparent',
                    color: playbackSpeed === s ? '#FFFFFF' : 'var(--text-muted)',
                    fontWeight: 600,
                    fontSize: '0.75rem',
                    cursor: 'pointer',
                  }}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>

          {/* Zaman Çubuğu (Scrubber) */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, flex: 1, margin: '0 2rem' }}>
            <span className="tabular" style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', minWidth: 95 }}>
              {currentStep.date}
            </span>
            <input
              type="range"
              min={0}
              max={totalSteps - 1}
              value={currentStepIndex}
              onChange={(e) => {
                setIsPlaying(false);
                setCurrentStepIndex(Number(e.target.value));
              }}
              className="scrubber-slider"
            />
            <span className="tabular" style={{ fontSize: '0.8rem', color: 'var(--text-muted)', minWidth: 70 }}>
              {currentStepIndex + 1} / {totalSteps}G
            </span>
          </div>

          {/* Anlık Portföy Büyüklükleri */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
            <div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', display: 'block' }}>
                NeuroQuant AI Portföyü
              </span>
              <div className="tabular" style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--bull-text)' }}>
                ${currentStep.ai_equity.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                <span style={{ fontSize: '0.8rem', marginLeft: 6 }}>
                  (%{aiReturnPct >= 0 ? `+${aiReturnPct.toFixed(2)}` : aiReturnPct.toFixed(2)})
                </span>
              </div>
            </div>

            <div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', display: 'block' }}>
                10k Sabit Al-Tut (Market)
              </span>
              <div className="tabular" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                ${currentStep.buy_hold_equity.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                <span style={{ fontSize: '0.8rem', marginLeft: 6 }}>
                  (%{bhReturnPct >= 0 ? `+${bhReturnPct.toFixed(2)}` : bhReturnPct.toFixed(2)})
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* 3. Tam Ekran Sinematik SVG Eğrisi */}
        <div style={{ width: '100%', overflowX: 'auto', background: '#0B0E14', borderRadius: 'var(--radius-md)', padding: '1.5rem 1rem', border: '1px solid var(--border-subtle)' }}>
          <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
            {/* 10.000$ Referans Çizgisi */}
            <line x1={padding.left} y1={y10k} x2={svgWidth - padding.right} y2={y10k} stroke="rgba(255,255,255,0.15)" strokeDasharray="3,3" strokeWidth="1" />
            <text x={padding.left - 10} y={y10k + 4} fill="#64748B" fontSize="11" textAnchor="end" fontFamily="monospace">
              $10.000
            </text>

            {/* Buy & Hold Çizgisi (Gri Kesikli) */}
            <path d={bhPath} fill="none" stroke="#64748B" strokeWidth="2" strokeDasharray="4,4" opacity="0.75" />

            {/* NeuroQuant AI Çizgisi (Zümrüt Yeşil İpeksi Çizgi) */}
            <path d={aiPath} fill="none" stroke="#10B981" strokeWidth="3.2" filter="drop-shadow(0 0 6px rgba(16, 185, 129, 0.35))" />

            {/* O ana kadarki AL / SAT İşaretleri (Tıklanabilir) */}
            {simulation.trades.map((tr) => {
              if (tr.day_index > currentStepIndex) return null;
              const xPos = getX(tr.day_index);
              const yPos = getY(timeline[tr.day_index].ai_equity);
              const isBuy = tr.action === 'ALIM';

              return (
                <g key={tr.day_index} style={{ cursor: 'pointer' }} onClick={() => setSelectedTrade(tr)}>
                  <circle
                    cx={xPos}
                    cy={yPos}
                    r={6}
                    fill={isBuy ? '#10B981' : '#F43F5E'}
                    stroke="#090C10"
                    strokeWidth="2"
                  />
                  <rect
                    x={xPos - 22}
                    y={isBuy ? yPos - 24 : yPos + 10}
                    width={44}
                    height={16}
                    rx={4}
                    fill="#0F141C"
                    stroke={isBuy ? '#10B981' : '#F43F5E'}
                    strokeWidth="1"
                  />
                  <text
                    x={xPos}
                    y={isBuy ? yPos - 13 : yPos + 22}
                    fill={isBuy ? '#10B981' : '#F43F5E'}
                    fontSize="9"
                    fontWeight="bold"
                    textAnchor="middle"
                    fontFamily="monospace"
                  >
                    {tr.badge}
                  </text>
                </g>
              );
            })}

            {/* Canlı Koşan İmleç (Marker) */}
            {activeTimeline.length > 0 && (
              <circle
                cx={getX(currentStepIndex)}
                cy={getY(currentStep.ai_equity)}
                r={6}
                fill="#0EA5E9"
                stroke="#FFFFFF"
                strokeWidth="2"
                filter="drop-shadow(0 0 8px #0EA5E9)"
              />
            )}
          </svg>
        </div>

        {/* 4. Canlı Günlük Tahsis & Güven Skoru */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '1.25rem', padding: '1rem 1.5rem', background: 'var(--bg-surface)', borderRadius: 'var(--radius-md)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
              O Anki Varlık Dağılımı:
            </span>
            <span className="tabular" style={{ fontSize: '0.9rem', color: 'var(--bull-text)' }}>
              Hisse: ${currentStep.ai_stock_value.toFixed(2)} (%{currentStep.weight_pct})
            </span>
            <span className="tabular" style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
              Nakit: ${currentStep.ai_cash_value.toFixed(2)} (%{100 - currentStep.weight_pct})
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Bileşik Model Güveni:</span>
            <span className="tabular" style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--accent-sky)' }}>
              %{currentStep.confidence_score.toFixed(1)}
            </span>
          </div>
        </div>
      </div>

      {/* 5. Kurumsal Skor Kartı (Tear Sheet) */}
      <div className="card" style={{ padding: '2rem' }}>
        <div className="card-header">
          <div className="card-title-group">
            <h2 className="card-title">
              <Award size={20} color="var(--accent-amber)" />
              Kurumsal Performans Metrikleri & İşlem Kütüğü
            </h2>
            <p className="card-subtitle">
              Modelin 6 aylık sürede aşırı işlem (overtrading) yapmadığının ve kayma/komisyon sonrası net getirisinin kanıtı.
            </p>
          </div>
          <span className="tabular" style={{ fontSize: '0.85rem', color: 'var(--text-muted)', background: 'var(--bg-surface)', padding: '4px 10px', borderRadius: 6 }}>
            {simulation.start_date} → {simulation.end_date} (126 İş Günü)
          </span>
        </div>

        {/* 4 Özet Kutu */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem', marginBottom: '2rem' }}>
          <div className="metric-pill" style={{ borderLeft: '3px solid var(--bull-text)' }}>
            <div className="metric-pill-label">AI Net Getiri</div>
            <div className="metric-pill-val tabular" style={{ color: 'var(--bull-text)' }}>
              +%{simulation.performance.ai_total_return_pct}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>
              Final: ${simulation.performance.ai_final_equity.toLocaleString()}
            </div>
          </div>

          <div className="metric-pill" style={{ borderLeft: '3px solid #64748B' }}>
            <div className="metric-pill-label">Al-Tut Net Getiri</div>
            <div className="metric-pill-val tabular" style={{ color: 'var(--text-secondary)' }}>
              +%{simulation.performance.buy_hold_total_return_pct}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>
              Final: ${simulation.performance.buy_hold_final_equity.toLocaleString()}
            </div>
          </div>

          <div className="metric-pill" style={{ borderLeft: '3px solid var(--accent-sky)' }}>
            <div className="metric-pill-label">Yıllık Sharpe Oranı</div>
            <div className="metric-pill-val tabular" style={{ color: 'var(--accent-sky)' }}>
              {simulation.performance.ai_sharpe}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>
              Sortino Oranı: {simulation.performance.ai_sortino}
            </div>
          </div>

          <div className="metric-pill" style={{ borderLeft: '3px solid var(--bear-text)' }}>
            <div className="metric-pill-label">Maksimum Çekilme (DD)</div>
            <div className="metric-pill-val tabular" style={{ color: 'var(--bull-text)' }}>
              %{simulation.performance.ai_max_drawdown_pct}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--bear-text)', marginTop: 4 }}>
              Piyasa Çekilmesi: %{simulation.performance.buy_hold_max_drawdown_pct}
            </div>
          </div>
        </div>

        {/* Taktiksel İşlem Kütüğü Tablosu */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Taktiksel Alım/Satım Kütüğü ({simulation.trades.length} İşlem Gerçekleştirildi)
          </span>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Satıra tıklayarak o kararın yapay zeka gerekçelerini sağ panelde açabilirsiniz.
          </span>
        </div>

        <div className="screener-table-container">
          <table className="screener-table">
            <thead>
              <tr>
                <th>Tarih</th>
                <th>İşlem / Büyüklük</th>
                <th>İcra Fiyatı</th>
                <th>Ağırlık Değişimi</th>
                <th>Güven Skoru</th>
                <th>Portföy Değeri</th>
                <th style={{ textAlign: 'right' }}>Gerekçe</th>
              </tr>
            </thead>
            <tbody>
              {simulation.trades.map((tr, idx) => (
                <tr
                  key={idx}
                  className="screener-row"
                  onClick={() => setSelectedTrade(tr)}
                >
                  <td className="tabular">{tr.date}</td>
                  <td>
                    <span className={`tag ${tr.action === 'ALIM' ? 'tag-bull' : 'tag-bear'}`}>
                      {tr.action} ({tr.badge})
                    </span>
                  </td>
                  <td className="tabular">${tr.price.toFixed(2)}</td>
                  <td className="tabular">%{tr.prev_weight_pct} → %{tr.new_weight_pct}</td>
                  <td className="tabular" style={{ color: 'var(--accent-sky)', fontWeight: 600 }}>
                    %{tr.confidence_score.toFixed(1)}
                  </td>
                  <td className="tabular" style={{ color: 'var(--bull-text)', fontWeight: 600 }}>
                    ${tr.total_portfolio.toLocaleString()}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <button className="btn-secondary" style={{ padding: '3px 10px', fontSize: '0.72rem' }}>
                      Gerekçeleri Gör <ChevronRight size={12} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 6. Sağdan Kayan Açıklanabilir Yapay Zeka Çekmecesi (Slide-Over Drawer) */}
      {selectedTrade && (
        <div className="drawer-overlay" onClick={() => setSelectedTrade(null)}>
          <div className="drawer-content" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span className={`tag ${selectedTrade.action === 'ALIM' ? 'tag-bull' : 'tag-bear'}`} style={{ fontSize: '0.85rem', padding: '4px 10px' }}>
                  {selectedTrade.action} ({selectedTrade.badge})
                </span>
                <span className="tabular" style={{ fontSize: '1.05rem', fontWeight: 700, color: '#FFFFFF' }}>
                  {selectedTrade.date}
                </span>
              </div>

              <button className="drawer-close-btn" onClick={() => setSelectedTrade(null)}>
                <X size={20} />
              </button>
            </div>

            <div className="drawer-body">
              {/* İşlem Özeti */}
              <div style={{ padding: '1.25rem', background: 'var(--bg-surface)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 14 }}>
                <div>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>İcra Fiyatı</span>
                  <div className="tabular" style={{ fontSize: '1.2rem', fontWeight: 700, color: '#FFFFFF' }}>
                    ${selectedTrade.price.toFixed(2)}
                  </div>
                </div>

                <div>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Bileşik Güven Skoru</span>
                  <div className="tabular" style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--accent-sky)' }}>
                    %{selectedTrade.confidence_score.toFixed(1)}
                  </div>
                </div>

                <div>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Portföy Pozisyonu</span>
                  <div className="tabular" style={{ fontSize: '0.95rem', fontWeight: 600 }}>
                    %{selectedTrade.prev_weight_pct} → %{selectedTrade.new_weight_pct}
                  </div>
                </div>

                <div>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Toplam Sermaye</span>
                  <div className="tabular" style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--bull-text)' }}>
                    ${selectedTrade.total_portfolio.toLocaleString()}
                  </div>
                </div>
              </div>

              {/* Model Karar Gerekçeleri (Explainable AI) */}
              <div>
                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'block', marginBottom: 12 }}>
                  Karara Dayanak Oluşturan Faktörler (XAI Breakdown)
                </span>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {selectedTrade.reasons.map((reason, rIdx) => (
                    <div
                      key={rIdx}
                      style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: 12,
                        padding: '12px 14px',
                        background: 'var(--bg-surface)',
                        borderRadius: 8,
                        border: '1px solid var(--border-subtle)',
                        fontSize: '0.85rem',
                        lineHeight: 1.5,
                      }}
                    >
                      <Shield size={16} color="var(--bull-text)" style={{ flexShrink: 0, marginTop: 3 }} />
                      <span style={{ color: 'var(--text-primary)' }}>{reason}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Bilgi Notu */}
              <div style={{ padding: '1rem', background: 'rgba(14, 165, 233, 0.05)', borderRadius: 8, border: '1px solid rgba(14, 165, 233, 0.15)', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                💡 <strong>Histerezis & Koruma:</strong> Bu işlem, minimum %25 ağırlık farkı eşiği ve 3 günlük asgari bekleme süresi filtresi onay verdikten sonra %0.10 komisyon ve kayma maliyeti kesilerek gerçekleştirilmiştir.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
