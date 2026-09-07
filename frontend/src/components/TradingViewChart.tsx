import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CandlestickSeries, LineSeries, HistogramSeries } from 'lightweight-charts';
import type { IChartApi } from 'lightweight-charts';
import type { MarketData } from '../types';

interface TradingViewChartProps {
  data: MarketData;
}

export const TradingViewChart: React.FC<TradingViewChartProps> = ({ data }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef     = useRef<IChartApi | null>(null);

  const [showSMA50,  setShowSMA50]  = useState(true);
  const [showSMA200, setShowSMA200] = useState(true);
  const [showBB,     setShowBB]     = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;
    if (chartRef.current) { chartRef.current.remove(); chartRef.current = null; }

    const container = containerRef.current;

    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: '#FAF8F3' }, // paper-card
        textColor: '#57534E',                                     // ink-secondary
        fontSize: 12,
        fontFamily: "'JetBrains Mono', 'Courier New', monospace",
      },
      grid: {
        vertLines: { color: 'rgba(26, 21, 18, 0.05)' },
        horzLines: { color: 'rgba(26, 21, 18, 0.05)' },
      },
      crosshair: {
        vertLine: { color: '#1E3A8A', width: 1, style: 2 },
        horzLine: { color: '#1E3A8A', width: 1, style: 2 },
      },
      timeScale: {
        borderColor: '#DFD7C8',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: '#DFD7C8',
        scaleMargins: { top: 0.15, bottom: 0.12 },
      },
      width:  container.clientWidth,
      height: 580,
    });

    chartRef.current = chart;

    // ── Candlesticks: Forest Green / Madder Red ──────────────────
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor:      '#14532D',   // forest-gain
      downColor:    '#881337',   // madder-loss
      borderVisible: false,
      wickUpColor:  '#166534',
      wickDownColor:'#9F1239',
    });
    candleSeries.setData(data.candles as any);

    // ── SMA 50: Oxford Cobalt ────────────────────────────────────
    if (showSMA50 && data.sma50?.length > 0) {
      const s50 = chart.addSeries(LineSeries, {
        color:     '#1E3A8A',
        lineWidth: 2,
        title:     'SMA 50',
      });
      s50.setData(data.sma50 as any);
    }

    // ── SMA 200: Amber Warm ──────────────────────────────────────
    if (showSMA200 && data.sma200?.length > 0) {
      const s200 = chart.addSeries(LineSeries, {
        color:     '#92400E',
        lineWidth: 2,
        title:     'SMA 200',
      });
      s200.setData(data.sma200 as any);
    }

    // ── Bollinger Bands: Cobalt dashed ───────────────────────────
    if (showBB && data.bb_upper && data.bb_lower) {
      const bbU = chart.addSeries(LineSeries, {
        color:     'rgba(30, 58, 138, 0.45)',
        lineWidth: 1,
        lineStyle: 2,
      });
      bbU.setData(data.bb_upper as any);

      const bbL = chart.addSeries(LineSeries, {
        color:     'rgba(30, 58, 138, 0.45)',
        lineWidth: 1,
        lineStyle: 2,
      });
      bbL.setData(data.bb_lower as any);
    }

    // ── Volume Histogram ─────────────────────────────────────────
    if (data.volume_series?.length > 0) {
      const volSeries = chart.addSeries(HistogramSeries, {
        priceFormat:  { type: 'volume' },
        priceScaleId: '',
      });
      volSeries.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
      volSeries.setData(data.volume_series as any);
    }

    chart.timeScale().fitContent();

    const onResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', onResize);

    return () => {
      window.removeEventListener('resize', onResize);
      chartRef.current?.remove();
      chartRef.current = null;
    };
  }, [data, showSMA50, showSMA200, showBB]);

  const ToggleBtn = ({
    active, onToggle, color, label,
  }: { active: boolean; onToggle: () => void; color: string; label: string }) => (
    <button
      className={`filter-btn ${active ? 'active' : ''}`}
      onClick={onToggle}
      style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.78rem' }}
    >
      <span style={{
        width: 10,
        height: 3,
        borderRadius: 2,
        background: color,
        opacity: active ? 1 : 0.4,
      }} />
      {label}
    </button>
  );

  return (
    <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)' }}>
      {/* Panel header */}
      <div style={{
        padding: '1rem 2rem',
        borderBottom: '2px solid var(--ink-primary)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div>
          <div style={{
            fontFamily: 'var(--font-display)',
            fontSize: '1rem',
            fontWeight: 700,
            fontStyle: 'italic',
            color: 'var(--ink-primary)',
          }}>
            Fiyat Grafiği &amp; İndikatör Katmanları
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--ink-secondary)', marginTop: 2 }}>
            TradingView Lightweight Charts · 50 ve 200 günlük hareketli ortalamalar
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <ToggleBtn active={showSMA50}  onToggle={() => setShowSMA50(!showSMA50)}   color="#1E3A8A" label="SMA 50"  />
          <ToggleBtn active={showSMA200} onToggle={() => setShowSMA200(!showSMA200)} color="#92400E" label="SMA 200" />
          <ToggleBtn active={showBB}     onToggle={() => setShowBB(!showBB)}         color="rgba(30,58,138,0.6)" label="Bollinger" />
        </div>
      </div>

      {/* Chart canvas */}
      <div
        ref={containerRef}
        style={{ width: '100%', height: 580, overflow: 'hidden' }}
      />
    </div>
  );
};
