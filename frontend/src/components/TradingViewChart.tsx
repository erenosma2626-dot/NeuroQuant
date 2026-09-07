import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CandlestickSeries, LineSeries, HistogramSeries } from 'lightweight-charts';
import type { IChartApi } from 'lightweight-charts';
import type { MarketData } from '../types';
import { Layers } from 'lucide-react';

interface TradingViewChartProps {
  data: MarketData;
}

export const TradingViewChart: React.FC<TradingViewChartProps> = ({ data }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<IChartApi | null>(null);

  const [showSMA50, setShowSMA50] = useState(true);
  const [showSMA200, setShowSMA200] = useState(true);
  const [showBB, setShowBB] = useState(false);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    if (chartInstanceRef.current) {
      chartInstanceRef.current.remove();
      chartInstanceRef.current = null;
    }

    const container = chartContainerRef.current;
    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: '#0B0E14' },
        textColor: '#94A3B8',
        fontSize: 12,
        fontFamily: "'Inter', monospace",
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.03)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.03)' },
      },
      crosshair: {
        vertLine: { color: '#0EA5E9', width: 1, style: 2 },
        horzLine: { color: '#0EA5E9', width: 1, style: 2 },
      },
      timeScale: {
        borderColor: 'rgba(255, 255, 255, 0.06)',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: 'rgba(255, 255, 255, 0.06)',
      },
      width: container.clientWidth,
      height: 600,
    });

    chartInstanceRef.current = chart;

    // 1. Mum Grafiği (Candlesticks)
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10B981',
      downColor: '#F43F5E',
      borderVisible: false,
      wickUpColor: '#10B981',
      wickDownColor: '#F43F5E',
    });
    candleSeries.setData(data.candles as any);

    // 2. SMA 50 Çizgisi
    if (showSMA50 && data.sma50 && data.sma50.length > 0) {
      const sma50Series = chart.addSeries(LineSeries, {
        color: '#0EA5E9',
        lineWidth: 2,
        title: 'SMA 50',
      });
      sma50Series.setData(data.sma50 as any);
    }

    // 3. SMA 200 Çizgisi
    if (showSMA200 && data.sma200 && data.sma200.length > 0) {
      const sma200Series = chart.addSeries(LineSeries, {
        color: '#F59E0B',
        lineWidth: 2,
        title: 'SMA 200',
      });
      sma200Series.setData(data.sma200 as any);
    }

    // 4. Bollinger Bantları
    if (showBB && data.bb_upper && data.bb_lower) {
      const bbUpperSeries = chart.addSeries(LineSeries, {
        color: 'rgba(99, 102, 241, 0.5)',
        lineWidth: 1,
        lineStyle: 2,
      });
      bbUpperSeries.setData(data.bb_upper as any);

      const bbLowerSeries = chart.addSeries(LineSeries, {
        color: 'rgba(99, 102, 241, 0.5)',
        lineWidth: 1,
        lineStyle: 2,
      });
      bbLowerSeries.setData(data.bb_lower as any);
    }

    // 5. Hacim Histogramı
    if (data.volume_series && data.volume_series.length > 0) {
      const volumeSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: '', // Alt panel
      });
      volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.82, bottom: 0 },
      });
      volumeSeries.setData(data.volume_series as any);
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current && chartInstanceRef.current) {
        chartInstanceRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartInstanceRef.current) {
        chartInstanceRef.current.remove();
        chartInstanceRef.current = null;
      }
    };
  }, [data, showSMA50, showSMA200, showBB]);

  return (
    <div className="card" style={{ padding: '1.75rem 2rem' }}>
      <div className="card-header">
        <div className="card-title-group">
          <h2 className="card-title">
            <Layers size={19} color="var(--accent-sky)" />
            Akıcı Mum Grafiği & Çoklu İndikatör Katmanı
          </h2>
          <p className="card-subtitle">
            TradingView Lightweight Charts v5 motoru ile 50 ve 200 günlük hareketli ortalamalar.
          </p>
        </div>

        {/* Minimalist İndikatör Kontrolleri */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button
            className={`filter-btn ${showSMA50 ? 'active' : ''}`}
            onClick={() => setShowSMA50(!showSMA50)}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#0EA5E9' }} />
            SMA 50
          </button>
          <button
            className={`filter-btn ${showSMA200 ? 'active' : ''}`}
            onClick={() => setShowSMA200(!showSMA200)}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#F59E0B' }} />
            SMA 200
          </button>
          <button
            className={`filter-btn ${showBB ? 'active' : ''}`}
            onClick={() => setShowBB(!showBB)}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#6366F1' }} />
            Bollinger Bantları
          </button>
        </div>
      </div>

      <div
        ref={chartContainerRef}
        style={{
          width: '100%',
          height: 600,
          borderRadius: 'var(--radius-md)',
          overflow: 'hidden',
          border: '1px solid var(--border-subtle)',
        }}
      />
    </div>
  );
};
