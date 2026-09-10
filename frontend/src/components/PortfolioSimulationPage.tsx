import React, { useState, useMemo, useRef, useEffect } from 'react';
import { 
  ArrowLeft, 
  Play, 
  DollarSign, 
  Sliders, 
  Trash2, 
  Activity,
  AlertCircle,
  ArrowUpRight,
  ArrowDownRight,
  X
} from 'lucide-react';
import type { 
  ScreenerItem, 
  UserPortfolio, 
  PortfolioSimResponse, 
  PortfolioSimBasketItem,
  PortfolioSimTrade
} from '../types';

interface PortfolioSimulationPageProps {
  onBack: () => void;
  userPortfolio?: UserPortfolio;
  screenerData?: ScreenerItem[];
}

const SIGNATURE_COLORS = [
  '#2563EB', // Cobalt Blue
  '#7C3AED', // Violet Purple
  '#D97706', // Amber Orange
  '#DB2777', // Pink Rose
  '#0D9488', // Teal
  '#EA580C', // Orange
];

const DEFAULT_BASKET: PortfolioSimBasketItem[] = [
  { ticker: 'NVDA', weight_pct: 35 },
  { ticker: 'AAPL', weight_pct: 25 },
  { ticker: 'MSFT', weight_pct: 20 },
];

const POPULAR_TICKERS = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'BTC-USD', 'AMD', 'JPM'];

export const PortfolioSimulationPage: React.FC<PortfolioSimulationPageProps> = ({
  onBack,
  userPortfolio,
  screenerData = [],
}) => {
  const suggestedTickers = useMemo(() => {
    if (screenerData && screenerData.length > 0) {
      return screenerData.slice(0, 10).map((s) => s.ticker);
    }
    return POPULAR_TICKERS;
  }, [screenerData]);

  // Sepet State
  const [basket, setBasket] = useState<PortfolioSimBasketItem[]>(() => {
    if (userPortfolio && userPortfolio.positions.length > 0) {
      const top3 = userPortfolio.positions.slice(0, 3);
      const totalW = top3.reduce((s, p) => s + p.weight_pct, 0);
      if (totalW > 0) {
        return top3.map(p => ({
          ticker: p.ticker,
          weight_pct: Math.round((p.weight_pct / totalW) * 75)
        }));
      }
    }
    return DEFAULT_BASKET;
  });

  const [cashWeight, setCashWeight] = useState<number>(20);
  const [maxCashPct, setMaxCashPct] = useState<number>(60);
  const [initialCapital] = useState<number>(100000);
  const [rebalanceInterval, setRebalanceInterval] = useState<number>(7);

  // Yeni ticker ekleme inputu
  const [newTickerInput, setNewTickerInput] = useState<string>('');

  // Simülasyon Çalışma Durumu
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [simResult, setSimResult] = useState<PortfolioSimResponse | null>(null);

  // Grafik Etkileşim State'leri
  const [showAiPort, setShowAiPort] = useState<boolean>(true);
  const [showBenchmark, setShowBenchmark] = useState<boolean>(true); // Başlangıçtaki Al-Tut ağırlıkları varsayılan olarak açık
  const [activeTickers, setActiveTickers] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<'return' | 'equity'>('return'); // 'return' = % Getiri, 'equity' = $ Değer
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const chartRef = useRef<SVGSVGElement | null>(null);
  const [selectedTradeCluster, setSelectedTradeCluster] = useState<{
    stepIdx: number;
    date: string;
    x: number;
    y: number;
    trades: PortfolioSimTrade[];
  } | null>(null);

  // Toplam Ağırlık Hesabı
  const totalStockWeight = useMemo(() => {
    return basket.reduce((sum, item) => sum + (Number(item.weight_pct) || 0), 0);
  }, [basket]);
  const totalAllocated = totalStockWeight + (Number(cashWeight) || 0);

  // Simülasyonu Başlat
  const runSimulation = async () => {
    if (basket.length === 0) {
      setError('Lütfen sepete en az bir hisse ekleyin.');
      return;
    }
    if (basket.length > 6) {
      setError('Sepette en fazla 6 hisse bulunabilir.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setHoverIndex(null);

    try {
      const payload = {
        basket: basket.map(b => ({
          ticker: b.ticker.toUpperCase().trim(),
          weight_pct: Number(b.weight_pct) || 0
        })),
        cash_weight_pct: Number(cashWeight) || 0,
        max_cash_pct: Number(maxCashPct) || 60,
        initial_capital: initialCapital,
        rebalance_interval_days: Number(rebalanceInterval) || 7
      };

      const res = await fetch('/api/simulation/portfolio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => null);
        throw new Error(errJson?.detail || `Sunucu hatası (${res.status})`);
      }

      const data: PortfolioSimResponse = await res.json();
      setSimResult(data);
      // Başlangıçta hisse çizgilerini kapalı tutuyoruz (yalnızca ana portföy ve benchmark görünsün)
      setActiveTickers(new Set());
    } catch (err: any) {
      setError(err.message || 'Simülasyon çalıştırılırken bir hata meydana geldi.');
    } finally {
      setIsLoading(false);
    }
  };

  // İlk yüklemede otomatik simülasyon koştur
  useEffect(() => {
    runSimulation();
  }, []);

  // Sepete Hisse Ekle
  const handleAddTicker = (t: string) => {
    const clean = t.trim().toUpperCase();
    if (!clean) return;
    if (basket.some(b => b.ticker === clean)) return;
    if (basket.length >= 6) {
      alert('En fazla 6 hisse seçebilirsiniz.');
      return;
    }
    const newBasket = [...basket, { ticker: clean, weight_pct: 15 }];
    setBasket(newBasket);
    setNewTickerInput('');
  };

  // Sepetten Hisse Çıkar
  const handleRemoveTicker = (ticker: string) => {
    const newBasket = basket.filter(b => b.ticker !== ticker);
    setBasket(newBasket);
    if (activeTickers.has(ticker)) {
      const next = new Set(activeTickers);
      next.delete(ticker);
      setActiveTickers(next);
    }
  };

  // Ağırlık Güncelle
  const handleWeightChange = (index: number, val: number) => {
    const updated = [...basket];
    updated[index].weight_pct = Math.max(0, Math.min(100, val));
    setBasket(updated);
  };

  // Ağırlıkları Otomatik Normalize Et (%100'e eşitle)
  const handleAutoNormalize = () => {
    if (basket.length === 0) return;
    const remainingForStocks = Math.max(0, 100 - cashWeight);
    const equalShare = Math.floor(remainingForStocks / basket.length);
    const remainder = remainingForStocks - equalShare * basket.length;
    const updated = basket.map((item, idx) => ({
      ...item,
      weight_pct: equalShare + (idx === 0 ? remainder : 0)
    }));
    setBasket(updated);
  };

  // Ticker Çizgisini Aç/Kapat (Toggle)
  const toggleTickerVisibility = (ticker: string) => {
    const next = new Set(activeTickers);
    if (next.has(ticker)) {
      next.delete(ticker);
    } else {
      next.add(ticker);
    }
    setActiveTickers(next);
  };

  // Renk Atama Haritası
  const tickerColorMap = useMemo(() => {
    const map: Record<string, string> = {};
    basket.forEach((b, idx) => {
      map[b.ticker] = SIGNATURE_COLORS[idx % SIGNATURE_COLORS.length];
    });
    return map;
  }, [basket]);

  /* ── GRAFİK HESAPLAMALARI (SVG Canvas) ─────────────────────────────────── */
  const timeline = simResult?.timeline || [];
  const N = timeline.length;

  const chartMetrics = useMemo(() => {
    if (N === 0) return null;

    const p0 = timeline[0].portfolio_equity;
    const b0 = timeline[0].benchmark_equity;

    // Her seri için değer serileri
    const aiSeries: number[] = [];
    const bhSeries: number[] = [];
    const assetSeriesMap: Record<string, number[]> = {};

    basket.forEach(b => {
      assetSeriesMap[b.ticker] = [];
    });

    timeline.forEach(step => {
      if (viewMode === 'return') {
        aiSeries.push(((step.portfolio_equity - p0) / p0) * 100);
        bhSeries.push(((step.benchmark_equity - b0) / b0) * 100);
        basket.forEach(b => {
          const aInfo = step.assets[b.ticker];
          assetSeriesMap[b.ticker].push(aInfo ? aInfo.return_pct : 0);
        });
      } else {
        aiSeries.push(step.portfolio_equity);
        bhSeries.push(step.benchmark_equity);
        basket.forEach(b => {
          const aInfo = step.assets[b.ticker];
          const initAlloc = initialCapital * (b.weight_pct / 100);
          const valNorm = aInfo ? initAlloc * (1 + aInfo.return_pct / 100) : initAlloc;
          assetSeriesMap[b.ticker].push(valNorm);
        });
      }
    });

    // Dinamik Min-Max (Yalnızca şu an görünür olan serilere göre hesaplanır)
    let allVisibleVals: number[] = [];
    if (showAiPort) allVisibleVals = allVisibleVals.concat(aiSeries);
    if (showBenchmark) allVisibleVals = allVisibleVals.concat(bhSeries);
    activeTickers.forEach(t => {
      if (assetSeriesMap[t]) {
        allVisibleVals = allVisibleVals.concat(assetSeriesMap[t]);
      }
    });
    if (allVisibleVals.length === 0) {
      allVisibleVals = [...aiSeries, ...bhSeries];
    }

    let minVal = Math.min(...allVisibleVals);
    let maxVal = Math.max(...allVisibleVals);

    if (minVal === maxVal) {
      minVal -= 10;
      maxVal += 10;
    }
    const padding = (maxVal - minVal) * 0.12;
    minVal -= padding;
    maxVal += padding;

    return {
      minVal,
      maxVal,
      aiSeries,
      bhSeries,
      assetSeriesMap
    };
  }, [timeline, viewMode, activeTickers, basket, initialCapital, N, showAiPort, showBenchmark]);

  // SVG Çizim Parametreleri
  const svgWidth = 960;
  const svgHeight = 440;
  const padLeft = 70;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 40;
  const plotWidth = svgWidth - padLeft - padRight;
  const plotHeight = svgHeight - padTop - padBottom;

  const getX = (idx: number) => {
    if (N <= 1) return padLeft;
    return padLeft + (idx / (N - 1)) * plotWidth;
  };

  const getY = (val: number) => {
    if (!chartMetrics) return padTop + plotHeight / 2;
    const { minVal, maxVal } = chartMetrics;
    const pct = (val - minVal) / (maxVal - minVal);
    return padTop + plotHeight - pct * plotHeight;
  };

  // SVG Path Üretici
  const generatePath = (vals: number[]) => {
    if (!vals || vals.length === 0) return '';
    return vals.reduce((acc, v, i) => {
      const x = getX(i);
      const y = getY(v);
      return i === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : `${acc} L ${x.toFixed(1)} ${y.toFixed(1)}`;
    }, '');
  };

  // İşlem Kümeleri (Çakışan veya yakın günlerdeki işlemleri 3-4 kart halinde üst üste gruplar)
  const tradeClusters = useMemo(() => {
    if (!simResult || !chartMetrics || N === 0) return [];

    const visibleTrades = simResult.trades.filter(tr => {
      if (activeTickers.size === 0) return true;
      return activeTickers.has(tr.ticker);
    });

    if (visibleTrades.length === 0) return [];

    const sorted = [...visibleTrades].sort((a, b) => a.day_index - b.day_index);

    const clusters: {
      stepIdx: number;
      date: string;
      x: number;
      y: number;
      trades: PortfolioSimTrade[];
    }[] = [];

    sorted.forEach(tr => {
      const stepIdx = tr.day_index;
      if (stepIdx < 0 || stepIdx >= N) return;
      const x = getX(stepIdx);
      const yVal = chartMetrics.aiSeries[stepIdx];
      const y = getY(yVal);

      // 12 piksel aralığındaki noktaları aynı tıklama kümesinde birleştir
      const existing = clusters.find(c => Math.abs(c.x - x) <= 12);
      if (existing) {
        existing.trades.push(tr);
      } else {
        clusters.push({
          stepIdx,
          date: tr.date,
          x,
          y,
          trades: [tr]
        });
      }
    });

    return clusters;
  }, [simResult, chartMetrics, N, activeTickers]);

  // Mouse Scrubbing
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!chartRef.current || N <= 1) return;
    const rect = chartRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const normX = (mouseX - padLeft) / plotWidth;
    const clamped = Math.max(0, Math.min(1, normX));
    const stepIdx = Math.round(clamped * (N - 1));
    setHoverIndex(stepIdx);
  };

  const handleMouseLeave = () => {
    setHoverIndex(null);
  };

  // Aktif adım bilgileri (hover yoksa son gün)
  const currentStepIndex = hoverIndex !== null ? hoverIndex : Math.max(0, N - 1);
  const currentStep = timeline[currentStepIndex] || null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', animation: 'fadeUp 0.3s ease' }}>
      
      {/* ── ÜST GEZİNME VE BAŞLIK ────────────────────────────────────────── */}
      <div style={{ borderTop: '3px solid var(--ink-primary)', paddingTop: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <button
            onClick={onBack}
            className="btn btn-secondary"
            style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: '0.78rem', padding: '6px 14px' }}
          >
            <ArrowLeft size={14} /> Portföy Atölyesine Dön
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="status-dot" style={{ background: 'var(--forest-gain)' }} />
            <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--ink-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Çoklu-Varlık Dinamik Motor (TimesFM + LightGBM + Değerleme)
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
          <div>
            <div style={{ fontSize: '0.65rem', fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 4 }}>
              1 Yıllık Geriye Dönük Portföy Simülasyonu
            </div>
            <h1 style={{ fontFamily: 'var(--font-display)', fontSize: '2.1rem', fontWeight: 700, color: 'var(--ink-primary)', letterSpacing: '-0.02em', lineHeight: 1.1 }}>
              Dinamik Varlık &amp; Nakit Tahsis Motoru
            </h1>
            <p style={{ color: 'var(--ink-secondary)', fontSize: '0.88rem', marginTop: 8, lineHeight: 1.6, maxWidth: 840 }}>
              $100.000 USD sermaye ile son 1 yıllık piyasa koşullarında (252 işlem günü); temel değerleme çarpanları, teknik 200 SMA çıpaları ve güven aralıkları ile fırsat alımı ve nakit biriktirme rotasyonunu canlı simüle edin.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <button
              onClick={runSimulation}
              disabled={isLoading}
              className="btn btn-primary"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                padding: '10px 22px',
                fontSize: '0.88rem',
                fontWeight: 600,
                background: 'var(--forest-gain)',
                borderColor: 'var(--forest-mid)',
                color: '#fff',
                cursor: isLoading ? 'wait' : 'pointer'
              }}
            >
              {isLoading ? (
                <>
                  <Activity size={16} className="animate-spin" />
                  Simülasyon Hesaplanıyor…
                </>
              ) : (
                <>
                  <Play size={16} fill="#fff" />
                  Simülasyonu Başlat (100k$)
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* ── SEPET VE AĞIRLIK YAPILANDIRMA MASASI ──────────────────────────── */}
      <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)', padding: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', borderBottom: '1px solid var(--rule-hairline)', paddingBottom: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Sliders size={18} style={{ color: 'var(--cobalt)' }} />
            <h2 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--ink-primary)', margin: 0 }}>
              Varlık Sepeti &amp; Başlangıç Ağırlık Masası
            </h2>
            <span style={{ fontSize: '0.75rem', color: 'var(--ink-muted)', marginLeft: 8 }}>
              (Maksimum 6 Hisse + Nakit)
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--ink-secondary)' }}>Toplam Ağırlık:</span>
              <span 
                className="tabular" 
                style={{ 
                  fontSize: '0.95rem', 
                  fontWeight: 700, 
                  color: totalAllocated === 100 ? 'var(--forest-gain)' : totalAllocated > 100 ? 'var(--madder-loss)' : 'var(--amber-warm)',
                  padding: '2px 8px',
                  background: 'var(--paper-card)',
                  borderRadius: 'var(--radius-xs)',
                  border: '1px solid var(--rule-strong)'
                }}
              >
                %{totalAllocated.toFixed(0)}
              </span>
            </div>

            <button
              onClick={handleAutoNormalize}
              className="btn btn-secondary"
              style={{ fontSize: '0.75rem', padding: '4px 10px' }}
              title="Kalan ağırlığı hisselere eşit dağıtarak toplamı %100 yapar"
            >
              Normalize Et (%100)
            </button>
          </div>
        </div>

        {/* Sepet Kartları Izgarası */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
          {basket.map((item, idx) => {
            const sigColor = tickerColorMap[item.ticker] || '#2563EB';
            return (
              <div 
                key={item.ticker}
                style={{
                  background: 'var(--paper-card)',
                  border: '1px solid var(--rule-strong)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '0.9rem 1rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                  position: 'relative'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span 
                      style={{ 
                        width: 10, 
                        height: 10, 
                        borderRadius: '50%', 
                        background: sigColor, 
                        display: 'inline-block' 
                      }} 
                    />
                    <span className="tabular" style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--ink-primary)' }}>
                      {item.ticker}
                    </span>
                  </div>

                  <button
                    onClick={() => handleRemoveTicker(item.ticker)}
                    style={{ background: 'none', border: 'none', color: 'var(--ink-muted)', cursor: 'pointer', padding: 2 }}
                    title="Varlığı Sepetten Çıkar"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={item.weight_pct}
                    onChange={(e) => handleWeightChange(idx, Number(e.target.value))}
                    style={{ flex: 1, accentColor: sigColor }}
                  />
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <input
                      type="number"
                      min={0}
                      max={100}
                      value={item.weight_pct}
                      onChange={(e) => handleWeightChange(idx, Number(e.target.value))}
                      className="tabular"
                      style={{
                        width: 48,
                        padding: '3px 6px',
                        fontSize: '0.85rem',
                        fontWeight: 600,
                        textAlign: 'right',
                        border: '1px solid var(--rule-strong)',
                        borderRadius: 'var(--radius-xs)',
                        background: 'var(--paper-base)'
                      }}
                    />
                    <span style={{ fontSize: '0.8rem', color: 'var(--ink-muted)' }}>%</span>
                  </div>
                </div>
              </div>
            );
          })}

          {/* Nakit Kartı */}
          <div 
            style={{
              background: 'rgba(20, 83, 45, 0.04)',
              border: '1px dashed var(--forest-mid)',
              borderRadius: 'var(--radius-sm)',
              padding: '0.9rem 1rem',
              display: 'flex',
              flexDirection: 'column',
              gap: 8
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <DollarSign size={16} style={{ color: 'var(--forest-gain)' }} />
                <span style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--forest-gain)' }}>
                  Başlangıç Nakit (USD)
                </span>
              </div>
              <span className="tabular" style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--ink-muted)' }}>
                ${((initialCapital * cashWeight) / 100).toLocaleString('en-US')}
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
              <input
                type="range"
                min={0}
                max={100}
                value={cashWeight}
                onChange={(e) => setCashWeight(Number(e.target.value))}
                style={{ flex: 1, accentColor: 'var(--forest-gain)' }}
              />
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <input
                  type="number"
                  min={0}
                  max={100}
                  value={cashWeight}
                  onChange={(e) => setCashWeight(Number(e.target.value))}
                  className="tabular"
                  style={{
                    width: 48,
                    padding: '3px 6px',
                    fontSize: '0.85rem',
                    fontWeight: 600,
                    textAlign: 'right',
                    border: '1px solid var(--forest-rule)',
                    borderRadius: 'var(--radius-xs)',
                    background: 'var(--paper-base)'
                  }}
                />
                <span style={{ fontSize: '0.8rem', color: 'var(--forest-gain)' }}>%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Sepet Yönetim Araç Çubuğu (Yeni Hisse Ekle + Popüler Hızlı Butonlar + Maks Nakit Sınırı) */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, paddingTop: '0.75rem', borderTop: '1px solid var(--rule-hairline)' }}>
          {/* Hızlı Sembol Ekleme */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--ink-muted)', textTransform: 'uppercase' }}>
              Önerilen Varlıklar:
            </span>
            {suggestedTickers.map(t => {
              const inBasket = basket.some(b => b.ticker === t);
              return (
                <button
                  key={t}
                  disabled={inBasket || basket.length >= 6}
                  onClick={() => handleAddTicker(t)}
                  style={{
                    padding: '3px 8px',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    borderRadius: 'var(--radius-xs)',
                    border: '1px solid var(--rule-strong)',
                    background: inBasket ? 'var(--paper-elevated)' : 'var(--paper-card)',
                    color: inBasket ? 'var(--ink-faint)' : 'var(--ink-primary)',
                    cursor: inBasket ? 'default' : 'pointer'
                  }}
                >
                  +{t}
                </button>
              );
            })}

            {/* Özel Ticker Girişi */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginLeft: 8 }}>
              <input
                type="text"
                placeholder="Ör: TSLA"
                value={newTickerInput}
                onChange={(e) => setNewTickerInput(e.target.value.toUpperCase())}
                onKeyDown={(e) => e.key === 'Enter' && handleAddTicker(newTickerInput)}
                style={{
                  width: 70,
                  padding: '3px 6px',
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                  border: '1px solid var(--rule-strong)',
                  borderRadius: 'var(--radius-xs)'
                }}
              />
              <button
                onClick={() => handleAddTicker(newTickerInput)}
                disabled={!newTickerInput || basket.length >= 6}
                className="btn btn-secondary"
                style={{ padding: '3px 8px', fontSize: '0.72rem' }}
              >
                Ekle
              </button>
            </div>
          </div>

          {/* Model Parametreleri (Maks Nakit Sınırı & Dengeleme Periyodu) */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--ink-secondary)', fontWeight: 600 }}>Maks. Nakit Sınırı:</span>
              <input
                type="number"
                min={10}
                max={100}
                value={maxCashPct}
                onChange={(e) => setMaxCashPct(Number(e.target.value))}
                className="tabular"
                style={{
                  width: 50,
                  padding: '3px 6px',
                  fontSize: '0.78rem',
                  fontWeight: 600,
                  textAlign: 'center',
                  border: '1px solid var(--rule-strong)',
                  borderRadius: 'var(--radius-xs)'
                }}
              />
              <span style={{ fontSize: '0.75rem', color: 'var(--ink-muted)' }}>%</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--ink-secondary)', fontWeight: 600 }}>Karar Periyodu:</span>
              <select
                value={rebalanceInterval}
                onChange={(e) => setRebalanceInterval(Number(e.target.value))}
                style={{
                  padding: '3px 6px',
                  fontSize: '0.78rem',
                  border: '1px solid var(--rule-strong)',
                  borderRadius: 'var(--radius-xs)',
                  background: 'var(--paper-card)'
                }}
              >
                <option value={3}>3 Günlük</option>
                <option value={7}>7 Günlük (Haftalık)</option>
                <option value={14}>14 Günlük (2 Haftalık)</option>
              </select>
            </div>
          </div>
        </div>

        {error && (
          <div style={{ marginTop: 12, padding: '8px 12px', background: 'var(--madder-tint)', border: '1px solid var(--madder-rule)', borderRadius: 'var(--radius-xs)', display: 'flex', alignItems: 'center', gap: 8, color: 'var(--madder-loss)', fontSize: '0.82rem' }}>
            <AlertCircle size={15} />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* ── PERFORMANS SKOR KARTI ────────────────────────────────────────── */}
      {simResult && (
        <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)', display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)' }}>
          <div className="metric-block">
            <div className="metric-label">AI Portföy Değeri ($)</div>
            <div className="metric-value tabular" style={{ color: 'var(--forest-gain)' }}>
              ${simResult.performance.ai_final_equity.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
            </div>
            <div className="metric-sub" style={{ color: simResult.performance.ai_total_return_pct >= 0 ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
              {simResult.performance.ai_total_return_pct >= 0 ? '+' : ''}{simResult.performance.ai_total_return_pct.toFixed(2)}% Toplam Getiri
            </div>
          </div>

          <div className="metric-block">
            <div className="metric-label">Al-Tut (B&amp;H) Karşılaştırması</div>
            <div className="metric-value tabular" style={{ color: 'var(--ink-primary)' }}>
              ${simResult.performance.buy_hold_final_equity.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
            </div>
            <div className="metric-sub" style={{ color: simResult.performance.buy_hold_total_return_pct >= 0 ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
              {simResult.performance.buy_hold_total_return_pct >= 0 ? '+' : ''}{simResult.performance.buy_hold_total_return_pct.toFixed(2)}% Getiri
            </div>
          </div>

          <div className="metric-block">
            <div className="metric-label">Alfa Farkı (AI - B&amp;H)</div>
            <div 
              className="metric-value tabular" 
              style={{ color: simResult.performance.alpha_spread_pct >= 0 ? 'var(--forest-gain)' : 'var(--madder-loss)' }}
            >
              {simResult.performance.alpha_spread_pct >= 0 ? '+' : ''}{simResult.performance.alpha_spread_pct.toFixed(2)}%
            </div>
            <div className="metric-sub">
              {simResult.performance.alpha_spread_pct >= 0 ? 'Pozitif Katma Değer' : 'Piyasa Altı'}
            </div>
          </div>

          <div className="metric-block">
            <div className="metric-label">Sharpe / Sortino</div>
            <div className="metric-value tabular" style={{ color: 'var(--cobalt)' }}>
              {simResult.performance.ai_sharpe.toFixed(2)} / {simResult.performance.ai_sortino.toFixed(2)}
            </div>
            <div className="metric-sub">Risk Düzeltilmiş Getiri</div>
          </div>

          <div className="metric-block">
            <div className="metric-label">Maks. Çekilme (Drawdown)</div>
            <div className="metric-value tabular" style={{ color: 'var(--madder-loss)' }}>
              -%{simResult.performance.ai_max_drawdown_pct.toFixed(2)}
            </div>
            <div className="metric-sub">{simResult.performance.total_trades} Rebalance İşlemi</div>
          </div>
        </div>
      )}

      {/* ── İNTERAKTİF ÇOK-VARLIK GRAFİĞİ ─────────────────────────────────── */}
      {simResult && chartMetrics && (
        <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)', padding: '1.25rem 1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          
          {/* Grafik Kontrolleri & Ticker Seçim Hapları */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
            <div>
              <div style={{ fontSize: '0.68rem', fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--ink-muted)' }}>
                İnteraktif Çok-Varlık Trajektorisi
              </div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--ink-primary)', marginTop: 2 }}>
                252 İşlem Günü Portföy &amp; Varlık Seyri
              </div>
            </div>

            {/* Seri ve Varlık Seçim Hapları */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--ink-muted)', textTransform: 'uppercase' }}>
                Portföy:
              </span>

              {/* 1. Model AI Portföyü Çizgisi */}
              <button
                onClick={() => setShowAiPort(prev => !prev)}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '5px 12px',
                  fontSize: '0.78rem',
                  fontWeight: 700,
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  background: showAiPort ? 'var(--forest-gain)' : 'var(--paper-card)',
                  color: showAiPort ? '#FFFFFF' : 'var(--ink-secondary)',
                  border: `1px solid ${showAiPort ? 'var(--forest-gain)' : 'var(--rule-strong)'}`,
                  boxShadow: showAiPort ? '0 2px 8px rgba(20, 83, 45, 0.25)' : 'none'
                }}
                title="Model AI dinamik yönetim portföy çizgisini aç/kapat"
              >
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: showAiPort ? '#fff' : 'var(--forest-gain)' }} />
                Model AI Portföyü
              </button>

              {/* 2. Başlangıçtaki Al-Tut Ağırlıkları (Varsayılan Olarak Açık) */}
              <button
                onClick={() => setShowBenchmark(prev => !prev)}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '5px 12px',
                  fontSize: '0.78rem',
                  fontWeight: 700,
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  background: showBenchmark ? '#475569' : 'var(--paper-card)',
                  color: showBenchmark ? '#FFFFFF' : 'var(--ink-secondary)',
                  border: `1px solid ${showBenchmark ? '#475569' : 'var(--rule-strong)'}`,
                  boxShadow: showBenchmark ? '0 2px 8px rgba(71, 85, 105, 0.25)' : 'none'
                }}
                title="Başlangıçta belirlenen sabit hisse ve nakit ağırlıklarıyla Al-Tut (Buy & Hold) karşılaştırma çizgisini aç/kapat"
              >
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: showBenchmark ? '#fff' : '#475569' }} />
                Başlangıç Al-Tut (B&amp;H)
              </button>

              <div style={{ width: 1, height: 18, background: 'var(--rule-hairline)', margin: '0 4px' }} />

              <span style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--ink-muted)', textTransform: 'uppercase' }}>
                Hisseler:
              </span>
              {basket.map(b => {
                const isSelected = activeTickers.has(b.ticker);
                const color = tickerColorMap[b.ticker] || '#2563EB';
                return (
                  <button
                    key={b.ticker}
                    onClick={() => toggleTickerVisibility(b.ticker)}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 6,
                      padding: '5px 12px',
                      fontSize: '0.78rem',
                      fontWeight: 700,
                      borderRadius: 'var(--radius-sm)',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      background: isSelected ? color : 'var(--paper-card)',
                      color: isSelected ? '#FFFFFF' : 'var(--ink-secondary)',
                      border: `1px solid ${isSelected ? color : 'var(--rule-strong)'}`,
                      boxShadow: isSelected ? `0 2px 8px ${color}33` : 'none'
                    }}
                    title={`${b.ticker} getiri çizgisini ve işlem noktalarını ${isSelected ? 'gizle' : 'göster'}`}
                  >
                    <span 
                      style={{ 
                        width: 8, 
                        height: 8, 
                        borderRadius: '50%', 
                        background: isSelected ? '#FFFFFF' : color 
                      }} 
                    />
                    {b.ticker}
                    <span style={{ fontSize: '0.7rem', opacity: 0.85 }}>
                      (%{b.weight_pct})
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Metrik Modu Butonları (% Getiri vs $ Değer) */}
            <div style={{ display: 'flex', alignItems: 'center', background: 'var(--paper-card)', border: '1px solid var(--rule-strong)', borderRadius: 'var(--radius-xs)', padding: 2 }}>
              <button
                onClick={() => setViewMode('return')}
                style={{
                  padding: '4px 10px',
                  fontSize: '0.74rem',
                  fontWeight: 600,
                  border: 'none',
                  borderRadius: 'var(--radius-xs)',
                  background: viewMode === 'return' ? 'var(--ink-primary)' : 'transparent',
                  color: viewMode === 'return' ? '#fff' : 'var(--ink-secondary)',
                  cursor: 'pointer'
                }}
              >
                % Getiri
              </button>
              <button
                onClick={() => setViewMode('equity')}
                style={{
                  padding: '4px 10px',
                  fontSize: '0.74rem',
                  fontWeight: 600,
                  border: 'none',
                  borderRadius: 'var(--radius-xs)',
                  background: viewMode === 'equity' ? 'var(--ink-primary)' : 'transparent',
                  color: viewMode === 'equity' ? '#fff' : 'var(--ink-secondary)',
                  cursor: 'pointer'
                }}
              >
                $ Bakiye
              </button>
            </div>
          </div>

          {/* Canlı Scrubbing HUD Paneli */}
          {currentStep && (
            <div 
              style={{
                background: 'var(--paper-card)',
                border: '1px solid var(--rule-strong)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.75rem 1rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: 16
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <div>
                  <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--ink-muted)', fontWeight: 600 }}>
                    Tarih
                  </div>
                  <div className="tabular" style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--ink-primary)' }}>
                    {currentStep.date}
                  </div>
                </div>

                <div style={{ width: 1, height: 28, background: 'var(--rule-hairline)' }} />

                <div>
                  <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--ink-muted)', fontWeight: 600 }}>
                    Model AI Portföyü
                  </div>
                  <div className="tabular" style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--forest-gain)' }}>
                    ${currentStep.portfolio_equity.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                    <span style={{ fontSize: '0.74rem', marginLeft: 4, fontWeight: 600 }}>
                      ({((currentStep.portfolio_equity - timeline[0].portfolio_equity) / timeline[0].portfolio_equity * 100) >= 0 ? '+' : ''}
                      {((currentStep.portfolio_equity - timeline[0].portfolio_equity) / timeline[0].portfolio_equity * 100).toFixed(1)}%)
                    </span>
                  </div>
                </div>

                <div style={{ width: 1, height: 28, background: 'var(--rule-hairline)' }} />

                <div>
                  <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--ink-muted)', fontWeight: 600 }}>
                    Başlangıç Al-Tut (B&amp;H)
                  </div>
                  <div className="tabular" style={{ fontSize: '0.92rem', fontWeight: 700, color: '#475569' }}>
                    ${currentStep.benchmark_equity.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                    <span style={{ fontSize: '0.74rem', marginLeft: 4, fontWeight: 600 }}>
                      ({((currentStep.benchmark_equity - timeline[0].benchmark_equity) / timeline[0].benchmark_equity * 100) >= 0 ? '+' : ''}
                      {((currentStep.benchmark_equity - timeline[0].benchmark_equity) / timeline[0].benchmark_equity * 100).toFixed(1)}%)
                    </span>
                  </div>
                </div>

                <div style={{ width: 1, height: 28, background: 'var(--rule-hairline)' }} />

                <div>
                  <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--ink-muted)', fontWeight: 600 }}>
                    Alfa Farkı
                  </div>
                  <div className="tabular" style={{ 
                    fontSize: '0.92rem', 
                    fontWeight: 700, 
                    color: currentStep.portfolio_equity >= currentStep.benchmark_equity ? 'var(--forest-gain)' : 'var(--madder-loss)' 
                  }}>
                    {(((currentStep.portfolio_equity - timeline[0].portfolio_equity) / timeline[0].portfolio_equity * 100) - ((currentStep.benchmark_equity - timeline[0].benchmark_equity) / timeline[0].benchmark_equity * 100)) >= 0 ? '+' : ''}
                    {(((currentStep.portfolio_equity - timeline[0].portfolio_equity) / timeline[0].portfolio_equity * 100) - ((currentStep.benchmark_equity - timeline[0].benchmark_equity) / timeline[0].benchmark_equity * 100)).toFixed(1)}%
                  </div>
                </div>

                <div style={{ width: 1, height: 28, background: 'var(--rule-hairline)' }} />

                <div>
                  <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--ink-muted)', fontWeight: 600 }}>
                    Nakit Oranı
                  </div>
                  <div className="tabular" style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--ink-primary)' }}>
                    %{currentStep.cash_pct.toFixed(1)} (${currentStep.cash_value.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })})
                  </div>
                </div>
              </div>

              {/* Varlıkların Fiyatları ve Sağda Anlık Portföy Ağırlıkları */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', borderTop: '1px solid var(--rule-hairline)', paddingTop: 8, marginTop: 4, flexWrap: 'wrap', gap: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
                  {basket.map(b => {
                    const aInfo = currentStep.assets[b.ticker];
                    if (!aInfo) return null;
                    const color = tickerColorMap[b.ticker] || '#2563EB';
                    return (
                      <div key={b.ticker} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem' }}>
                        <span style={{ width: 6, height: 6, borderRadius: '50%', background: color }} />
                        <span style={{ fontWeight: 700, color: 'var(--ink-primary)' }}>{b.ticker}:</span>
                        <span className="tabular" style={{ color: 'var(--ink-secondary)' }}>
                          ${aInfo.price.toFixed(2)}
                        </span>
                        <span className="tabular" style={{ fontWeight: 600, color: aInfo.return_pct >= 0 ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
                          ({aInfo.return_pct >= 0 ? '+' : ''}{aInfo.return_pct.toFixed(1)}%)
                        </span>
                      </div>
                    );
                  })}
                </div>

                {/* Sağ Tarafta O Anki Ağırlık Dağılımı */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, background: 'var(--paper-elevated)', padding: '4px 10px', borderRadius: 'var(--radius-xs)', border: '1px solid var(--rule-strong)' }}>
                  <span style={{ fontSize: '0.68rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--ink-muted)', letterSpacing: '0.06em' }}>
                    Anlık Ağırlıklar:
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    {basket.map(b => {
                      const aInfo = currentStep.assets[b.ticker];
                      const w = aInfo ? aInfo.weight_pct : 0;
                      const color = tickerColorMap[b.ticker] || '#2563EB';
                      return (
                        <span key={b.ticker} className="tabular" style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--ink-primary)', display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                          <span style={{ color, fontWeight: 700 }}>{b.ticker}</span>
                          <span style={{ fontWeight: 700 }}>%{w.toFixed(1)}</span>
                        </span>
                      );
                    })}
                    <span className="tabular" style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--forest-gain)', display: 'inline-flex', alignItems: 'center', gap: 3, borderLeft: '1px solid var(--rule-strong)', paddingLeft: 8 }}>
                      <span>Nakit</span>
                      <span>%{currentStep.cash_pct.toFixed(1)}</span>
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* SVG Canvas Grafiği */}
          <div style={{ width: '100%', overflowX: 'auto', background: 'var(--paper-card)', border: '1px solid var(--rule-strong)', borderRadius: 'var(--radius-sm)', position: 'relative' }}>
            <svg
              ref={chartRef}
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              style={{ width: '100%', height: 'auto', display: 'block', cursor: 'crosshair' }}
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
            >
              <defs>
                <linearGradient id="aiPortGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#14532D" stopOpacity="0.18" />
                  <stop offset="100%" stopColor="#14532D" stopOpacity="0.0" />
                </linearGradient>
              </defs>

              {/* Yatay Kılavuz Çizgileri ve Y-Eksen Etiketleri */}
              {[0, 0.25, 0.5, 0.75, 1.0].map((frac, i) => {
                const y = padTop + frac * plotHeight;
                const { minVal, maxVal } = chartMetrics;
                const val = maxVal - frac * (maxVal - minVal);
                return (
                  <g key={i}>
                    <line
                      x1={padLeft}
                      y1={y}
                      x2={svgWidth - padRight}
                      y2={y}
                      stroke="var(--rule-hairline)"
                      strokeDasharray="4 4"
                      strokeWidth={1}
                    />
                    <text
                      x={padLeft - 10}
                      y={y + 4}
                      textAnchor="end"
                      fontSize="10"
                      fontFamily="var(--font-mono)"
                      fill="var(--ink-muted)"
                    >
                      {viewMode === 'return' ? `${val.toFixed(1)}%` : `$${Math.round(val).toLocaleString('en-US')}`}
                    </text>
                  </g>
                );
              })}

              {/* 0% Getiri Referans Çizgisi (Eğer viewMode === 'return' ise) */}
              {viewMode === 'return' && chartMetrics.minVal <= 0 && chartMetrics.maxVal >= 0 && (
                <line
                  x1={padLeft}
                  y1={getY(0)}
                  x2={svgWidth - padRight}
                  y2={getY(0)}
                  stroke="var(--rule-strong)"
                  strokeWidth={1.2}
                />
              )}

              {/* Başlangıç Al-Tut (Buy & Hold) Karşılaştırma Çizgisi (Varsayılan Olarak Açık) */}
              {showBenchmark && (
                <path
                  d={generatePath(chartMetrics.bhSeries)}
                  fill="none"
                  stroke="#475569"
                  strokeWidth={2.6}
                  strokeDasharray="6 4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity={0.9}
                />
              )}

              {/* Aktif Ticker Çizgileri */}
              {basket.map(b => {
                const isSelected = activeTickers.has(b.ticker);
                if (!isSelected) return null;
                const color = tickerColorMap[b.ticker] || '#2563EB';
                const series = chartMetrics.assetSeriesMap[b.ticker];
                if (!series) return null;
                return (
                  <g key={`series-${b.ticker}`}>
                    <path
                      d={generatePath(series)}
                      fill="none"
                      stroke={color}
                      strokeWidth={2.2}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </g>
                );
              })}

              {/* Ana Model Portföy Çizgisi (Bold British Racing Green) */}
              {showAiPort && (
                <path
                  d={generatePath(chartMetrics.aiSeries)}
                  fill="none"
                  stroke="var(--forest-gain)"
                  strokeWidth={3.5}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}

              {/* İşlem Noktaları ve Kümeleri (Tıklanabilir) */}
              {tradeClusters.map((cluster, idx) => {
                const hasBuy = cluster.trades.some(t => t.action.includes('ALIM'));
                const hasSell = cluster.trades.some(t => !t.action.includes('ALIM'));
                const isMulti = cluster.trades.length > 1;
                const isSelected = selectedTradeCluster?.stepIdx === cluster.stepIdx;

                const dotColor = hasBuy && hasSell ? 'var(--cobalt)' : hasBuy ? 'var(--forest-gain)' : 'var(--madder-loss)';

                return (
                  <g 
                    key={`cluster-${idx}`} 
                    style={{ cursor: 'pointer' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedTradeCluster(cluster);
                    }}
                  >
                    {/* Genişletilmiş Tıklama Alanı (16px) */}
                    <circle
                      cx={cluster.x}
                      cy={cluster.y}
                      r={16}
                      fill="transparent"
                    />

                    {/* Seçili İse Vurgulu Dış Halka */}
                    {isSelected && (
                      <circle
                        cx={cluster.x}
                        cy={cluster.y}
                        r={12}
                        fill="none"
                        stroke={dotColor}
                        strokeWidth={2.5}
                        strokeDasharray="3 3"
                      />
                    )}

                    {/* Ana Nokta */}
                    <circle
                      cx={cluster.x}
                      cy={cluster.y}
                      r={isMulti ? 7 : 5.5}
                      fill={dotColor}
                      stroke="#FFFFFF"
                      strokeWidth={2}
                    />

                    {/* Çoklu işlem varsa rozet sayısı */}
                    {isMulti && (
                      <text
                        x={cluster.x}
                        y={cluster.y + 3}
                        textAnchor="middle"
                        fontSize="8.5"
                        fontWeight="800"
                        fontFamily="var(--font-mono)"
                        fill="#FFFFFF"
                        style={{ pointerEvents: 'none', userSelect: 'none' }}
                      >
                        {cluster.trades.length}
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Mouse Scrubbing Dikey Saç Çizgisi ve Kesişim Noktaları */}
              {hoverIndex !== null && (
                <g>
                  <line
                    x1={getX(hoverIndex)}
                    y1={padTop}
                    x2={getX(hoverIndex)}
                    y2={padTop + plotHeight}
                    stroke="var(--ink-primary)"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                  />
                  {/* Başlangıç Al-Tut (B&H) Kesişim Noktası */}
                  {showBenchmark && (
                    <circle
                      cx={getX(hoverIndex)}
                      cy={getY(chartMetrics.bhSeries[hoverIndex])}
                      r={5}
                      fill="#475569"
                      stroke="#fff"
                      strokeWidth={2}
                    />
                  )}
                  {/* AI Portföy Kesişim Noktası */}
                  {showAiPort && (
                    <circle
                      cx={getX(hoverIndex)}
                      cy={getY(chartMetrics.aiSeries[hoverIndex])}
                      r={6}
                      fill="var(--forest-gain)"
                      stroke="#fff"
                      strokeWidth={2}
                    />
                  )}
                </g>
              )}

              {/* X-Eksen Tarih Etiketleri */}
              {N > 0 && [0, Math.floor(N / 4), Math.floor(N / 2), Math.floor((3 * N) / 4), N - 1].map((stepIdx) => {
                const dateStr = timeline[stepIdx]?.date || '';
                return (
                  <text
                    key={stepIdx}
                    x={getX(stepIdx)}
                    y={svgHeight - padBottom + 20}
                    textAnchor="middle"
                    fontSize="10"
                    fontFamily="var(--font-mono)"
                    fill="var(--ink-muted)"
                  >
                    {dateStr}
                  </text>
                );
              })}
            </svg>

            {/* Tıklanan Noktadaki İşlem Kartları Yığını (3-4 Kart Üst Üste) */}
            {selectedTradeCluster && (
              <div
                style={{
                  position: 'absolute',
                  left: `${(selectedTradeCluster.x / svgWidth) * 100}%`,
                  top: `${(selectedTradeCluster.y / svgHeight) * 100}%`,
                  transform: selectedTradeCluster.y < 170
                    ? 'translate(-50%, 14px)'
                    : 'translate(-50%, -100%) translateY(-14px)',
                  zIndex: 80,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                  minWidth: 230,
                  maxWidth: 320,
                  pointerEvents: 'auto',
                  animation: 'fadeIn 0.15s ease'
                }}
                onClick={(e) => e.stopPropagation()}
              >
                {/* Küme Başlığı */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  background: 'var(--ink-primary)',
                  color: '#FAF8F3',
                  padding: '5px 10px',
                  borderRadius: 'var(--radius-xs)',
                  fontSize: '0.72rem',
                  fontWeight: 600,
                  boxShadow: '0 2px 8px rgba(0,0,0,0.25)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Activity size={12} style={{ color: 'var(--forest-gain)' }} />
                    <span>{selectedTradeCluster.date}</span>
                    <span style={{ opacity: 0.6 }}>•</span>
                    <span>{selectedTradeCluster.trades.length} İşlem</span>
                  </div>
                  <button
                    onClick={() => setSelectedTradeCluster(null)}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#FAF8F3',
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      lineHeight: 1,
                      padding: '0 2px',
                      display: 'flex',
                      alignItems: 'center'
                    }}
                    title="Kapat"
                  >
                    <X size={13} />
                  </button>
                </div>

                {/* 3-4 Kart Üst Üste Hizalanmış Liste */}
                {selectedTradeCluster.trades.map((tr, i) => {
                  const isBuy = tr.action.includes('ALIM');
                  const sigColor = tickerColorMap[tr.ticker] || (isBuy ? 'var(--forest-gain)' : 'var(--madder-loss)');
                  return (
                    <div
                      key={i}
                      style={{
                        background: 'var(--paper-card)',
                        border: '1px solid var(--rule-strong)',
                        borderLeft: `4px solid ${isBuy ? 'var(--forest-gain)' : 'var(--madder-loss)'}`,
                        borderRadius: 'var(--radius-xs)',
                        padding: '8px 12px',
                        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.16)',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 4
                      }}
                    >
                      {/* Üst Satır: Sol üstte Hisse + Eylem, Sağ üstte O Anki Fiyat */}
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <span style={{ width: 8, height: 8, borderRadius: '50%', background: sigColor }} />
                          <span style={{ fontWeight: 800, fontSize: '0.92rem', color: 'var(--ink-primary)' }}>
                            {tr.ticker}
                          </span>
                          <span
                            style={{
                              fontSize: '0.65rem',
                              fontWeight: 700,
                              padding: '1px 5px',
                              borderRadius: 'var(--radius-xs)',
                              background: isBuy ? 'var(--forest-tint)' : 'var(--madder-tint)',
                              color: isBuy ? 'var(--forest-gain)' : 'var(--madder-loss)',
                            }}
                          >
                            {isBuy ? 'ALIM' : 'KÂR AL / SATIŞ'}
                          </span>
                        </div>
                        <div className="tabular" style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--ink-primary)' }}>
                          ${tr.price.toFixed(2)}
                        </div>
                      </div>

                      {/* Alt Satır: Eski Ağırlık → Yeni Ağırlık */}
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--ink-secondary)', marginTop: 2 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <span>Ağırlık:</span>
                          <span className="tabular" style={{ fontWeight: 600 }}>%{tr.prev_weight_pct.toFixed(1)}</span>
                          <span style={{ color: 'var(--ink-muted)', margin: '0 2px' }}>→</span>
                          <span className="tabular" style={{ fontWeight: 700, color: isBuy ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
                            %{tr.new_weight_pct.toFixed(1)}
                          </span>
                        </div>
                        {tr.delta_notional > 0 && (
                          <span className="tabular" style={{ fontSize: '0.72rem', color: 'var(--ink-muted)' }}>
                            ${tr.delta_notional.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Grafik Altı Açıklama & Lejant */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, fontSize: '0.78rem', color: 'var(--ink-secondary)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 14, height: 3, background: 'var(--forest-gain)', display: 'inline-block' }} />
                <span style={{ fontWeight: 700, color: 'var(--forest-gain)' }}>Model AI Portföyü (Dinamik)</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 14, height: 3, background: '#475569', borderTop: '2px dashed #475569', display: 'inline-block' }} />
                <span style={{ fontWeight: 700, color: '#475569' }}>Başlangıç Al-Tut Ağırlıkları (Statik B&amp;H)</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--forest-gain)', display: 'inline-block' }} />
                <span>Fırsat / Dip Alımı</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--madder-loss)', display: 'inline-block' }} />
                <span>Kâr Realizasyonu / Nakte Geçiş</span>
              </div>
            </div>

            <div style={{ fontStyle: 'italic', color: 'var(--ink-muted)', fontSize: '0.72rem' }}>
              * Başlangıç Al-Tut ve hisse butonlarına tıklayarak çizgileri açıp kapatabilir, fareyle grafik üzerinde gezinebilirsiniz.
            </div>
          </div>
        </div>
      )}

      {/* ── VARLIK DEĞERLEME VE ÇIPA TABLOSU ──────────────────────────────── */}
      {simResult && simResult.assets_summary && (
        <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)', padding: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <div>
              <div style={{ fontSize: '0.68rem', fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--ink-muted)' }}>
                Orta-Uzun Vadeli Çıpa Seviyeleri
              </div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--ink-primary)', margin: '2px 0 0 0' }}>
                Varlık Bazında Değerleme Bölgeleri &amp; Ağırlık Seyri
              </h3>
            </div>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table className="broadsheet-table">
              <thead>
                <tr>
                  <th>Varlık</th>
                  <th style={{ textAlign: 'right' }}>Başlangıç Fiyatı (P₀)</th>
                  <th style={{ textAlign: 'right' }}>Son Fiyat (P_t)</th>
                  <th style={{ textAlign: 'right' }}>Al-Tut Getiri (%)</th>
                  <th style={{ textAlign: 'right' }}>Ucuz Bölge (P_ucuz)</th>
                  <th style={{ textAlign: 'right' }}>Pahalı Bölge (P_pahali)</th>
                  <th style={{ textAlign: 'right' }}>Başlangıç %</th>
                  <th style={{ textAlign: 'right' }}>Nihai %</th>
                  <th>Model Değerleme Tespiti</th>
                </tr>
              </thead>
              <tbody>
                {simResult.assets_summary.map((a) => {
                  const sigColor = tickerColorMap[a.ticker] || '#2563EB';
                  const isUp = a.buy_hold_return_pct >= 0;
                  return (
                    <tr key={a.ticker}>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span style={{ width: 8, height: 8, borderRadius: '50%', background: sigColor }} />
                          <span className="tabular" style={{ fontWeight: 700, color: 'var(--ink-primary)' }}>{a.ticker}</span>
                        </div>
                      </td>
                      <td className="tabular" style={{ textAlign: 'right' }}>${a.p0.toFixed(2)}</td>
                      <td className="tabular" style={{ textAlign: 'right', fontWeight: 600 }}>${a.p_final.toFixed(2)}</td>
                      <td className="tabular" style={{ textAlign: 'right', fontWeight: 700, color: isUp ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
                        {isUp ? '+' : ''}{a.buy_hold_return_pct.toFixed(2)}%
                      </td>
                      <td className="tabular" style={{ textAlign: 'right', color: 'var(--forest-gain)', fontWeight: 600 }}>
                        ${a.cheap_price.toFixed(2)}
                      </td>
                      <td className="tabular" style={{ textAlign: 'right', color: 'var(--madder-loss)', fontWeight: 600 }}>
                        ${a.expensive_price.toFixed(2)}
                      </td>
                      <td className="tabular" style={{ textAlign: 'right' }}>%{a.initial_weight_pct.toFixed(1)}</td>
                      <td className="tabular" style={{ textAlign: 'right', fontWeight: 700 }}>%{a.final_weight_pct.toFixed(1)}</td>
                      <td>
                        <span 
                          style={{
                            fontSize: '0.72rem',
                            fontWeight: 700,
                            padding: '2px 8px',
                            borderRadius: 'var(--radius-xs)',
                            background: a.valuation_status.includes('UCUZ') ? 'var(--forest-tint)' : a.valuation_status.includes('PAHALI') ? 'var(--madder-tint)' : 'var(--paper-elevated)',
                            color: a.valuation_status.includes('UCUZ') ? 'var(--forest-gain)' : a.valuation_status.includes('PAHALI') ? 'var(--madder-loss)' : 'var(--ink-secondary)'
                          }}
                        >
                          {a.valuation_status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── İŞLEM GÜNLÜĞÜ VE MODEL GEREKÇELERİ ────────────────────────────── */}
      {simResult && (
        <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)', padding: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <div>
              <div style={{ fontSize: '0.68rem', fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--ink-muted)' }}>
                Rebalance &amp; Sermaye Rotasyonu Kararları
              </div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--ink-primary)', margin: '2px 0 0 0' }}>
                Gerçekleşen İşlem Kütüğü ({simResult.trades.length} İşlem)
              </h3>
            </div>
            <span style={{ fontSize: '0.75rem', color: 'var(--ink-muted)' }}>
              10 Bps Sürtünme / Komisyon Maliyeti Dahildir
            </span>
          </div>

          {simResult.trades.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--ink-muted)', fontSize: '0.88rem' }}>
              Model belirlenen periyotta başlangıç ağırlıklarını korumayı tercih etti.
            </div>
          ) : (
            <div style={{ maxHeight: 380, overflowY: 'auto' }}>
              <table className="broadsheet-table">
                <thead>
                  <tr>
                    <th>Tarih</th>
                    <th>Varlık</th>
                    <th>Eylem</th>
                    <th style={{ textAlign: 'right' }}>İşlem Fiyatı</th>
                    <th style={{ textAlign: 'right' }}>Adet</th>
                    <th style={{ textAlign: 'right' }}>Nominal Tutar ($)</th>
                    <th style={{ textAlign: 'right' }}>Ağırlık Değişimi</th>
                    <th>Model Karar Gerekçesi</th>
                  </tr>
                </thead>
                <tbody>
                  {simResult.trades.map((tr, i) => {
                    const isBuy = tr.action.includes('ALIM');
                    return (
                      <tr key={i}>
                        <td className="tabular" style={{ fontWeight: 600 }}>{tr.date}</td>
                        <td className="tabular" style={{ fontWeight: 700 }}>{tr.ticker}</td>
                        <td>
                          <span 
                            style={{
                              fontSize: '0.72rem',
                              fontWeight: 700,
                              padding: '2px 6px',
                              borderRadius: 'var(--radius-xs)',
                              background: isBuy ? 'var(--forest-tint)' : 'var(--madder-tint)',
                              color: isBuy ? 'var(--forest-gain)' : 'var(--madder-loss)',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: 4
                            }}
                          >
                            {isBuy ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                            {tr.action} {tr.badge}
                          </span>
                        </td>
                        <td className="tabular" style={{ textAlign: 'right' }}>${tr.price.toFixed(2)}</td>
                        <td className="tabular" style={{ textAlign: 'right' }}>{tr.shares.toFixed(2)}</td>
                        <td className="tabular" style={{ textAlign: 'right', fontWeight: 600 }}>
                          ${tr.delta_notional.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                        </td>
                        <td className="tabular" style={{ textAlign: 'right' }}>
                          %{tr.prev_weight_pct.toFixed(1)} → %{tr.new_weight_pct.toFixed(1)}
                        </td>
                        <td style={{ fontSize: '0.78rem', color: 'var(--ink-secondary)', maxWidth: 360 }}>
                          {tr.reason}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

    </div>
  );
};
