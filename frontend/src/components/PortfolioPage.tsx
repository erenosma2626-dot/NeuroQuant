import React, { useState } from 'react';
import { PlusCircle, MinusCircle, Wallet } from 'lucide-react';
import type { ScreenerItem, UserPortfolio } from '../types';

interface PortfolioPageProps {
  screenerData: ScreenerItem[];
  userPortfolio: UserPortfolio;
  onUpdatePortfolio: (updated: UserPortfolio) => void;
  onSelectTicker: (ticker: string) => void;
  onNavigateTab: (tab: 'dashboard' | 'terminal' | 'simulation' | 'portfolio') => void;
}

// A palette of editorial / restrained colours for allocation bar
const ALLOC_COLORS = ['#14532D', '#1E3A8A', '#881337', '#92400E', '#4B5563', '#166534', '#0F766E', '#4338CA'];

export const PortfolioPage: React.FC<PortfolioPageProps> = ({
  screenerData,
  userPortfolio,
  onUpdatePortfolio,
  onSelectTicker,
  onNavigateTab,
}) => {
  const [selectedTickerToAdd, setSelectedTickerToAdd] = useState<string>(screenerData[0]?.ticker || 'NVDA');
  const [customTickerMode, setCustomTickerMode]       = useState<boolean>(false);
  const [customTickerInput, setCustomTickerInput]     = useState<string>('');
  const [customPriceInput, setCustomPriceInput]       = useState<number>(100);
  const [sharesToAdd, setSharesToAdd]                 = useState<number>(10);

  // Nakit Yönetimi State'leri
  const [cashAction, setCashAction]                   = useState<'add' | 'withdraw'>('add');
  const [cashAmountInput, setCashAmountInput]         = useState<number>(5000);
  const [cashFeedback, setCashFeedback]               = useState<string | null>(null);

  /* ── Portfolio Stats ──────────────────────────────────────── */
  const totalStockVal    = userPortfolio.positions.reduce((s, p) => s + p.shares * p.current_price, 0);
  const totalPortfolioVal = userPortfolio.cash + totalStockVal;
  const initialCap       = userPortfolio.initial_capital;
  const totalGain        = totalPortfolioVal - initialCap;
  const totalReturnPct   = initialCap > 0 ? (totalGain / initialCap) * 100 : 0;
  const isPositive       = totalReturnPct >= 0;

  const weightedBeta = totalStockVal > 0
    ? userPortfolio.positions.reduce((s, p) => {
        const item = screenerData.find((x) => x.ticker === p.ticker);
        return s + (p.shares * p.current_price / totalStockVal) * (item?.beta ?? 1);
      }, 0)
    : 0;

  const weightedAlpha = totalStockVal > 0
    ? userPortfolio.positions.reduce((s, p) => {
        const item = screenerData.find((x) => x.ticker === p.ticker);
        return s + (p.shares * p.current_price / totalStockVal) * (item?.alpha_20d_cum ?? 0);
      }, 0)
    : 0;

  /* ── Nakit Ekle / Çıkar İşlemleri ────────────────────────── */
  const handleCashTransaction = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const amount = Number(cashAmountInput);
    if (isNaN(amount) || amount <= 0) {
      setCashFeedback('Lütfen geçerli bir tutar girin.');
      return;
    }

    if (cashAction === 'add') {
      // Fon Girişi: Nakit ve ana sermaye artırılır
      const newCash = userPortfolio.cash + amount;
      const newInitial = userPortfolio.initial_capital + amount;
      onUpdatePortfolio({
        ...userPortfolio,
        cash: newCash,
        initial_capital: newInitial,
      });
      setCashFeedback(`+${amount.toLocaleString('tr-TR')} ₺ nakit başarıyla eklendi.`);
    } else {
      // Fon Çıkışı: Serbest nakit kontrolü
      if (amount > userPortfolio.cash) {
        setCashFeedback(`Yetersiz serbest nakit! Mevcut nakit: ${userPortfolio.cash.toLocaleString('tr-TR')} ₺`);
        return;
      }
      const newCash = userPortfolio.cash - amount;
      const newInitial = Math.max(0, userPortfolio.initial_capital - amount);
      onUpdatePortfolio({
        ...userPortfolio,
        cash: newCash,
        initial_capital: newInitial,
      });
      setCashFeedback(`-${amount.toLocaleString('tr-TR')} ₺ nakit çekimi gerçekleştirildi.`);
    }

    setTimeout(() => setCashFeedback(null), 4000);
  };

  const handleQuickCash = (amount: number, type: 'add' | 'withdraw') => {
    setCashAction(type);
    setCashAmountInput(amount);
    if (type === 'add') {
      const newCash = userPortfolio.cash + amount;
      const newInitial = userPortfolio.initial_capital + amount;
      onUpdatePortfolio({
        ...userPortfolio,
        cash: newCash,
        initial_capital: newInitial,
      });
      setCashFeedback(`+${amount.toLocaleString('tr-TR')} ₺ nakit eklendi.`);
    } else {
      if (amount > userPortfolio.cash) {
        setCashFeedback(`Yetersiz nakit! En fazla ${userPortfolio.cash.toLocaleString('tr-TR')} ₺ çekebilirsiniz.`);
        return;
      }
      const newCash = userPortfolio.cash - amount;
      const newInitial = Math.max(0, userPortfolio.initial_capital - amount);
      onUpdatePortfolio({
        ...userPortfolio,
        cash: newCash,
        initial_capital: newInitial,
      });
      setCashFeedback(`-${amount.toLocaleString('tr-TR')} ₺ nakit çekildi.`);
    }
    setTimeout(() => setCashFeedback(null), 4000);
  };

  /* ── Pozisyon Ekle / Çıkar İşlemleri ───────────────────────── */
  const handleAddPosition = (e: React.FormEvent) => {
    e.preventDefault();

    let targetTicker = selectedTickerToAdd;
    let targetPrice = 0;
    let targetName = targetTicker;
    let targetCategory = 'Global';

    if (customTickerMode) {
      targetTicker = customTickerInput.trim().toUpperCase();
      if (!targetTicker) {
        alert('Lütfen bir sembol adı girin.');
        return;
      }
      targetPrice = Number(customPriceInput);
      if (isNaN(targetPrice) || targetPrice <= 0) {
        alert('Lütfen geçerli bir birim fiyat girin.');
        return;
      }
      targetName = targetTicker;
      targetCategory = targetTicker.endsWith('.IS') ? 'BIST' : (targetTicker.includes('-USD') ? 'Crypto' : 'Global');
    } else {
      const asset = screenerData.find((s) => s.ticker === selectedTickerToAdd);
      if (!asset) return;
      targetPrice = asset.last_close;
      targetName = asset.name;
      targetCategory = asset.category;
    }

    const cost = sharesToAdd * targetPrice;
    if (cost > userPortfolio.cash) {
      alert(`Yetersiz nakit! Gerekli: ${cost.toFixed(2)} ₺, Mevcut: ${userPortfolio.cash.toFixed(2)} ₺. Lütfen önce "Nakit Ekle" ile portföyünüze fon sağlayın.`);
      return;
    }

    const existingIdx = userPortfolio.positions.findIndex((p) => p.ticker.toUpperCase() === targetTicker.toUpperCase());
    let newPositions = [...userPortfolio.positions];
    if (existingIdx >= 0) {
      const ex = newPositions[existingIdx];
      const newShares = ex.shares + sharesToAdd;
      newPositions[existingIdx] = {
        ...ex,
        shares: newShares,
        buy_price: (ex.shares * ex.buy_price + cost) / newShares,
        current_price: targetPrice,
        weight_pct: 0,
      };
    } else {
      newPositions.push({
        ticker: targetTicker,
        name: targetName,
        category: targetCategory,
        shares: sharesToAdd,
        buy_price: targetPrice,
        current_price: targetPrice,
        weight_pct: 0,
      });
    }

    onUpdatePortfolio({
      ...userPortfolio,
      cash: userPortfolio.cash - cost,
      positions: newPositions
    });

    if (customTickerMode) {
      setCustomTickerInput('');
    }
  };

  const handleRemovePosition = (ticker: string) => {
    const pos = userPortfolio.positions.find((p) => p.ticker === ticker);
    if (!pos) return;
    onUpdatePortfolio({
      ...userPortfolio,
      cash: userPortfolio.cash + pos.shares * pos.current_price,
      positions: userPortfolio.positions.filter((p) => p.ticker !== ticker),
    });
  };

  const cashPct = totalPortfolioVal > 0 ? (userPortfolio.cash / totalPortfolioVal) * 100 : 100;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', animation: 'fadeUp 0.35s ease' }}>

      {/* ── PAGE HEADER ────────────────────────────────────────────── */}
      <div style={{ borderTop: '3px solid var(--ink-primary)', paddingTop: '1.5rem' }}>
        <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 6 }}>
          Portföy Yönetim Masası
        </div>
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
          <div>
            <h1 style={{ fontFamily: 'var(--font-display)', fontSize: '2rem', fontWeight: 700, color: 'var(--ink-primary)', letterSpacing: '-0.02em', lineHeight: 1 }}>
              Portföy Yöneticisi &amp; Sermaye Masası
            </h1>
            <p style={{ color: 'var(--ink-secondary)', fontSize: '0.88rem', marginTop: 8, lineHeight: 1.6, maxWidth: 800 }}>
              Kendi çok-varlıklı hisse ve kripto sepetinizi oluşturun; fon ekleme/çıkarma işlemleriyle nakit dengenizi yönetin, ağırlıklı risk ve alfa metriklerini takip edin.
            </p>
          </div>

          {/* Nakit Göstergesi Rozeti */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            background: 'var(--paper-card)',
            border: '1px solid var(--rule-strong)',
            padding: '8px 16px',
            borderRadius: 'var(--radius-sm)'
          }}>
            <Wallet size={16} style={{ color: 'var(--forest-gain)' }} />
            <div>
              <div style={{ fontSize: '0.62rem', fontWeight: 600, textTransform: 'uppercase', color: 'var(--ink-muted)' }}>
                Serbest Nakit
              </div>
              <div className="tabular" style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--ink-primary)' }}>
                {userPortfolio.cash.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ₺
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── METRICS STRIP ──────────────────────────────────────────── */}
      <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)' }}>
        <div className="metric-block">
          <div className="metric-label">Toplam Varlık Değeri</div>
          <div className="metric-value tabular">{totalPortfolioVal.toLocaleString('tr-TR', { minimumFractionDigits: 0 })} ₺</div>
          <div className="metric-sub">Sermaye Matrahı: {initialCap.toLocaleString('tr-TR')} ₺</div>
        </div>
        <div className="metric-block">
          <div className="metric-label">Net Getiri / Zarar</div>
          <div className="metric-value tabular" style={{ color: isPositive ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
            {isPositive ? '+' : ''}{totalReturnPct.toFixed(2)}%
          </div>
          <div className="metric-sub" style={{ color: isPositive ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
            {isPositive ? '+' : ''}{totalGain.toLocaleString('tr-TR', { minimumFractionDigits: 0 })} ₺
          </div>
        </div>
        <div className="metric-block">
          <div className="metric-label">Ağırlıklı Beta</div>
          <div className="metric-value tabular" style={{ color: 'var(--cobalt)' }}>{weightedBeta.toFixed(2)}</div>
          <div className="metric-sub">{weightedBeta < 1 ? 'Defansif Portföy' : 'Agresif / Piyasa Üstü'}</div>
        </div>
        <div className="metric-block">
          <div className="metric-label">Sektörel Alfa (20G)</div>
          <div className="metric-value tabular" style={{ color: weightedAlpha >= 0 ? 'var(--forest-gain)' : 'var(--ink-muted)' }}>
            {weightedAlpha >= 0 ? '+' : ''}{weightedAlpha.toFixed(2)}%
          </div>
          <div className="metric-sub">Endekse Karşı Göreceli Güç</div>
        </div>
      </div>

      {/* ── ALLOCATION BAR ─────────────────────────────────────────── */}
      <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)', padding: '1.25rem 2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
          <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)' }}>
            Varlık Tahsis Dağılımı
          </div>
          <div className="tabular" style={{ fontSize: '0.75rem', color: 'var(--ink-muted)' }}>
            Nakit: {userPortfolio.cash.toLocaleString('tr-TR', { maximumFractionDigits: 0 })} ₺ ({cashPct.toFixed(1)}%)
          </div>
        </div>

        {/* Stacked allocation bar */}
        <div style={{ height: 10, background: 'var(--rule-light)', display: 'flex', overflow: 'hidden', borderRadius: 2 }}>
          {/* Nakit bar */}
          <div
            style={{ width: `${cashPct}%`, background: 'var(--rule-strong)', transition: 'width 0.4s ease' }}
            title={`Nakit: ${cashPct.toFixed(1)}%`}
          />
          {/* Pozisyon barları */}
          {userPortfolio.positions.map((pos, i) => {
            const posVal = pos.shares * pos.current_price;
            const posPct = totalPortfolioVal > 0 ? (posVal / totalPortfolioVal) * 100 : 0;
            return (
              <div
                key={pos.ticker}
                style={{
                  width: `${posPct}%`,
                  background: ALLOC_COLORS[i % ALLOC_COLORS.length],
                  transition: 'width 0.4s ease',
                }}
                title={`${pos.ticker}: ${posPct.toFixed(1)}%`}
              />
            );
          })}
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', gap: '1.5rem', marginTop: 10, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 8, height: 8, background: 'var(--rule-strong)', borderRadius: 1 }} />
            <span style={{ fontSize: '0.72rem', color: 'var(--ink-muted)' }}>Nakit ({cashPct.toFixed(1)}%)</span>
          </div>
          {userPortfolio.positions.map((pos, i) => {
            const posVal = pos.shares * pos.current_price;
            const posPct = totalPortfolioVal > 0 ? (posVal / totalPortfolioVal) * 100 : 0;
            return (
              <div key={pos.ticker} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 8, height: 8, background: ALLOC_COLORS[i % ALLOC_COLORS.length], borderRadius: 1 }} />
                <span style={{ fontSize: '0.72rem', color: 'var(--ink-secondary)' }}>
                  {pos.ticker} ({posPct.toFixed(1)}%)
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── NAKİT YÖNETİMİ PANELİ (Fon Ekle / Çıkar) ────────────────── */}
      <div className="panel" style={{ borderTop: '3px solid var(--cobalt)', padding: '1.5rem 2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.15rem', fontWeight: 700, color: 'var(--ink-primary)' }}>
              Nakit &amp; Sermaye Masası
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--ink-secondary)', marginTop: 2 }}>
              Portföyünüze sermaye ekleyin veya serbest nakit çekerek likiditeyi ayarlayın.
            </div>
          </div>

          {/* Eylem Seçici (Nakit Ekle vs Çıkar) */}
          <div style={{ display: 'flex', gap: 4, background: 'var(--paper-elevated)', padding: 3, borderRadius: 'var(--radius-sm)' }}>
            <button
              className={`filter-btn ${cashAction === 'add' ? 'active' : ''}`}
              onClick={() => setCashAction('add')}
              style={{ fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: 5 }}
            >
              <PlusCircle size={13} style={{ color: 'var(--forest-gain)' }} />
              Nakit Ekle (Fon Girişi)
            </button>
            <button
              className={`filter-btn ${cashAction === 'withdraw' ? 'active' : ''}`}
              onClick={() => setCashAction('withdraw')}
              style={{ fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: 5 }}
            >
              <MinusCircle size={13} style={{ color: 'var(--madder-loss)' }} />
              Nakit Çıkar (Fon Çıkışı)
            </button>
          </div>
        </div>

        {/* Hızlı İşlem Butonları & Tutar Girişi */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
          <form onSubmit={handleCashTransaction} style={{ display: 'flex', alignItems: 'center', gap: 8, flex: '1 1 280px' }}>
            <div style={{ position: 'relative', flex: 1 }}>
              <input
                type="number"
                min={1}
                step={100}
                value={cashAmountInput}
                onChange={(e) => setCashAmountInput(Number(e.target.value))}
                className="search-input"
                style={{ width: '100%', padding: '8px 32px 8px 12px', fontSize: '0.85rem' }}
                placeholder="Tutar girin (₺)..."
              />
              <span style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', fontSize: '0.8rem', color: 'var(--ink-muted)', fontWeight: 600 }}>
                ₺
              </span>
            </div>
            <button
              type="submit"
              className="btn btn-primary"
              style={{
                background: cashAction === 'add' ? 'var(--forest-gain)' : 'var(--madder-loss)',
                borderColor: cashAction === 'add' ? 'var(--forest-gain)' : 'var(--madder-loss)',
                fontSize: '0.8rem',
                padding: '8px 16px'
              }}
            >
              {cashAction === 'add' ? (
                <>+ Nakit Ekle</>
              ) : (
                <>- Nakit Çıkar</>
              )}
            </button>
          </form>

          {/* Hızlı Kısayol Butonları */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--ink-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Hızlı:
            </span>
            {cashAction === 'add' ? (
              <>
                <button className="btn btn-secondary" onClick={() => handleQuickCash(1000, 'add')} style={{ fontSize: '0.72rem', padding: '4px 10px' }}>
                  +1.000 ₺
                </button>
                <button className="btn btn-secondary" onClick={() => handleQuickCash(5000, 'add')} style={{ fontSize: '0.72rem', padding: '4px 10px' }}>
                  +5.000 ₺
                </button>
                <button className="btn btn-secondary" onClick={() => handleQuickCash(10000, 'add')} style={{ fontSize: '0.72rem', padding: '4px 10px' }}>
                  +10.000 ₺
                </button>
                <button className="btn btn-secondary" onClick={() => handleQuickCash(50000, 'add')} style={{ fontSize: '0.72rem', padding: '4px 10px' }}>
                  +50.000 ₺
                </button>
              </>
            ) : (
              <>
                <button className="btn btn-secondary" onClick={() => handleQuickCash(1000, 'withdraw')} style={{ fontSize: '0.72rem', padding: '4px 10px' }}>
                  -1.000 ₺
                </button>
                <button className="btn btn-secondary" onClick={() => handleQuickCash(5000, 'withdraw')} style={{ fontSize: '0.72rem', padding: '4px 10px' }}>
                  -5.000 ₺
                </button>
                {userPortfolio.cash > 0 && (
                  <button className="btn btn-secondary" onClick={() => handleQuickCash(userPortfolio.cash, 'withdraw')} style={{ fontSize: '0.72rem', padding: '4px 10px', color: 'var(--madder-loss)' }}>
                    Tüm Serbest Nakti Çek
                  </button>
                )}
              </>
            )}
          </div>
        </div>

        {/* Geribildirim Notu */}
        {cashFeedback && (
          <div style={{
            marginTop: 12,
            padding: '6px 12px',
            borderRadius: 'var(--radius-xs)',
            background: cashFeedback.includes('Yetersiz') ? 'var(--madder-tint)' : 'var(--forest-tint)',
            color: cashFeedback.includes('Yetersiz') ? 'var(--madder-loss)' : 'var(--forest-gain)',
            fontSize: '0.78rem',
            fontWeight: 600,
            display: 'inline-block'
          }}>
            {cashFeedback}
          </div>
        )}
      </div>

      {/* ── POSITIONS TABLE & ADD POSITION FORM ─────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '2rem', alignItems: 'start' }}>

        {/* Current Positions */}
        <div className="panel" style={{ borderTop: '2px solid var(--ink-primary)' }}>
          <div style={{ padding: '1rem 1.5rem', borderBottom: '2px solid var(--ink-primary)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.15rem', fontWeight: 700, color: 'var(--ink-primary)' }}>
              Mevcut Pozisyonlar ({userPortfolio.positions.length})
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--ink-secondary)' }}>
              Hisse Varlık Değeri: <strong className="tabular">{totalStockVal.toLocaleString('tr-TR', { minimumFractionDigits: 0 })} ₺</strong>
            </div>
          </div>

          {userPortfolio.positions.length === 0 ? (
            <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--ink-muted)' }}>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', fontStyle: 'italic', marginBottom: 6 }}>
                Portföyde henüz hisse pozisyonu bulunmuyor.
              </div>
              <div style={{ fontSize: '0.8rem' }}>
                Sağdaki formdan dilediğiniz hisseyi seçip veya arayıp portföyünüze ekleyebilirsiniz.
              </div>
            </div>
          ) : (
            <table className="broadsheet-table">
              <thead>
                <tr>
                  <th style={{ paddingLeft: '2rem' }}>Varlık</th>
                  <th>Adet</th>
                  <th>Maliyet (₺)</th>
                  <th>Güncel (₺)</th>
                  <th>Değer (₺)</th>
                  <th>Kar / Zarar</th>
                  <th style={{ textAlign: 'right', paddingRight: '2rem' }}>İşlem</th>
                </tr>
              </thead>
              <tbody>
                {userPortfolio.positions.map((pos) => {
                  const marketVal     = pos.shares * pos.current_price;
                  const gain          = marketVal - pos.shares * pos.buy_price;
                  const gainPct       = pos.buy_price > 0 ? (gain / (pos.shares * pos.buy_price)) * 100 : 0;
                  const isProfitable  = gain >= 0;

                  return (
                    <tr key={pos.ticker}>
                      <td style={{ paddingLeft: '2rem' }}>
                        <div
                          style={{ cursor: 'pointer' }}
                          onClick={() => { onSelectTicker(pos.ticker); onNavigateTab('terminal'); }}
                          title="Kantitatif Terminalde Aç"
                        >
                          <span className="tabular" style={{ fontWeight: 700, color: 'var(--ink-primary)', textDecoration: 'underline' }}>
                            {pos.ticker}
                          </span>
                          <div style={{ fontSize: '0.72rem', color: 'var(--ink-muted)', marginTop: 1 }}>{pos.name}</div>
                        </div>
                      </td>
                      <td className="tabular" style={{ fontWeight: 600, color: 'var(--ink-primary)' }}>{pos.shares}</td>
                      <td className="tabular" style={{ color: 'var(--ink-secondary)' }}>{pos.buy_price.toFixed(2)}</td>
                      <td className="tabular" style={{ fontWeight: 700, color: 'var(--ink-primary)' }}>{pos.current_price.toFixed(2)}</td>
                      <td className="tabular" style={{ fontWeight: 600, color: 'var(--cobalt)' }}>{marketVal.toFixed(0)} ₺</td>
                      <td>
                        <span className={`signal ${isProfitable ? 'signal-buy' : 'signal-sell'} tabular`}>
                          {isProfitable ? '+' : ''}{gainPct.toFixed(2)}%
                        </span>
                      </td>
                      <td style={{ textAlign: 'right', paddingRight: '2rem' }}>
                        <button
                          onClick={() => handleRemovePosition(pos.ticker)}
                          className="btn btn-secondary"
                          style={{ padding: '3px 8px', fontSize: '0.72rem', color: 'var(--madder-loss)', borderColor: 'var(--madder-rule)' }}
                          title="Pozisyonu Kapat (Nakte Dön)"
                        >
                          Kapat ✕
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* Add position form */}
        <div className="panel" style={{ borderTop: '3px solid var(--forest-gain)' }}>
          <div style={{ padding: '1rem 1.5rem', borderBottom: '2px solid var(--ink-primary)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', fontWeight: 700, fontStyle: 'italic', color: 'var(--ink-primary)' }}>
              Portföye Varlık Ekle
            </div>
            <button
              className="btn btn-secondary"
              onClick={() => setCustomTickerMode(!customTickerMode)}
              style={{ fontSize: '0.7rem', padding: '2px 8px' }}
            >
              {customTickerMode ? 'Listeden Seç' : '+ Özel Sembol Gir'}
            </button>
          </div>

          <form onSubmit={handleAddPosition} style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: 14 }}>
            {!customTickerMode ? (
              <div>
                <label style={{ fontSize: '0.65rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', display: 'block', marginBottom: 6 }}>
                  Listeden Varlık Seçin
                </label>
                <select
                  value={selectedTickerToAdd}
                  onChange={(e) => setSelectedTickerToAdd(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 10px',
                    background: 'var(--paper-card)',
                    border: '1px solid var(--rule-strong)',
                    borderRadius: 'var(--radius-sm)',
                    color: 'var(--ink-primary)',
                    fontFamily: 'var(--font-body)',
                    fontSize: '0.82rem',
                    outline: 'none',
                  }}
                >
                  {screenerData.map((item) => (
                    <option key={item.ticker} value={item.ticker} style={{ background: '#FAF8F3' }}>
                      {item.ticker} — {item.name} ({item.last_close.toFixed(2)} ₺)
                    </option>
                  ))}
                </select>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div>
                  <label style={{ fontSize: '0.65rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', display: 'block', marginBottom: 4 }}>
                    Özel Sembol (Yahoo Ticker)
                  </label>
                  <input
                    type="text"
                    placeholder="Örn: AAPL, GARAN.IS, MSFT, KO..."
                    value={customTickerInput}
                    onChange={(e) => setCustomTickerInput(e.target.value.toUpperCase())}
                    className="search-input"
                    style={{ width: '100%', padding: '8px 10px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.65rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', display: 'block', marginBottom: 4 }}>
                    Birim Alış Fiyatı (₺ / $)
                  </label>
                  <input
                    type="number"
                    min={0.01}
                    step={0.01}
                    value={customPriceInput}
                    onChange={(e) => setCustomPriceInput(Number(e.target.value))}
                    className="search-input"
                    style={{ width: '100%', padding: '8px 10px' }}
                  />
                </div>
              </div>
            )}

            <div>
              <label style={{ fontSize: '0.65rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', display: 'block', marginBottom: 6 }}>
                Lot / Adet Miktarı
              </label>
              <input
                type="number"
                min={1}
                value={sharesToAdd}
                onChange={(e) => setSharesToAdd(Math.max(1, Number(e.target.value)))}
                className="search-input"
                style={{ width: '100%', padding: '8px 10px' }}
              />
            </div>

            {/* Cost estimate */}
            {(() => {
              let price = 0;
              if (customTickerMode) {
                price = Number(customPriceInput) || 0;
              } else {
                const item = screenerData.find((s) => s.ticker === selectedTickerToAdd);
                price = item ? item.last_close : 0;
              }
              const cost = price * sharesToAdd;
              const canAfford = cost <= userPortfolio.cash;
              return (
                <div style={{
                  padding: '10px 12px',
                  background: 'var(--paper-elevated)',
                  border: `1px solid ${canAfford ? 'var(--rule-hairline)' : 'var(--madder-rule)'}`,
                  borderRadius: 'var(--radius-xs)',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.76rem', marginBottom: 4 }}>
                    <span style={{ color: 'var(--ink-muted)' }}>Tahmini Alış Maliyeti</span>
                    <span className="tabular" style={{ fontWeight: 700, color: canAfford ? 'var(--forest-gain)' : 'var(--madder-loss)' }}>
                      {cost.toFixed(2)} ₺
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem' }}>
                    <span style={{ color: 'var(--ink-muted)' }}>Serbest Nakit</span>
                    <span className="tabular" style={{ color: 'var(--ink-secondary)' }}>{userPortfolio.cash.toFixed(2)} ₺</span>
                  </div>
                </div>
              );
            })()}

            <button type="submit" className="btn btn-primary" style={{ justifyContent: 'center' }}>
              + Pozisyonu Portföye Ekle
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};
