import React, { useState } from 'react';
import type { NewsData } from '../types';
import { Newspaper, ExternalLink, ShieldAlert, Sparkles, Clock, Globe, Building2 } from 'lucide-react';

interface NewsFeedProps {
  tickerNews: NewsData | null;
  globalNews: NewsData | null;
  currentTicker: string;
  defaultMode?: 'ticker' | 'global';
}

export const NewsFeed: React.FC<NewsFeedProps> = ({
  tickerNews,
  globalNews,
  currentTicker,
  defaultMode = 'ticker',
}) => {
  const [activeMode, setActiveMode] = useState<'ticker' | 'global'>(defaultMode);

  const activeData = activeMode === 'ticker' ? tickerNews : globalNews;

  const isPos = (activeData?.overall_sentiment_score || 0) > 0.1;
  const isNeg = (activeData?.overall_sentiment_score || 0) < -0.1;

  return (
    <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)', animation: 'fadeUp 0.35s ease' }}>
      {/* ── HEADER ── */}
      <div style={{
        padding: '1rem 2rem',
        borderBottom: '2px solid var(--ink-primary)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 12,
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              fontFamily: 'var(--font-display)',
              fontSize: '1rem',
              fontWeight: 700,
              fontStyle: 'italic',
              color: 'var(--ink-primary)',
            }}>
              Finansal Haber Akışı &amp; Üstel Duygu Motoru
            </div>
            {activeData && (
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.68rem',
                fontWeight: 700,
                padding: '2px 8px',
                borderRadius: 'var(--radius-xs)',
                background: isPos ? 'rgba(20, 83, 45, 0.1)' : isNeg ? 'rgba(136, 19, 55, 0.1)' : 'rgba(87, 83, 78, 0.1)',
                color: isPos ? 'var(--forest-gain)' : isNeg ? 'var(--madder-loss)' : 'var(--ink-secondary)',
                border: `1px solid ${isPos ? 'rgba(20, 83, 45, 0.25)' : isNeg ? 'rgba(136, 19, 55, 0.25)' : 'var(--rule-strong)'}`,
              }}>
                {activeData.overall_label} ({activeData.overall_sentiment_score > 0 ? '+' : ''}{activeData.overall_sentiment_score})
              </span>
            )}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--ink-secondary)', marginTop: 2 }}>
            Google News RSS · 24-Saatlik yarılanma ömürlü üstel zaman çürümesi (time-decay)
          </div>
        </div>

        {/* ── DUAL-MODE SEGMENTED SWITCH ── */}
        <div style={{
          display: 'flex',
          background: 'var(--paper-elevated)',
          border: '1px solid var(--rule-strong)',
          borderRadius: 4,
          padding: 2,
        }}>
          <button
            onClick={() => setActiveMode('ticker')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '5px 12px',
              border: 'none',
              borderRadius: 3,
              background: activeMode === 'ticker' ? 'var(--ink-primary)' : 'transparent',
              color: activeMode === 'ticker' ? 'var(--paper-card)' : 'var(--ink-secondary)',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.72rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            <Building2 size={13} />
            {currentTicker} Başlıkları
          </button>

          <button
            onClick={() => setActiveMode('global')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '5px 12px',
              border: 'none',
              borderRadius: 3,
              background: activeMode === 'global' ? 'var(--ink-primary)' : 'transparent',
              color: activeMode === 'global' ? 'var(--paper-card)' : 'var(--ink-secondary)',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.72rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            <Globe size={13} />
            Küresel Wall Street
          </button>
        </div>
      </div>

      {/* ── ALERTS (RISK & CATALYST) ── */}
      {activeData && (activeData.riskiest_headline || activeData.top_catalyst_headline) && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: activeData.riskiest_headline && activeData.top_catalyst_headline ? '1fr 1fr' : '1fr',
          borderBottom: '1px solid var(--rule-hairline)',
        }}>
          {activeData.riskiest_headline && (
            <div style={{
              padding: '0.85rem 1.5rem',
              background: 'var(--madder-tint)',
              borderRight: activeData.top_catalyst_headline ? '1px solid var(--madder-rule)' : 'none',
              display: 'flex',
              alignItems: 'flex-start',
              gap: 10,
            }}>
              <ShieldAlert size={16} color="var(--madder-loss)" style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <div style={{ fontSize: '0.6rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--madder-loss)', marginBottom: 2 }}>
                  En Riskli Başlık Alarmı
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--madder-loss)', lineHeight: 1.4, fontWeight: 500 }}>
                  "{activeData.riskiest_headline}"
                </div>
              </div>
            </div>
          )}

          {activeData.top_catalyst_headline && (
            <div style={{
              padding: '0.85rem 1.5rem',
              background: 'var(--forest-tint)',
              display: 'flex',
              alignItems: 'flex-start',
              gap: 10,
            }}>
              <Sparkles size={16} color="var(--forest-gain)" style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <div style={{ fontSize: '0.6rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--forest-gain)', marginBottom: 2 }}>
                  Öne Çıkan Katalizör
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--forest-gain)', lineHeight: 1.4, fontWeight: 500 }}>
                  "{activeData.top_catalyst_headline}"
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── NEWS ITEMS LIST ── */}
      {!activeData || activeData.news.length === 0 ? (
        <div style={{ padding: '3rem 2rem', textAlign: 'center', color: 'var(--ink-muted)' }}>
          <Newspaper size={28} style={{ margin: '0 auto 8px', opacity: 0.5 }} />
          <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: '0.95rem' }}>
            Haber akışı derleniyor veya önbellek yenileniyor…
          </div>
        </div>
      ) : (
        <div style={{ maxHeight: 420, overflowY: 'auto' }}>
          {activeData.news.map((item, idx) => {
            const isItemPos = item.score > 0.15;
            const isItemNeg = item.score < -0.15;
            return (
              <div
                key={idx}
                style={{
                  padding: '1rem 1.75rem',
                  borderBottom: idx < activeData.news.length - 1 ? '1px solid var(--rule-hairline)' : 'none',
                  background: idx % 2 === 0 ? 'var(--paper-card)' : 'var(--paper-elevated)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                  transition: 'background 0.15s ease',
                }}
              >
                {/* Meta line */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.72rem', color: 'var(--ink-muted)' }}>
                    <span style={{ color: 'var(--cobalt)', fontWeight: 700 }}>
                      {item.source}
                    </span>
                    <span>·</span>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                      <Clock size={11} />
                      {item.elapsed_hours}s önce
                    </span>
                    <span>·</span>
                    <span className="tabular" style={{ fontFamily: 'var(--font-mono)' }}>
                      Ağırlık: {item.decay_weight}
                    </span>
                  </div>

                  <span
                    className="tabular"
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      padding: '1px 6px',
                      borderRadius: 2,
                      background: isItemPos ? 'rgba(20, 83, 45, 0.08)' : isItemNeg ? 'rgba(136, 19, 55, 0.08)' : 'rgba(87, 83, 78, 0.08)',
                      color: isItemPos ? 'var(--forest-gain)' : isItemNeg ? 'var(--madder-loss)' : 'var(--ink-secondary)',
                    }}
                  >
                    {item.label} ({item.score > 0 ? '+' : ''}{item.score})
                  </span>
                </div>

                {/* Title & Link */}
                <a
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    color: 'var(--ink-primary)',
                    textDecoration: 'none',
                    fontFamily: 'var(--font-body)',
                    fontSize: '0.86rem',
                    fontWeight: 600,
                    lineHeight: 1.45,
                    display: 'flex',
                    alignItems: 'baseline',
                    justifyContent: 'space-between',
                    gap: 12,
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--cobalt)')}
                  onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--ink-primary)')}
                >
                  <span>{item.title}</span>
                  <ExternalLink size={12} style={{ color: 'var(--ink-muted)', flexShrink: 0, marginTop: 3 }} />
                </a>
              </div>
            );
          })}
        </div>
      )}

      {/* ── FOOTER NOTE ── */}
      <div style={{
        padding: '0.65rem 2rem',
        borderTop: '1px solid var(--rule-strong)',
        fontSize: '0.68rem',
        color: 'var(--ink-muted)',
        fontStyle: 'italic',
        background: 'var(--paper-card)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <span>
          Matematiksel Ağırlıklandırma: w = e^(-0.0288 × saat). 24 saat sonra bir haberin duygu etkisi %50'ye iner.
        </span>
        <span className="tabular" style={{ fontFamily: 'var(--font-mono)' }}>
          {activeData?.total_news_count || 0} Başlık İncelendi
        </span>
      </div>
    </div>
  );
};
