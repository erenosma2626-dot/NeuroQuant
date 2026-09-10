import React, { useState } from 'react';
import type { NewsData } from '../types';
import { Newspaper, ExternalLink, Globe, Building2, Flame } from 'lucide-react';

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
  const overallScore = activeData?.overall_sentiment_score || 0;
  const isPos = overallScore > 0.10;
  const isNeg = overallScore < -0.10;

  // En etkili 3 haber (varsa top_3_impactful, yoksa ilk 3)
  const topItems = (activeData?.top_3_impactful && activeData.top_3_impactful.length > 0)
    ? activeData.top_3_impactful
    : (activeData?.news || []).slice(0, 3).map((n) => ({
        title: n.title,
        link: n.link,
        source: n.source,
        published: n.published,
        elapsed_hours: n.elapsed_hours,
        score: n.score,
        impact_type: n.score >= 0.20 ? 'GÜÇLÜ KATALİZÖR' : n.score <= -0.20 ? 'KRİTİK RİSK' : 'MAKRO AKIŞ',
        badge_color: n.score >= 0.20 ? 'var(--forest-gain)' : n.score <= -0.20 ? 'var(--madder-loss)' : 'var(--cobalt)',
      }));

  return (
    <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)', animation: 'fadeUp 0.35s ease' }}>
      {/* ── HEADER ── */}
      <div style={{
        padding: '1.5rem 2.25rem',
        borderBottom: '1px solid var(--rule-strong)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1.5rem',
      }}>
        {/* Sol Başlık */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span className="section-label">Finansal Duygu &amp; Haber Akışı</span>
            <span style={{
              fontSize: '0.62rem',
              fontWeight: 700,
              padding: '2px 8px',
              borderRadius: 3,
              background: 'var(--paper-elevated)',
              color: 'var(--ink-secondary)',
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
            }}>
              Loughran-McDonald &middot; Sürekli Model
            </span>
          </div>

          <div style={{
            fontFamily: 'var(--font-display)',
            fontSize: '1.35rem',
            fontWeight: 700,
            color: 'var(--ink-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}>
            <Flame size={18} style={{ color: isPos ? 'var(--forest-gain)' : isNeg ? 'var(--madder-loss)' : 'var(--ink-secondary)' }} />
            En Etkili 3 Piyasa Haberi &amp; Katalizör
          </div>
        </div>

        {/* Sağ: Mod Değiştirici & Genel Skor Rozeti */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
          {/* Dual-Mode Toggle */}
          <div style={{
            display: 'inline-flex',
            background: 'var(--paper-elevated)',
            border: '1px solid var(--rule-strong)',
            borderRadius: 'var(--radius-sm)',
            padding: 2,
          }}>
            <button
              onClick={() => setActiveMode('ticker')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '6px 14px',
                border: 'none',
                borderRadius: 'var(--radius-xs)',
                background: activeMode === 'ticker' ? 'var(--paper-card)' : 'transparent',
                color: activeMode === 'ticker' ? 'var(--ink-primary)' : 'var(--ink-muted)',
                fontWeight: activeMode === 'ticker' ? 700 : 500,
                fontSize: '0.78rem',
                cursor: 'pointer',
                boxShadow: activeMode === 'ticker' ? '0 1px 3px rgba(0,0,0,0.06)' : 'none',
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
                padding: '6px 14px',
                border: 'none',
                borderRadius: 'var(--radius-xs)',
                background: activeMode === 'global' ? 'var(--paper-card)' : 'transparent',
                color: activeMode === 'global' ? 'var(--ink-primary)' : 'var(--ink-muted)',
                fontWeight: activeMode === 'global' ? 700 : 500,
                fontSize: '0.78rem',
                cursor: 'pointer',
                boxShadow: activeMode === 'global' ? '0 1px 3px rgba(0,0,0,0.06)' : 'none',
                transition: 'all 0.15s ease',
              }}
            >
              <Globe size={13} />
              Küresel Wall Street
            </button>
          </div>

          {/* Genel Duygu Skoru Rozeti */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '6px 14px',
            background: isPos ? 'var(--forest-tint)' : isNeg ? 'var(--madder-tint)' : 'var(--paper-elevated)',
            border: `1px solid ${isPos ? 'var(--forest-rule)' : isNeg ? 'var(--madder-rule)' : 'var(--rule-strong)'}`,
            borderRadius: 'var(--radius-xs)',
          }}>
            <span style={{ fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--ink-muted)' }}>
              Net Duygu:
            </span>
            <span className="tabular" style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.92rem',
              fontWeight: 700,
              color: isPos ? 'var(--forest-gain)' : isNeg ? 'var(--madder-loss)' : 'var(--ink-primary)',
            }}>
              {overallScore > 0 ? `+${overallScore.toFixed(2)}` : overallScore.toFixed(2)}
            </span>
            <span style={{
              fontSize: '0.72rem',
              fontWeight: 700,
              color: isPos ? 'var(--forest-gain)' : isNeg ? 'var(--madder-loss)' : 'var(--ink-secondary)',
            }}>
              ({activeData?.overall_label || 'NÖTR'})
            </span>
          </div>
        </div>
      </div>

      {/* ── EN ETKİLİ 3 HABER LİSTESİ (SABİT & FERAH KARTLAR) ── */}
      <div style={{ padding: '1.75rem 2.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {topItems.length === 0 ? (
          <div style={{ padding: '2.5rem', textAlign: 'center', color: 'var(--ink-muted)' }}>
            <Newspaper size={28} style={{ margin: '0 auto 8px', opacity: 0.4 }} />
            <div style={{ fontStyle: 'italic', fontFamily: 'var(--font-display)', fontSize: '1rem' }}>
              Bu kategori için henüz haber başlığı taranmadı veya veri bekleniyor…
            </div>
          </div>
        ) : (
          topItems.map((item, idx) => {
            const isItemPos = item.score > 0.15;
            const isItemNeg = item.score < -0.15;
            const accentColor = isItemPos ? 'var(--forest-gain)' : isItemNeg ? 'var(--madder-loss)' : 'var(--cobalt)';

            return (
              <div
                key={idx}
                style={{
                  background: 'var(--paper-card)',
                  border: '1px solid var(--rule-hairline)',
                  borderLeft: `4px solid ${accentColor}`,
                  borderRadius: 'var(--radius-xs)',
                  padding: '1.2rem 1.6rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '1.5rem',
                  transition: 'background 0.15s ease, transform 0.15s ease',
                }}
              >
                {/* Sol: Rozet & Başlık */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                    <span style={{
                      fontSize: '0.65rem',
                      fontWeight: 700,
                      padding: '2px 8px',
                      borderRadius: 3,
                      background: isItemPos ? 'var(--forest-tint)' : isItemNeg ? 'var(--madder-tint)' : 'var(--cobalt-tint)',
                      color: accentColor,
                      letterSpacing: '0.06em',
                      textTransform: 'uppercase',
                    }}>
                      {item.impact_type}
                    </span>

                    <span className="tabular" style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.72rem',
                      fontWeight: 700,
                      color: accentColor,
                    }}>
                      {item.score > 0 ? `+${item.score.toFixed(2)}` : item.score.toFixed(2)}
                    </span>

                    <span style={{ color: 'var(--rule-strong)' }}>&middot;</span>

                    <span style={{ fontSize: '0.72rem', color: 'var(--ink-muted)' }}>
                      {item.source} &middot; {item.elapsed_hours < 24 ? `${Math.round(item.elapsed_hours)} saat önce` : item.published}
                    </span>
                  </div>

                  <a
                    href={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      fontFamily: 'var(--font-display)',
                      fontSize: '1.05rem',
                      fontWeight: 600,
                      color: 'var(--ink-primary)',
                      textDecoration: 'none',
                      lineHeight: 1.4,
                      display: 'block',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--cobalt)')}
                    onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--ink-primary)')}
                  >
                    {item.title}
                  </a>
                </div>

                {/* Sağ: Dış Bağlantı Butonu */}
                <a
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-secondary"
                  style={{
                    padding: '6px 12px',
                    fontSize: '0.74rem',
                    flexShrink: 0,
                    textDecoration: 'none',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 5,
                  }}
                  title="Haberi Kaynağında Oku"
                >
                  Habere Git <ExternalLink size={12} />
                </a>
              </div>
            );
          })
        )}
      </div>

      {/* ── ALT BİLGİLENDİRME (AÇIK & ŞEFFAF) ── */}
      <div style={{
        padding: '0.75rem 2.25rem',
        borderTop: '1px solid var(--rule-hairline)',
        fontSize: '0.72rem',
        color: 'var(--ink-muted)',
        fontFamily: 'var(--font-mono)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'var(--paper-elevated)',
      }}>
        <span>
          Net duygu skoru son 14 günün tüm ({activeData?.total_news_count || 0}) haberinin 24s yarılanma ömürlü üstel zaman çürümesiyle hesaplanmıştır.
        </span>
        <span style={{ color: 'var(--cobalt)', fontWeight: 600 }}>
          Wall Street Duygu Konsensüsü
        </span>
      </div>
    </div>
  );
};

export default NewsFeed;
