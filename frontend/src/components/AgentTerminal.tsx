import React from 'react';
import type { AgentCommentData } from '../types';

interface AgentTerminalProps {
  comment: AgentCommentData;
}

export const AgentTerminal: React.FC<AgentTerminalProps> = ({ comment }) => {
  // Varsayılan listeli aksiyon maddeleri (eğer API henüz dönmediyse fallback)
  const actions = comment.strategic_actions || [
    { title: 'Taktiksel Yaklaşım', detail: comment.suggested_action },
    { title: 'Piyasa Katalizörü', detail: comment.sentiment_and_catalysts },
    { title: 'Risk Disiplini', detail: comment.risk_factors?.[0] || '200 günlük ortalama altında zarar kes disiplini' }
  ];

  return (
    <div className="panel" style={{ borderTop: '2px solid var(--ink-secondary)', animation: 'fadeUp 0.35s ease' }}>
      {/* ── HEADER ── */}
      <div style={{
        padding: '1.25rem 2.25rem',
        borderBottom: '1px solid var(--rule-strong)',
        display: 'flex',
        alignItems: 'baseline',
        justifyContent: 'space-between',
      }}>
        <div style={{
          fontFamily: 'var(--font-display)',
          fontSize: '1.25rem',
          fontWeight: 700,
          color: 'var(--ink-primary)',
        }}>
          Stratejist Değerlendirmesi
        </div>
      </div>

      {/* ── YÖNETİCİ ÖZETİ ── */}
      <div style={{
        padding: '1.5rem 2.25rem',
        borderBottom: '1px solid var(--rule-hairline)',
        background: 'var(--paper-card)',
      }}>
        <p style={{
          fontFamily: 'var(--font-display)',
          fontSize: '1.1rem',
          color: 'var(--ink-primary)',
          fontWeight: 500,
          lineHeight: 1.6,
          fontStyle: 'italic',
          margin: 0,
        }}>
          "{comment.executive_summary}"
        </p>
      </div>

      {/* ── TEKNİK & TEMEL GÖRÜNÜM (İKİ SÜTUNLU FERAH IZGARA) ── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        borderBottom: '1px solid var(--rule-hairline)',
      }}>
        <div style={{
          padding: '1.5rem 2.25rem',
          borderRight: '1px solid var(--rule-hairline)',
        }}>
          <div className="section-label" style={{ marginBottom: 6 }}>
            Teknik Rejim &amp; Trend
          </div>
          <div style={{ fontSize: '0.88rem', color: 'var(--ink-secondary)', lineHeight: 1.6 }}>
            {comment.technical_regime}
          </div>
        </div>

        <div style={{
          padding: '1.5rem 2.25rem',
        }}>
          <div className="section-label" style={{ marginBottom: 6 }}>
            Değerleme &amp; Likidite
          </div>
          <div style={{ fontSize: '0.88rem', color: 'var(--ink-secondary)', lineHeight: 1.6 }}>
            {comment.fundamental_valuation}
          </div>
        </div>
      </div>

      {/* ── STRATEJİK YOL HARİTASI (LİSTELİ & AÇIKLAYICI) ── */}
      <div style={{
        padding: '1.75rem 2.25rem',
        background: 'var(--paper-elevated)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
          <div>
            <div className="section-label" style={{ marginBottom: 3 }}>Stratejik Görünüm</div>
            <div style={{
              fontFamily: 'var(--font-display)',
              fontSize: '1.2rem',
              fontWeight: 700,
              color: 'var(--forest-gain)',
            }}>
              {comment.suggested_action}
            </div>
          </div>
        </div>

        {/* 3 Maddeli Açıklayıcı Liste */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '1.25rem',
        }}>
          {actions.map((act, idx) => (
            <div
              key={idx}
              style={{
                background: 'var(--paper-card)',
                border: '1px solid var(--rule-hairline)',
                borderTop: '2.5px solid var(--forest-gain)',
                borderRadius: 'var(--radius-xs)',
                padding: '1.1rem 1.3rem',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
              }}
            >
              <div style={{
                fontSize: '0.72rem',
                fontWeight: 700,
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
                color: 'var(--ink-primary)',
              }}>
                {idx + 1}. {act.title}
              </div>
              <div style={{
                fontSize: '0.84rem',
                color: 'var(--ink-secondary)',
                lineHeight: 1.55,
              }}>
                {act.detail}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AgentTerminal;
