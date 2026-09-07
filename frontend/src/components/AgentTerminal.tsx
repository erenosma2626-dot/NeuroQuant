import React from 'react';
import type { AgentCommentData } from '../types';

interface AgentTerminalProps {
  comment: AgentCommentData;
}

export const AgentTerminal: React.FC<AgentTerminalProps> = ({ comment }) => {
  return (
    <div className="panel" style={{ borderTop: '2px solid var(--cobalt)', borderLeft: '3px solid var(--cobalt)' }}>
      {/* Header */}
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
            Kurumsal Yapay Zeka Stratejist Raporu
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--ink-secondary)', marginTop: 2 }}>
            Hibrit çıkarım · Teknik + Temel + Makro sentezi
          </div>
        </div>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.65rem',
          fontWeight: 600,
          letterSpacing: '0.12em',
          textTransform: 'uppercase',
          color: 'var(--cobalt)',
          border: '1px solid var(--cobalt-rule)',
          padding: '3px 8px',
          borderRadius: 'var(--radius-xs)',
          background: 'var(--cobalt-tint)',
        }}>
          NQ-AI v3.0
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
        {/* Executive Summary */}
        <div style={{
          padding: '1.5rem 2rem',
          borderBottom: '1px solid var(--rule-hairline)',
          background: 'var(--cobalt-tint)',
        }}>
          <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--cobalt)', marginBottom: 8 }}>
            Yönetici Özeti
          </div>
          <p style={{
            fontFamily: 'var(--font-display)',
            fontSize: '1rem',
            color: 'var(--ink-primary)',
            fontWeight: 500,
            lineHeight: 1.65,
            fontStyle: 'italic',
          }}>
            "{comment.executive_summary}"
          </p>
        </div>

        {/* Technical Regime */}
        <div style={{
          padding: '1.25rem 2rem',
          borderBottom: '1px solid var(--rule-hairline)',
        }}>
          <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 6 }}>
            Teknik &amp; Makro Rejim
          </div>
          <p style={{ fontSize: '0.88rem', color: 'var(--ink-secondary)', lineHeight: 1.6 }}>
            {comment.technical_regime}
          </p>
        </div>

        {/* Fundamental Valuation */}
        <div style={{
          padding: '1.25rem 2rem',
          borderBottom: '1px solid var(--rule-hairline)',
        }}>
          <div style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--ink-muted)', marginBottom: 6 }}>
            Temel Değerleme &amp; Bilanço
          </div>
          <p style={{ fontSize: '0.88rem', color: 'var(--ink-secondary)', lineHeight: 1.6 }}>
            {comment.fundamental_valuation}
          </p>
        </div>

        {/* Suggested Action */}
        <div style={{
          padding: '1.25rem 2rem',
          background: 'var(--forest-tint)',
          borderTop: '2px solid var(--forest-rule)',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '1.5rem',
        }}>
          <div style={{ flexShrink: 0 }}>
            <div style={{ fontSize: '0.6rem', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--forest-gain)', marginBottom: 4 }}>
              Portföy Aksiyonu
            </div>
            <div style={{
              fontFamily: 'var(--font-display)',
              fontSize: '1.25rem',
              fontWeight: 700,
              color: 'var(--forest-gain)',
              letterSpacing: '-0.01em',
            }}>
              {comment.suggested_action}
            </div>
          </div>
          <div style={{
            width: 1,
            alignSelf: 'stretch',
            background: 'var(--forest-rule)',
            flexShrink: 0,
          }} />
          <p style={{ fontSize: '0.85rem', color: 'var(--ink-secondary)', lineHeight: 1.6, paddingTop: 2 }}>
            Yapay zeka modelinin teknik, temel ve makro analizine dayalı portföy aksiyonu önerisi. Bu öneri bir finansal tavsiye niteliği taşımamaktadır.
          </p>
        </div>
      </div>
    </div>
  );
};
