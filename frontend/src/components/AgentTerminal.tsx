import React from 'react';
import type { AgentCommentData } from '../types';
import { Bot, ArrowRightCircle, Sparkles } from 'lucide-react';

interface AgentTerminalProps {
  comment: AgentCommentData;
}

export const AgentTerminal: React.FC<AgentTerminalProps> = ({ comment }) => {
  return (
    <div className="card" style={{ border: '1px solid rgba(56, 189, 248, 0.25)' }}>
      <div className="card-header">
        <span className="card-title">
          <Bot size={18} color="#38BDF8" />
          Kurumsal Yapay Zeka Stratejist Raporu
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.72rem', color: '#38BDF8' }}>
          <Sparkles size={14} />
          <span className="mono">HİBRİT ÇIKARIM</span>
        </div>
      </div>

      {/* Yönetici Özeti (Executive Summary) */}
      <div style={{
        padding: '1rem 1.25rem',
        borderRadius: 'var(--radius-md)',
        background: 'rgba(56, 189, 248, 0.08)',
        border: '1px solid rgba(56, 189, 248, 0.25)',
        marginBottom: '1rem'
      }}>
        <span style={{ fontSize: '0.72rem', color: '#38BDF8', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Yönetici Özeti (Executive Takeaway)
        </span>
        <p style={{ fontSize: '0.92rem', color: '#F8FAFC', marginTop: 4, fontWeight: 500, lineHeight: 1.6 }}>
          {comment.executive_summary}
        </p>
      </div>

      {/* 3 Detay Bölümü */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: '1rem' }}>
        <div style={{ padding: '0.75rem 1rem', background: 'var(--bg-surface)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-hairline)' }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
            Teknik & Makro Rejim
          </span>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: 2 }}>
            {comment.technical_regime}
          </p>
        </div>

        <div style={{ padding: '0.75rem 1rem', background: 'var(--bg-surface)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-hairline)' }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
            Temel Değerleme & Bilanço
          </span>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: 2 }}>
            {comment.fundamental_valuation}
          </p>
        </div>
      </div>

      {/* Önerilen Aksiyon Planı (Action Banner) */}
      <div style={{
        padding: '0.85rem 1.25rem',
        borderRadius: 'var(--radius-md)',
        background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(56, 189, 248, 0.15) 100%)',
        border: '1px solid rgba(16, 185, 129, 0.35)',
        display: 'flex',
        alignItems: 'center',
        gap: 12
      }}>
        <ArrowRightCircle size={22} color="#10B981" style={{ flexShrink: 0 }} />
        <div>
          <span style={{ fontSize: '0.7rem', color: '#10B981', fontWeight: 700, textTransform: 'uppercase' }}>
            Önerilen Portföy Aksiyonu
          </span>
          <div style={{ fontSize: '0.9rem', color: '#F8FAFC', fontWeight: 600, marginTop: 2 }}>
            {comment.suggested_action}
          </div>
        </div>
      </div>
    </div>
  );
};
