import React from 'react';
import type { NewsData } from '../types';
import { Newspaper, ExternalLink, ShieldAlert, Clock } from 'lucide-react';

interface NewsFeedProps {
  data: NewsData;
}

export const NewsFeed: React.FC<NewsFeedProps> = ({ data }) => {
  const isPos = data.overall_sentiment_score > 0.1;
  const isNeg = data.overall_sentiment_score < -0.1;

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">
          <Newspaper size={18} color="#38BDF8" />
          Üstel Zaman Çürümeli Haber Akışı
        </span>
        <span
          className="mono"
          style={{
            fontSize: '0.75rem',
            padding: '2px 8px',
            borderRadius: 4,
            fontWeight: 700,
            background: isPos ? 'var(--bull-glow)' : isNeg ? 'var(--bear-glow)' : 'rgba(148, 163, 184, 0.15)',
            color: isPos ? '#10B981' : isNeg ? '#EF4444' : '#94A3B8',
          }}
        >
          {data.overall_label} ({data.overall_sentiment_score > 0 ? '+' : ''}{data.overall_sentiment_score})
        </span>
      </div>

      {/* En Riskli Başlık */}
      {data.riskiest_headline && (
        <div style={{
          padding: '0.75rem 1rem',
          borderRadius: 'var(--radius-md)',
          background: 'var(--bear-glow)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          marginBottom: '0.75rem',
          display: 'flex',
          alignItems: 'center',
          gap: 10
        }}>
          <ShieldAlert size={18} color="#EF4444" style={{ flexShrink: 0 }} />
          <span style={{ fontSize: '0.8rem', color: '#F8FAFC' }}>
            <strong>Risk Uyarısı:</strong> {data.riskiest_headline}
          </span>
        </div>
      )}

      {/* Haberler Listesi */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 380, overflowY: 'auto' }}>
        {data.news.map((item, idx) => (
          <div
            key={idx}
            style={{
              padding: '0.75rem 1rem',
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-hairline)',
              borderRadius: 'var(--radius-md)',
              display: 'flex',
              flexDirection: 'column',
              gap: 4
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                <span style={{ color: '#38BDF8', fontWeight: 600 }}>{item.source}</span>
                <span>•</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Clock size={12} />
                  {item.elapsed_hours}s önce
                </span>
                <span>•</span>
                <span className="mono">Ağırlık: {item.decay_weight}</span>
              </div>

              <span
                className="mono"
                style={{
                  fontSize: '0.7rem',
                  fontWeight: 700,
                  color: item.score > 0.15 ? '#10B981' : item.score < -0.15 ? '#EF4444' : '#94A3B8'
                }}
              >
                {item.label}
              </span>
            </div>

            <a
              href={item.link}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                color: '#F8FAFC',
                textDecoration: 'none',
                fontSize: '0.84rem',
                fontWeight: 500,
                display: 'flex',
                alignItems: 'baseline',
                justifyContent: 'space-between',
                gap: 8
              }}
            >
              <span>{item.title}</span>
              <ExternalLink size={13} color="#64748B" style={{ flexShrink: 0 }} />
            </a>
          </div>
        ))}
      </div>
    </div>
  );
};
