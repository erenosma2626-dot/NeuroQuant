export interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface IndicatorPoint {
  time: string;
  value: number;
}

export interface MacdPoint {
  time: string;
  macd: number;
  signal: number;
  hist: number;
}

export interface MarketData {
  ticker: string;
  benchmark: string;
  current_price: number;
  change_pct: number;
  volume: number;
  volume_ratio: number;
  beta: number;
  alpha_20d_cum: number;
  dist_sma50_pct: number;
  dist_sma200_pct: number;
  is_golden_cross: boolean;
  is_above_sma200: boolean;
  candles: Candle[];
  volume_series: { time: string; value: number; color: string }[];
  sma50: IndicatorPoint[];
  sma200: IndicatorPoint[];
  bb_upper: IndicatorPoint[];
  bb_lower: IndicatorPoint[];
  macd: MacdPoint[];
}

export interface ConeStep {
  step: number;
  date: string;
  median_price: number;
  lower_80_price: number;
  upper_80_price: number;
  median_return_pct: number;
}

export interface ForecastData {
  ticker: string;
  benchmark: string;
  current_price: number;
  as_of_date: string;
  median_5d_return_pct: number;
  lower_80_return_pct: number;
  upper_80_return_pct: number;
  up_probability: number;
  decision: string;
  decision_color: string;
  cone_series: ConeStep[];
  features_used: string[];
  engine?: string;
  engine_type?: string;
  available_engines?: { id: string; name: string; badge: string }[];
}

export interface EarningsRecord {
  date: string;
  days_diff: number;
  eps_estimate: number | null;
  reported_eps: number | null;
  surprise_pct: number | null;
}

export interface FundamentalsData {
  ticker: string;
  is_equity: boolean;
  trailing_pe: number | null;
  forward_pe: number | null;
  price_to_book: number | null;
  peg_ratio: number | null;
  market_cap: number | null;
  valuation_status: string;
  valuation_score: number;
  earnings_regime: string;
  days_to_earnings: number | null;
  next_earnings_date: string | null;
  last_eps_surprise_pct: number | null;
  earnings_history: EarningsRecord[];
}

export interface TradeEvent {
  day_index: number;
  date: string;
  action: string;
  badge: string;
  price: number;
  confidence_score: number;
  prev_weight_pct: number;
  new_weight_pct: number;
  stock_value: number;
  cash_value: number;
  total_portfolio: number;
  reasons: string[];
}

export interface SimulationStep {
  step: number;
  date: string;
  price: number;
  ai_equity: number;
  buy_hold_equity: number;
  ai_stock_value: number;
  ai_cash_value: number;
  weight_pct: number;
  confidence_score: number;
  trade_event: TradeEvent | null;
}

export interface SimulationPerformance {
  ai_final_equity: number;
  buy_hold_final_equity: number;
  ai_total_return_pct: number;
  buy_hold_total_return_pct: number;
  alpha_spread_pct: number;
  ai_sharpe: number;
  ai_sortino: number;
  ai_max_drawdown_pct: number;
  buy_hold_max_drawdown_pct: number;
  total_trades: number;
}

export interface SimulationData {
  ticker: string;
  benchmark: string;
  initial_capital: number;
  test_period_days: number;
  start_date: string;
  end_date: string;
  performance: SimulationPerformance;
  trades: TradeEvent[];
  timeline: SimulationStep[];
}

export interface NewsItem {
  title: string;
  link: string;
  source: string;
  published: string;
  elapsed_hours: number;
  decay_weight: number;
  score: number;
  label: string;
}

export interface TopImpactfulNews {
  title: string;
  link: string;
  source: string;
  published: string;
  elapsed_hours: number;
  score: number;
  impact_type: string;
  badge_color: string;
}

export interface NewsData {
  ticker: string;
  total_news_count: number;
  overall_sentiment_score: number;
  overall_label: string;
  top_3_impactful?: TopImpactfulNews[];
  riskiest_headline?: string | null;
  top_catalyst_headline?: string | null;
  news: NewsItem[];
}

export interface AgentCommentData {
  executive_summary: string;
  technical_regime: string;
  fundamental_valuation: string;
  sentiment_and_catalysts: string;
  risk_factors?: string[];
  suggested_action: string;
  strategic_actions?: Array<{ title: string; detail: string }>;
}

export interface ScreenerItem {
  ticker: string;
  name: string;
  sector: string;
  category: string;
  last_close: number;
  change_pct: number;
  dist_sma200_pct: number;
  is_golden_cross: boolean;
  alpha_20d_cum: number;
  beta: number;
  ai_signal: string;
  confidence_score: number;
  volume_ratio: number;
}

export interface PortfolioPosition {
  ticker: string;
  name: string;
  category: string;
  shares: number;
  buy_price: number;
  current_price: number;
  weight_pct: number;
}

export interface UserPortfolio {
  name: string;
  initial_capital: number;
  cash: number;
  positions: PortfolioPosition[];
}

export interface MacroBarometerItem {
  key: string;
  name: string;
  symbol: string;
  value: number;
  unit: string;
  change_pct: number;
  change_5d?: number;
  trend_60d?: number;
  sma50?: number;
  sma200?: number;
  dist_sma200?: number;
  rating?: string;
  status: string;
  tag: string;
  fed_badge?: string;
  irx_rate?: number;
  is_risk_on: boolean;
}

export interface MacroRegimeBreakdown {
  vix_contribution?: number;
  tnx_contribution?: number;
  tnx_trend_contribution?: number;
  fed_stance_contribution?: number;
  cnn_fg_contribution?: number;
  oil_contribution?: number;
  safe_haven_contribution?: number;
  news_sentiment_contribution?: number;
  news_contribution?: number;
  [key: string]: number | undefined;
}

export interface MacroBarometerData {
  as_of: string;
  composite_score: number;
  regime_label: string;
  regime_badge: string;
  regime_color: string;
  investor_note?: string;
  fed_stance?: {
    score: number;
    label: string;
    badge: string;
    yield_spread: number;
    short_rate_3m: number;
  };
  cnn_fear_greed?: {
    score: number;
    rating: string;
    rating_raw: string;
    previous_close: number;
    change_1w: number;
    is_live: boolean;
  };
  barometers: MacroBarometerItem[];
  breakdown?: MacroRegimeBreakdown;
  global_news_sentiment: {
    score: number;
    label: string;
    total_news_count: number;
  };
}
