/**
 * Metrics components index file.
 * 
 * Exports all metrics-related components for easy importing.
 */

// Main components
export { MetricsTab } from './MetricsTab';
export { MetricsControlPanel } from './MetricsControlPanel';
export { MetricsMainContent } from './MetricsMainContent';

// Section components
export { BenchmarkSection } from './BenchmarkSection';
export { ClusterPlotsSection } from './ClusterPlotsSection';
export { ModelCardsSection } from './ModelCardsSection';

// Types (re-export for convenience)
export type {
  ChartConfig,
  ChartDataPoint,
  PlotlyTrace,
  SignificanceBadgeProps,
  TagChipsProps,
  ClusterItemData,
  ModelCardProps,
  FilterSummaryProps,
  ChartTheme,
  MetricsFilters,
  MetricsPlotType,
  MetricsSortOption,
  ModelClusterRow,
  ModelBenchmarkRow,
  ModelClusterPayload,
  ModelBenchmarkPayload,
  MetricsSummary,
  SignificanceType
} from './types';

// Default export for main component
export { MetricsTab as default } from './MetricsTab';