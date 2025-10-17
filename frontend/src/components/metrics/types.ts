/**
 * Component-specific types for metrics components.
 * 
 * These types are specific to the metrics UI components and extend
 * the base types from src/types/metrics.ts.
 */

import type { 
  ModelClusterRow, 
  MetricsFilters, 
  MetricsPlotType,
  SignificanceType 
} from '../../types/metrics';

/**
 * Chart configuration for individual chart components
 */
export interface ChartConfig {
  plotType: MetricsPlotType;
  qualityMetric?: string;
  showCI: boolean;
  topN: number;
  models: string[];
  height?: number;
  width?: number;
}

/**
 * Data point for Plotly charts
 */
export interface ChartDataPoint {
  x: string;  // Cluster name (truncated for display)
  y: number;  // Value (proportion, quality, etc.)
  hovertext: string;  // Full cluster name for hover
  error_y?: {
    type: 'data';
    array: number[];
    visible: boolean;
  };
  marker?: {
    color: string;
    line?: {
      color: string;
      width: number;
    };
  };
}

/**
 * Plotly trace data structure
 */
export interface PlotlyTrace {
  type: 'bar' | 'scatter';
  x: string[];
  y: number[];
  hovertext: string[];
  name: string;
  marker?: {
    color: string | string[];
    line?: {
      color: string | string[];
      width: number;
    };
  };
  error_y?: {
    type: 'data';
    array: number[];
    visible: boolean;
  };
}

/**
 * Props for significance badge component
 */
export interface SignificanceBadgeProps {
  type: SignificanceType;
  significant: boolean;
  size?: 'small' | 'medium';
  className?: string;
}

/**
 * Props for tag chips component
 */
export interface TagChipsProps {
  tags: string[];
  maxTags?: number;
  size?: 'small' | 'medium';
  color?: 'primary' | 'secondary' | 'default';
}

/**
 * Cluster item data for model cards
 */
export interface ClusterItemData {
  name: string;
  proportion: number;
  proportion_delta: number;
  quality?: Record<string, number>;
  quality_delta?: Record<string, number>;
  proportion_delta_significant: boolean;
  quality_delta_significant?: Record<string, boolean>;
  metadata?: {
    group?: string;
    tags?: string[];
  };
  size: number;
}

/**
 * Model card component props
 */
export interface ModelCardProps {
  modelName: string;
  clusters: ClusterItemData[];
  totalBattles: number;
  selectedQualityMetric: string;
  maxClusters?: number;
}

/**
 * Filter summary props for displaying active filters
 */
export interface FilterSummaryProps {
  filters: MetricsFilters;
  totalRows: number;
  filteredRows: number;
  onClearFilter: (filterKey: keyof MetricsFilters) => void;
  onClearAll: () => void;
}

/**
 * Chart theme configuration
 */
export interface ChartTheme {
  colors: {
    primary: string;
    secondary: string;
    positive: string;
    negative: string;
    neutral: string;
  };
  fonts: {
    family: string;
    size: number;
  };
  margins: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
}

/**
 * Export all types from base metrics types for convenience
 */
export type {
  MetricsFilters,
  MetricsPlotType,
  MetricsSortOption,
  ModelClusterRow,
  ModelBenchmarkRow,
  ModelClusterPayload,
  ModelBenchmarkPayload,
  MetricsSummary,
  SignificanceType
} from '../../types/metrics';