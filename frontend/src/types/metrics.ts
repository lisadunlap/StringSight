/**
 * TypeScript interfaces for metrics data.
 * 
 * These interfaces define the data contracts between the backend API
 * and the React frontend for metrics functionality.
 */

// Base interfaces matching the backend data structure

/**
 * Individual model-cluster row from the flattened JSONL format.
 */
export interface ModelClusterRow {
  model: string;
  cluster: string;
  size: number;
  proportion: number;
  proportion_delta: number;
  metadata?: Record<string, any>;
  examples?: any[];
  
  // Quality metrics (dynamic based on available data)
  // Pattern: quality_<metric_name>: number
  // Note: These will conflict with specific patterns below, but that's intentional for runtime flexibility
  quality?: Record<string, number>;
  quality_ci?: Record<string, { lower: number; upper: number; mean?: number }>;
  quality_delta?: Record<string, number>;
  quality_delta_ci?: Record<string, { lower: number; upper: number; mean?: number }>;
  quality_delta_significant?: Record<string, boolean>;
  
  // Confidence intervals (optional)
  proportion_ci_lower?: number;
  proportion_ci_upper?: number;
  proportion_ci_mean?: number;
  
  proportion_delta_ci_lower?: number;
  proportion_delta_ci_upper?: number;
  proportion_delta_ci_mean?: number;
  
  // Significance flags
  proportion_delta_significant?: boolean;
}

/**
 * Model benchmark row (per-model aggregates across all clusters).
 */
export interface ModelBenchmarkRow {
  model: string;
  cluster: "all_clusters";  // Always this value for benchmark data
  size: number;
  proportion: number;  // Always 1.0 for model aggregates
  examples?: any[];
  
  // Confidence intervals (optional)
  proportion_ci_lower?: number;
  proportion_ci_upper?: number; 
  proportion_ci_mean?: number;
}

// API response payloads

/**
 * Response from /metrics/model-cluster endpoint.
 */
export interface ModelClusterPayload {
  data: ModelClusterRow[];
  models: string[];
  clusters: string[];
  quality_metrics: string[];
  total_battles: number;
  source: "jsonl" | "json" | "computed" | "none";
}

/**
 * Response from /metrics/benchmark endpoint.
 */
export interface ModelBenchmarkPayload {
  data: ModelBenchmarkRow[];
  models: string[];
  quality_metrics: string[];
  source: "jsonl" | "json" | "computed" | "none";
}

/**
 * Response from /metrics/summary endpoint.
 */
export interface MetricsSummary {
  source: "jsonl" | "json" | "computed" | "none";
  models: number;
  clusters: number;
  total_battles: number;
  quality_metrics: number;
  quality_metric_names: string[];
  has_confidence_intervals: boolean;
  significant_differences: number;
}

/**
 * Response from /metrics/quality-metrics endpoint.
 */
export interface QualityMetricsResponse {
  quality_metrics: string[];
}

// Frontend-specific interfaces for component props and state

/**
 * Configuration for metrics filtering and sorting.
 */
export interface MetricsFilters {
  selectedModels: string[];
  selectedGroups: string[];  // Based on metadata.group values
  topN: number;
  sortBy: MetricsSortOption;
  significanceOnly: boolean;
  qualityMetric: string;
  showCI: boolean;
}

/**
 * Sorting options for metrics data.
 */
export type MetricsSortOption = 
  | "proportion_desc"
  | "proportion_asc" 
  | "proportion_delta_desc"
  | "proportion_delta_asc"
  | "quality_desc"
  | "quality_asc"
  | "quality_delta_desc"
  | "quality_delta_asc"
  | "size_desc"
  | "size_asc";

/**
 * Model card data for model cards section.
 */
export interface ModelCardData {
  model: string;
  totalBattles: number;
  clusters: ModelClusterRow[];
  topClusters: ModelClusterRow[];
}

/**
 * Plot type for metrics visualizations.
 */
export type MetricsPlotType = 
  | "frequency"        // Absolute proportion
  | "frequency_delta"  // Proportion delta (with zero line)
  | "quality"          // Absolute quality
  | "quality_delta";   // Quality delta (with zero line)

/**
 * Configuration for individual chart components.
 */
export interface ChartConfig {
  plotType: MetricsPlotType;
  qualityMetric?: string;
  showCI: boolean;
  topN: number;
  models: string[];
}

/**
 * Data for model cards display.
 */
export interface ModelCardData {
  model: string;
  totalBattles: number;
  clusters: ModelClusterRow[];
  topClusters: ModelClusterRow[];  // Pre-filtered top N clusters
}

/**
 * Significance badge types.
 */
export type SignificanceType = "frequency" | "quality";

/**
 * Significance badge props.
 */
export interface SignificanceBadgeProps {
  type: SignificanceType;
  significant: boolean;
  className?: string;
}

/**
 * Tag chip data for cluster metadata.
 */
export interface TagChipData {
  label: string;
  color?: "primary" | "secondary" | "error" | "warning" | "info" | "success";
  size?: "small" | "medium";
}

// Utility types for working with dynamic metric columns

/**
 * Helper type to extract quality metric names from column names.
 */
export type ExtractQualityMetric<T extends string> = 
  T extends `quality_${infer Metric}` 
    ? Metric extends `${infer M}_ci_${string}` 
      ? M 
      : Metric extends `${infer M}_significant`
      ? never  // Skip significance columns
      : Metric
    : never;

/**
 * Helper type to extract quality delta metric names.
 */
export type ExtractQualityDeltaMetric<T extends string> = 
  T extends `quality_delta_${infer Metric}` 
    ? Metric extends `${infer M}_ci_${string}` 
      ? M 
      : Metric extends `${infer M}_significant`
      ? never  // Skip significance columns
      : Metric
    : never;

// API client types for React Query

/**
 * Parameters for model-cluster metrics API call.
 */
export interface ModelClusterParams {
  resultsDir: string;
  models?: string[];
  qualityMetric?: string;
  significantOnly?: boolean;
}

/**
 * Parameters for benchmark metrics API call.
 */
export interface BenchmarkParams {
  resultsDir: string;
  models?: string[];
}

/**
 * Error response from API.
 */
export interface ApiError {
  detail: string;
}

// Re-export for convenience
export type { ModelClusterRow as ModelClusterData };
export type { ModelBenchmarkRow as ModelBenchmarkData };