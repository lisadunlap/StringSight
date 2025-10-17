/**
 * useMetricsData - React Query hook for loading metrics data.
 * 
 * This hook manages the loading of metrics data from the backend API
 * with automatic fallback from JSONL → JSON → computed data.
 * 
 * Features:
 * - React Query caching and background refetching
 * - Error handling with retry logic
 * - Automatic data transformation
 * - Loading states and error reporting
 */

import { useQuery } from '@tanstack/react-query';
import type {
  ModelClusterPayload,
  ModelBenchmarkPayload,
  MetricsSummary,
  QualityMetricsResponse
} from '../../types/metrics';

// API base URL - adjust based on your setup
const API_BASE_URL = 'http://localhost:8000';

/**
 * API client functions
 */
const apiClient = {
  async fetchSummary(resultsDir: string): Promise<MetricsSummary> {
    const response = await fetch(`${API_BASE_URL}/metrics/summary/${resultsDir}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch summary: ${response.statusText}`);
    }
    return response.json();
  },

  async fetchModelCluster(resultsDir: string): Promise<ModelClusterPayload> {
    const response = await fetch(`${API_BASE_URL}/metrics/model-cluster/${resultsDir}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch model-cluster data: ${response.statusText}`);
    }
    return response.json();
  },

  async fetchBenchmark(resultsDir: string): Promise<ModelBenchmarkPayload> {
    const response = await fetch(`${API_BASE_URL}/metrics/benchmark/${resultsDir}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch benchmark data: ${response.statusText}`);
    }
    return response.json();
  },

  async fetchQualityMetrics(resultsDir: string): Promise<QualityMetricsResponse> {
    const response = await fetch(`${API_BASE_URL}/metrics/quality-metrics/${resultsDir}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch quality metrics: ${response.statusText}`);
    }
    return response.json();
  }
};

/**
 * Hook for loading metrics data with React Query
 */
export function useMetricsData(resultsDir: string) {
  // Demo mode flag (can be configured via env variable)
  const useDemo = false;

  // Summary data
  const {
    data: summary,
    isLoading: summaryLoading,
    error: summaryError
  } = useQuery({
    queryKey: ['metrics', 'summary', resultsDir],
    queryFn: () => apiClient.fetchSummary(resultsDir),
    retry: 2,
    staleTime: 5 * 60 * 1000, // 5 minutes
    enabled: !!resultsDir && !useDemo
  });

  // Model-cluster data
  const {
    data: modelClusterData,
    isLoading: modelClusterLoading,
    error: modelClusterError
  } = useQuery({
    queryKey: ['metrics', 'model-cluster', resultsDir],
    queryFn: () => apiClient.fetchModelCluster(resultsDir),
    retry: 2,
    staleTime: 5 * 60 * 1000,
    enabled: !!resultsDir && !useDemo
  });

  // Benchmark data
  const {
    data: benchmarkData,
    isLoading: benchmarkLoading,
    error: benchmarkError
  } = useQuery({
    queryKey: ['metrics', 'benchmark', resultsDir],
    queryFn: () => apiClient.fetchBenchmark(resultsDir),
    retry: 2,
    staleTime: 5 * 60 * 1000,
    enabled: !!resultsDir && !useDemo
  });

  // Quality metrics
  const {
    data: qualityMetricsResponse,
    isLoading: qualityMetricsLoading,
    error: qualityMetricsError
  } = useQuery({
    queryKey: ['metrics', 'quality-metrics', resultsDir],
    queryFn: () => apiClient.fetchQualityMetrics(resultsDir),
    retry: 2,
    staleTime: 5 * 60 * 1000,
    enabled: !!resultsDir && !useDemo
  });

  // Derived data
  const qualityMetrics = qualityMetricsResponse?.quality_metrics || [];

  // Combined loading state
  const isLoading = summaryLoading || modelClusterLoading || benchmarkLoading || qualityMetricsLoading;

  // Combined error state
  const error = summaryError || modelClusterError || benchmarkError || qualityMetricsError;

  // Refetch all data
  const refetch = async () => {
    // Note: Individual query refetch would be more granular
    window.location.reload(); // Simple approach for now
  };

  return {
    summary,
    modelClusterData,
    benchmarkData,
    qualityMetrics,
    isLoading,
    error,
    refetch
  };
}

/**
 * Hook for loading filtered metrics data
 */
export function useFilteredMetricsData(
  resultsDir: string,
  options: {
    models?: string[];
    qualityMetric?: string;
    significantOnly?: boolean;
  } = {}
) {
  const { models, qualityMetric, significantOnly } = options;

  return useQuery({
    queryKey: ['metrics', 'model-cluster', resultsDir, { models, qualityMetric, significantOnly }],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (models && models.length > 0) {
        models.forEach(model => params.append('models', model));
      }
      if (qualityMetric) {
        params.set('quality_metric', qualityMetric);
      }
      if (significantOnly) {
        params.set('significant_only', 'true');
      }

      const url = `${API_BASE_URL}/metrics/model-cluster/${resultsDir}?${params.toString()}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch filtered data: ${response.statusText}`);
      }
      
      return response.json() as Promise<ModelClusterPayload>;
    },
    retry: 2,
    staleTime: 2 * 60 * 1000, // 2 minutes (shorter for filtered data)
    enabled: !!resultsDir
  });
}

export default useMetricsData;