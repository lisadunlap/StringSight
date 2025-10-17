/**
 * MetricsTab - Main container for the metrics visualization interface.
 * 
 * This component displays model performance metrics with filtering controlled
 * by the sidebar. It provides a comprehensive view of benchmarks, cluster plots,
 * and model cards based on the filters passed from the parent component.
 * 
 * Layout:
 * ┌─────────────────────────────────────────────────────────┐
 * │ Main Content Area (Full Width)                          │
 * │ - Benchmark Section                                      │
 * │ - Cluster Plots (2 types)                               │
 * │ - Model Cards (2-column)                                │
 * └─────────────────────────────────────────────────────────┘
 * 
 * Filter controls are in the sidebar (MetricsPanel component).
 */

import { useEffect, useMemo } from 'react';
import { 
  Box, 
  Typography, 
  Alert, 
  CircularProgress, 
  Fade 
} from '@mui/material';
import { MetricsMainContent } from './MetricsMainContent';
import type { MetricsFilters, MetricsSummary, ModelClusterPayload, ModelBenchmarkPayload } from '../../types/metrics';

interface MetricsTabProps {
  /** Pre-loaded results data */
  resultsData: {
    model_cluster_scores?: any;
    cluster_scores?: any; 
    model_scores?: any;
  };
  
  /** Filters controlled by the sidebar */
  filters: MetricsFilters;
  
  /** Callback to update available data for sidebar */
  onDataProcessed?: (data: {
    availableModels: string[];
    availableGroups: string[];
    availableQualityMetrics: string[];
    summary: MetricsSummary | null;
  }) => void;
  
  /** Whether to show debug information */
  debug?: boolean;

  /** Section visibility controls (default: all true) */
  showBenchmark?: boolean;
  showClusterPlots?: boolean;
  showModelCards?: boolean;
}

export function MetricsTab({ 
  resultsData, 
  filters,
  onDataProcessed,
  debug = false,
  showBenchmark = true,
  showClusterPlots = true,
  showModelCards = true
}: MetricsTabProps) {

  // Process the existing resultsData instead of fetching from API
  const processedData = useMemo(() => {
    if (!resultsData?.model_cluster_scores) {
      return {
        summary: null as MetricsSummary | null,
        modelClusterData: null as ModelClusterPayload | null,
        benchmarkData: null as ModelBenchmarkPayload | null,
        qualityMetrics: [] as string[],
        availableGroups: [] as string[],
        isLoading: false,
        error: new Error("No model cluster scores data available"),
        refetch: () => Promise.resolve()
      };
    }

    // Server now always returns JSONL format (array of objects)
    const modelClusterScores = resultsData.model_cluster_scores || [];
    
    // Extract models, clusters, and groups
    const models = [...new Set(modelClusterScores.map((row: any) => row.model))].sort();
    const clusters = [...new Set(modelClusterScores.map((row: any) => row.cluster))];
    
    // Extract groups from metadata
    const groups = new Set<string>();
    modelClusterScores.forEach((row: any) => {
      if (row.metadata && typeof row.metadata === 'object' && row.metadata.group) {
        groups.add(row.metadata.group);
      }
    });
    
    // Extract quality metrics from JSONL format (e.g., "quality_omni_math_accuracy_0_1")
    const qualityMetrics = new Set<string>();
    modelClusterScores.forEach((row: any) => {
      Object.keys(row).forEach(key => {
        if (key.startsWith('quality_') && !key.endsWith('_delta') && !key.endsWith('_significant') && !key.includes('_ci_')) {
          const metric = key.replace('quality_', '');
          qualityMetrics.add(metric);
        }
      });
    });

    // Calculate actual battle count (unique conversations) from examples
    const uniqueConversations = new Set<string>();
    modelClusterScores.forEach((row: any) => {
      if (row.examples && Array.isArray(row.examples)) {
        row.examples.forEach((example: any) => {
          if (example && Array.isArray(example) && example[0]) {
            uniqueConversations.add(String(example[0])); // conversation_id is first element
          }
        });
      }
    });

    // Detect confidence intervals from data
    const hasConfidenceIntervals = modelClusterScores.some((row: any) => {
      return row.proportion_ci_lower !== undefined || 
             row.proportion_ci_upper !== undefined ||
             Object.keys(row).some(key => key.includes('_ci_lower') || key.includes('_ci_upper'));
    });

    // Debug logging for metric extraction
    if (import.meta.env.DEV) {
      console.log('MetricsTab Debug - Extracted quality metrics:', Array.from(qualityMetrics));
      console.log('MetricsTab Debug - Extracted groups:', Array.from(groups));
      console.log('MetricsTab Debug - Has confidence intervals:', hasConfidenceIntervals);
      console.log('MetricsTab Debug - Sample row keys:', modelClusterScores[0] ? Object.keys(modelClusterScores[0]).filter(k => k.startsWith('quality_')) : []);
      console.log('MetricsTab Debug - CI keys:', modelClusterScores[0] ? Object.keys(modelClusterScores[0]).filter(k => k.includes('_ci_')) : []);
      console.log('MetricsTab Debug - Full sample row:', modelClusterScores[0]);
      console.log('MetricsTab Debug - Sample row.quality:', modelClusterScores[0]?.quality);
    }

    // Create summary
    const summary: MetricsSummary = {
      source: 'json' as const,
      models: models.length,
      clusters: clusters.length,
      total_battles: uniqueConversations.size, // Use actual conversation count, not property count
      quality_metrics: qualityMetrics.size,
      quality_metric_names: Array.from(qualityMetrics),
      has_confidence_intervals: hasConfidenceIntervals,
      significant_differences: 0 // TODO: compute from data
    };

    // Transform model cluster data for frontend
    const modelClusterData: ModelClusterPayload = {
      source: 'json' as const,
      models: models as string[],
      clusters: clusters as string[],
      quality_metrics: Array.from(qualityMetrics),
      total_battles: modelClusterScores.length,
      data: modelClusterScores
    };

    // Process benchmark data if available
    const benchmarkData: ModelBenchmarkPayload | undefined = resultsData.model_scores ? {
      source: 'json' as const,
      models: models as string[],
      quality_metrics: Array.from(qualityMetrics),
      data: Array.isArray(resultsData.model_scores) ? resultsData.model_scores : []
    } : undefined;

    return {
      summary,
      modelClusterData,
      benchmarkData,
      qualityMetrics: Array.from(qualityMetrics),
      availableGroups: Array.from(groups),
      isLoading: false,
      error: null,
      refetch: () => Promise.resolve()
    };
  }, [resultsData]);

  const {
    summary,
    modelClusterData, 
    benchmarkData,
    qualityMetrics,
    availableGroups,
    isLoading,
    error,
    refetch
  } = processedData;

  // Notify parent when data is processed (for sidebar to update available options)
  useEffect(() => {
    if (onDataProcessed && modelClusterData) {
      onDataProcessed({
        availableModels: modelClusterData.models,
        availableGroups: availableGroups,
        availableQualityMetrics: qualityMetrics,
        summary: summary
      });
    }
  }, [onDataProcessed, modelClusterData, availableGroups, qualityMetrics, summary]);

  // Loading state
  if (isLoading) {
    return (
      <Box 
        sx={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '60vh',
          flexDirection: 'column',
          gap: 2
        }}
      >
        <CircularProgress size={48} />
        <Typography variant="body1" color="text.secondary">
          Loading metrics data...
        </Typography>
        {debug && (
          <Typography variant="caption" color="text.disabled">
            Data source: {summary?.source || 'loaded_results'}
          </Typography>
        )}
      </Box>
    );
  }

  // Error state
  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert 
          severity="error" 
          sx={{ mb: 2 }}
          action={
            <button onClick={() => refetch()}>
              Retry
            </button>
          }
        >
          <strong>Failed to load metrics data</strong>
          <br />
          {error.message}
          {debug && (
            <>
              <br />
              <Typography variant="caption" component="div" sx={{ mt: 1 }}>
                Data source: {(summary as any)?.source || 'unavailable'}
              </Typography>
            </>
          )}
        </Alert>
      </Box>
    );
  }

  // No data state
  if (!modelClusterData || modelClusterData.data.length === 0) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="info">
          <strong>No metrics data available</strong>
          <br />
          No model-cluster metrics found in the specified results directory. 
          Make sure the metrics have been computed and saved.
          {debug && (
            <>
              <br />
              <Typography variant="caption" component="div" sx={{ mt: 1 }}>
                Data source: {(summary as any)?.source || 'unavailable'}
              </Typography>
            </>
          )}
        </Alert>
      </Box>
    );
  }

  

  return (
    <Fade in={true} timeout={300}>
      <Box sx={{ height: 'calc(100vh - 120px)', overflow: 'hidden' }}>
        {/* Main Content Area - Full Width */}
        <MetricsMainContent
          filters={filters}
          modelClusterData={modelClusterData}
          benchmarkData={benchmarkData}
          qualityMetrics={qualityMetrics || []}
          summary={summary}
          showBenchmark={showBenchmark}
          showClusterPlots={showClusterPlots}
          showModelCards={showModelCards}
        />
      </Box>
    </Fade>
  );
}

export default MetricsTab;