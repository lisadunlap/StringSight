/**
 * MetricsMainContent - Main content area displaying metrics visualizations.
 * 
 * This component renders the main metrics content in three sections:
 * 1. Benchmark Section - Per-model performance bar chart
 * 2. Cluster Plots - Four chart types (frequency, frequency_delta, quality, quality_delta)
 * 3. Model Cards - Two-column grid with top clusters per model
 * 
 * The layout follows the specifications in METRICS_README.md.
 */

import React, { useMemo } from 'react';
import {
  Box,
  Stack,
  Typography,
  Divider,
  Alert
} from '@mui/material';
import { BenchmarkSection } from './BenchmarkSection';
import { ClusterPlotsSection } from './ClusterPlotsSection';
import { ModelCardsSection } from './ModelCardsSection';
import type {
  MetricsFilters,
  ModelClusterPayload,
  ModelBenchmarkPayload,
  MetricsSummary
} from '../../types/metrics';

interface MetricsMainContentProps {
  filters: MetricsFilters;
  modelClusterData: ModelClusterPayload;
  benchmarkData?: ModelBenchmarkPayload;
  qualityMetrics: string[];
  summary?: MetricsSummary;
  /** Controls visibility of sections; default true for all when omitted */
  showBenchmark?: boolean;
  showClusterPlots?: boolean;
  showModelCards?: boolean;
}

export function MetricsMainContent({
  filters,
  modelClusterData,
  benchmarkData,
  qualityMetrics,
  summary,
  showBenchmark = true,
  showClusterPlots = true,
  showModelCards = true
}: MetricsMainContentProps) {

  // Apply filters to the data
  const { filteredData, topClusters } = useMemo(() => {
    let filtered = [...modelClusterData.data];
    
    // Filter by selected models
    if (filters.selectedModels.length > 0) {
      filtered = filtered.filter(row => 
        filters.selectedModels.includes(row.model)
      );
    }
    
    // Filter by selected groups
    if (filters.selectedGroups.length > 0) {
      filtered = filtered.filter(row => {
        const group = row.metadata?.group;
        return group && filters.selectedGroups.includes(group);
      });
    }
    
    // Filter by significance (if enabled)
    if (filters.significanceOnly) {
      filtered = filtered.filter(row => {
        // Check proportion significance
        if (row.proportion_delta_significant) return true;
        
        // Check quality significance for current metric
        const qualitySigKey = `quality_delta_${filters.qualityMetric}_significant`;
        if (row[qualitySigKey as keyof typeof row]) return true;
        
        return false;
      });
    }
    
    // STEP 1: Find top N clusters globally (before applying topN row limit)
    // Group by cluster and compute ranking metric
    const clusterStats = filtered.reduce((acc, row) => {
      if (!acc[row.cluster]) {
        acc[row.cluster] = { 
          cluster: row.cluster, 
          maxProportion: 0, 
          maxSize: 0,
          totalSize: 0,
          models: new Set()
        };
      }
      const stats = acc[row.cluster];
      stats.maxProportion = Math.max(stats.maxProportion, row.proportion || 0);
      stats.maxSize = Math.max(stats.maxSize, row.size || 0);
      stats.totalSize += row.size || 0;
      stats.models.add(row.model);
      return acc;
    }, {} as Record<string, any>);
    
    // Sort clusters by the same metric as rows, then take topN
    const sortedClusters = Object.values(clusterStats).sort((a, b) => {
      let aVal: number, bVal: number;
      
      // Use same sorting logic but applied to cluster-level stats
      switch (filters.sortBy) {
        case 'proportion_desc':
        case 'proportion_asc':
          aVal = a.maxProportion;
          bVal = b.maxProportion;
          break;
        case 'size_desc':
        case 'size_asc':
          aVal = a.totalSize;
          bVal = b.totalSize;
          break;
        default:
          // For other metrics, fall back to max proportion
          aVal = a.maxProportion;
          bVal = b.maxProportion;
          break;
      }
      
      const ascending = filters.sortBy.includes('_asc');
      return ascending ? aVal - bVal : bVal - aVal;
    });
    
    const topClusterNames = sortedClusters.slice(0, filters.topN).map(c => c.cluster);
    
    // STEP 2: Filter data to only include top clusters
    const clusterFiltered = filtered.filter(row => 
      topClusterNames.includes(row.cluster)
    );
    
    // STEP 3: Sort rows within the filtered clusters
    clusterFiltered.sort((a, b) => {
      const getSortValue = (row: typeof a, sortBy: string): number => {
        switch (sortBy) {
          case 'proportion_desc':
          case 'proportion_asc':
            return row.proportion || 0;
          case 'proportion_delta_desc':
          case 'proportion_delta_asc':
            return row.proportion_delta || 0;
          case 'quality_desc':
          case 'quality_asc':
            const qualityKey = `quality_${filters.qualityMetric}`;
            return (row[qualityKey as keyof typeof row] as number) || 0;
          case 'quality_delta_desc':
          case 'quality_delta_asc':
            const qualityDeltaKey = `quality_delta_${filters.qualityMetric}`;
            return (row[qualityDeltaKey as keyof typeof row] as number) || 0;
          case 'size_desc':
          case 'size_asc':
            return row.size || 0;
          default:
            return 0;
        }
      };
      
      const aVal = getSortValue(a, filters.sortBy);
      const bVal = getSortValue(b, filters.sortBy);
      
      const ascending = filters.sortBy.includes('_asc');
      return ascending ? aVal - bVal : bVal - aVal;
    });
    
    return { 
      filteredData: clusterFiltered, 
      topClusters: topClusterNames 
    };
  }, [modelClusterData.data, filters]);

  // No data after filtering
  if (filteredData.length === 0) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="info">
          <strong>No data matches the current filters</strong>
          <br />
          Try adjusting your filter settings to see results.
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ 
      height: '100%', 
      width: '100%',
      overflow: 'auto', 
      p: 3,
      '& > *': { mb: 4 }
    }}>
      {/* Model Cards Section */}
      {showModelCards && (
        <>
          <ModelCardsSection
            data={filteredData}
            filters={filters}
            qualityMetrics={qualityMetrics}
            totalBattles={modelClusterData.total_battles}
          />
          <Divider />
        </>
      )}

      {/* Benchmark Section */}
      {showBenchmark && benchmarkData && benchmarkData.data.length > 0 && (
        <>
          <BenchmarkSection
            data={benchmarkData}
            qualityMetrics={qualityMetrics}
          />
          <Divider />
        </>
      )}

      {/* Cluster Plots Section */}
      {showClusterPlots && (
        <ClusterPlotsSection
          data={filteredData}
          filters={filters}
          qualityMetrics={qualityMetrics}
          showCI={filters.showCI && (summary?.has_confidence_intervals || false)}
          topClusters={topClusters}
        />
      )}
    </Box>
  );
}

export default MetricsMainContent;