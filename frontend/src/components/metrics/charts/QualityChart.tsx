/**
 * QualityChart - Absolute quality scores by cluster.
 * 
 * Shows grouped bars by model displaying the raw quality metric scores
 * for each model within each cluster.
 */

import React, { useMemo } from 'react';
import { Box, Typography, Alert } from '@mui/material';
import { PlotlyChartBase, getModelColor, truncateLabel, createHoverTemplate } from './PlotlyChartBase';
import { getDisplayName, getOriginalMetricName, sanitizeMetricName } from '../utils/metricUtils';
import type { ModelClusterRow, MetricsFilters } from '../../../types/metrics';

interface QualityChartProps {
  /** Model-cluster data */
  data: ModelClusterRow[];
  /** Current filters */
  filters: MetricsFilters;
  /** Pre-computed top clusters (in order) */
  topClusters?: string[];
  /** Whether to show confidence intervals */
  showCI?: boolean;
  /** Chart height */
  height?: number;
}

export function QualityChart({
  data,
  filters,
  topClusters,
  showCI = false,
  height = 400
}: QualityChartProps) {
  
  // Check for early returns first
  if (!data.length || !filters.qualityMetric) {
    return (
      <Box sx={{ height }}>
        <Alert severity="info">
          No cluster data available for quality analysis.
        </Alert>
      </Box>
    );
  }

  // Filter data by selected models
  const filteredData = data.filter(row => 
    filters.selectedModels.length === 0 || filters.selectedModels.includes(row.model)
  );

  if (!filteredData.length) {
    return (
      <Box sx={{ height }}>
        <Alert severity="info">
          No data available for selected models.
        </Alert>
      </Box>
    );
  }

  // Prepare both name variants: original (for nested object) and sanitized (for flattened columns)
  const originalMetric = getOriginalMetricName(filters.qualityMetric);
  const sanitizedMetric = sanitizeMetricName(filters.qualityMetric);
  
  // Debug logging
  if (import.meta.env.DEV) {
    const sampleRow = filteredData[0];
    console.log('QualityChart Debug:', {
      qualityMetric: filters.qualityMetric,
      originalMetric,
      hasQualityObject: sampleRow && 'quality' in sampleRow,
      qualityKeys: sampleRow?.quality ? Object.keys(sampleRow.quality) : [],
      metricInQuality: sampleRow?.quality && originalMetric in sampleRow.quality
    });
  }

  // Accept either nested quality object or flattened quality_<metric> columns
  const dataWithQuality = filteredData.filter(row => {
    // Nested
    const nestedVal = (row as any)?.quality?.[originalMetric];
    if (typeof nestedVal === 'number' && isFinite(nestedVal)) return true;
    // Flattened (prefer exact filter key, then sanitized)
    const flat1 = (row as any)[`quality_${filters.qualityMetric}`];
    const flat2 = (row as any)[`quality_${sanitizedMetric}`];
    return [flat1, flat2].some(v => typeof v === 'number' && isFinite(v as number));
  });

  if (!dataWithQuality.length) {
    return (
      <Box sx={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Alert severity="warning" sx={{ width: '100%' }}>
          <Typography variant="body2">
            No data available for quality metric "{getDisplayName(filters.qualityMetric)}".
            <br />
            Try selecting a different metric or adjusting filters.
          </Typography>
        </Alert>
      </Box>
    );
  }

  const plotData = useMemo(() => {

    // Group by cluster (data is already sorted and filtered by MetricsMainContent)
    const clusterGroups = dataWithQuality.reduce((groups, row) => {
      if (!groups[row.cluster]) {
        groups[row.cluster] = [];
      }
      groups[row.cluster].push(row);
      return groups;
    }, {} as Record<string, ModelClusterRow[]>);

    // Use pre-computed top clusters, or fall back to extracting from data
    const clustersToShow = topClusters || (() => {
      const seenClusters = new Set<string>();
      const localTopClusters: string[] = [];
      for (const row of dataWithQuality) {
        if (!seenClusters.has(row.cluster) && localTopClusters.length < filters.topN) {
          seenClusters.add(row.cluster);
          localTopClusters.push(row.cluster);
        }
      }
      return localTopClusters;
    })();
    
    // Get all models that should appear (from original filters, not just filtered data)
    // This ensures we show all selected models even if they have zero values
    const allModels = filters.selectedModels.length > 0 
      ? filters.selectedModels 
      : [...new Set(data.map(row => row.model))].sort();
    
    // Create grouped bar chart data (one trace per model)
    return allModels.map(model => {
      const modelData = clustersToShow.map(cluster => {
        const row = clusterGroups[cluster]?.find(r => r.model === model);
        const nestedVal = row?.quality?.[originalMetric];
        const flat1 = (row as any)?.[`quality_${filters.qualityMetric}`];
        const flat2 = (row as any)?.[`quality_${sanitizedMetric}`];
        const qualityValue = typeof nestedVal === 'number' && isFinite(nestedVal)
          ? nestedVal
          : (typeof flat1 === 'number' && isFinite(flat1) ? flat1 : (typeof flat2 === 'number' && isFinite(flat2) ? flat2 : 0));
        
        return {
          cluster,
          quality: qualityValue,
          // CI support for both nested and flattened format
          ciLower: showCI ? (
            row?.quality_ci?.[originalMetric]?.lower ||
            (row as any)?.[`quality_${filters.qualityMetric}_ci_lower`] ||
            (row as any)?.[`quality_${sanitizedMetric}_ci_lower`]
          ) : undefined,
          ciUpper: showCI ? (
            row?.quality_ci?.[originalMetric]?.upper ||
            (row as any)?.[`quality_${filters.qualityMetric}_ci_upper`] ||
            (row as any)?.[`quality_${sanitizedMetric}_ci_upper`]
          ) : undefined,
          size: row?.size || 0
        };
      });

      const qualities = modelData.map(d => d.quality);
      const clusterLabels = clustersToShow.map(cluster => truncateLabel(cluster, 20));

      const trace: any = {
        type: 'bar',
        name: model.split('/').pop() || model, // Short model name for legend
        x: clusterLabels,
        y: qualities,
        marker: {
          color: getModelColor(model, allModels),
          opacity: 0.8
        },
        // Values shown in hover only
        hovertemplate: clustersToShow.map((cluster, i) => 
          createHoverTemplate(
            cluster, 
            qualities[i], 
            getDisplayName(filters.qualityMetric), 
            3
          ) + 
          `<br>Conversations: ${modelData[i].size}`
        )
      };

      // Add confidence intervals if available
      if (showCI && modelData.some(d => d.ciLower !== undefined && d.ciUpper !== undefined)) {
        const arrayminus = modelData.map(d => 
          (d.ciLower !== undefined && d.quality !== undefined) ? 
            Math.max(0, d.quality - d.ciLower) : 0
        );
        const arrayplus = modelData.map(d => 
          (d.ciUpper !== undefined && d.quality !== undefined) ? 
            Math.max(0, d.ciUpper - d.quality) : 0
        );
        
        trace.error_y = {
          type: 'data',
          symmetric: false,
          array: arrayplus,
          arrayminus: arrayminus,
          visible: true,
          thickness: 2,
          width: 5,
          color: getModelColor(model, allModels)
        };
      }

      return trace;
    });
  }, [data, filters, showCI]);

  if (!data.length) {
    return (
      <Box sx={{ height }}>
        <Alert severity="info">
          No cluster data available for quality analysis.
        </Alert>
      </Box>
    );
  }

  if (!filters.qualityMetric) {
    return (
      <Box sx={{ height }}>
        <Alert severity="warning">
          Please select a quality metric to display quality scores.
        </Alert>
      </Box>
    );
  }

  if (!plotData.length) {
    return (
      <Box sx={{ height }}>
        <Alert severity="warning">
          No data available for quality metric "{getDisplayName(filters.qualityMetric)}". 
          Try selecting a different metric or adjusting filters.
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
        <Typography variant="h6" component="h3">
          Quality by Cluster
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {getDisplayName(filters.qualityMetric)} scores per model in each cluster
          {showCI && ' (with confidence intervals)'}
        </Typography>
      </Box>
      
      <PlotlyChartBase
        data={plotData}
        height={height}
        yAxisLabel={`${getDisplayName(filters.qualityMetric)} Score`}
        layout={{
          barmode: 'group',
          bargap: 0.2,
          bargroupgap: 0.1,
          yaxis: {
            tickformat: '.2f'
          },
          xaxis: {
            tickangle: -45
          }
        }}
      />
    </Box>
  );
}

export default QualityChart;
