/**
 * FrequencyChart - Absolute frequency (proportion) by cluster.
 * 
 * Shows grouped bars by model displaying what fraction of each model's
 * conversations fall into each cluster (0-1 scale).
 */

import React, { useMemo } from 'react';
import { Box, Typography, Alert } from '@mui/material';
import { PlotlyChartBase, getModelColor, truncateLabel, createHoverTemplate } from './PlotlyChartBase';
import type { ModelClusterRow, MetricsFilters } from '../../../types/metrics';

interface FrequencyChartProps {
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

export function FrequencyChart({
  data,
  filters,
  topClusters,
  showCI = false,
  height = 400
}: FrequencyChartProps) {
  
  const plotData = useMemo(() => {
    if (!data.length) {
      return [];
    }

    // Filter data by selected models
    const filteredData = data.filter(row => 
      filters.selectedModels.length === 0 || filters.selectedModels.includes(row.model)
    );

    if (!filteredData.length) {
      return [];
    }

    // Group by cluster for easy lookup
    const clusterGroups = filteredData.reduce((groups, row) => {
      if (!groups[row.cluster]) {
        groups[row.cluster] = [];
      }
      groups[row.cluster].push(row);
      return groups;
    }, {} as Record<string, ModelClusterRow[]>);

    // Use pre-computed top clusters, or fall back to extracting from data
    const clustersToShow = topClusters || [
      ...new Set(filteredData.map(row => row.cluster))
    ].slice(0, filters.topN);
    
    // Get all models that should appear (from original filters, not just filtered data)
    // This ensures we show all selected models even if they have zero values
    const allModels = filters.selectedModels.length > 0 
      ? filters.selectedModels 
      : [...new Set(data.map(row => row.model))].sort();
    
    // Create grouped bar chart data (one trace per model)
    return allModels.map(model => {
      const modelData = clustersToShow.map(cluster => {
        const row = clusterGroups[cluster]?.find(r => r.model === model);
        return {
          cluster,
          proportion: row?.proportion || 0,  // Use 0 for missing model-cluster combinations
          ciLower: showCI ? row?.proportion_ci_lower : undefined,
          ciUpper: showCI ? row?.proportion_ci_upper : undefined,
          size: row?.size || 0
        };
      });

      const proportions = modelData.map(d => d.proportion);
      const clusterLabels = clustersToShow.map(cluster => truncateLabel(cluster, 20));


      const trace: any = {
        type: 'bar',
        name: model.split('/').pop() || model, // Short model name for legend
        x: clusterLabels,
        y: proportions,
        marker: {
          color: getModelColor(model, allModels),
          opacity: 0.8
        },
        // Values shown in hover only
        hovertemplate: clustersToShow.map((cluster, i) => 
          createHoverTemplate(
            cluster, 
            proportions[i] * 100, 
            'Frequency (%)', 
            1
          ) + 
          `<br>Conversations: ${modelData[i].size}`
        )
      };

      // Add confidence intervals if available
      if (showCI && modelData.some(d => d.ciLower !== undefined && d.ciUpper !== undefined)) {
        // For bar charts, error bars should extend from the bar top
        // arrayminus: how far DOWN from the bar value (proportion)
        // arrayplus: how far UP from the bar value (proportion)
        const arrayminus = modelData.map(d => 
          (d.ciLower !== undefined && d.proportion !== undefined) ? 
            Math.max(0, d.proportion - d.ciLower) : 0
        );
        const arrayplus = modelData.map(d => 
          (d.ciUpper !== undefined && d.proportion !== undefined) ? 
            Math.max(0, d.ciUpper - d.proportion) : 0
        );
        
        trace.error_y = {
          type: 'data',
          symmetric: false,
          array: arrayplus,      // Upper error (from bar top)
          arrayminus: arrayminus, // Lower error (from bar top)
          visible: true,
          thickness: 2,
          width: 5,
          color: getModelColor(model, allModels) // Match bar color
        };
      }

      return trace;
    });
  }, [data, filters, showCI]);

  if (!data.length) {
    return (
      <Box sx={{ height }}>
        <Alert severity="info">
          No cluster data available for frequency analysis.
        </Alert>
      </Box>
    );
  }

  if (!plotData.length) {
    return (
      <Box sx={{ height }}>
        <Alert severity="warning">
          No data matches the current filters. Try adjusting your model selection.
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
        <Typography variant="h6" component="h3">
          Frequency by Cluster
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Proportion of conversations per model in each cluster
          {showCI && ' (with confidence intervals)'}
        </Typography>
      </Box>
      
      <PlotlyChartBase
        data={plotData}
        height={height}
        yAxisLabel="Frequency (Proportion)"
        layout={{
          barmode: 'group',
          bargap: 0.2,
          bargroupgap: 0.1,
          yaxis: {
            tickformat: '.1%',
            range: [0, Math.max(1, Math.max(...plotData.flatMap(trace => trace.y)) * 1.1)]
          },
          xaxis: {
            tickangle: -45
          }
        }}
      />
    </Box>
  );
}

export default FrequencyChart;
