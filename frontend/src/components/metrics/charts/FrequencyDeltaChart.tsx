/**
 * FrequencyDeltaChart - Frequency delta (salience) by cluster.
 * 
 * Shows grouped bars by model displaying how much each model over/under-represents
 * in each cluster compared to the cross-model average. Includes zero line.
 */

import React, { useMemo } from 'react';
import { Box, Typography, Alert } from '@mui/material';
import { PlotlyChartBase, getModelColor, truncateLabel, createHoverTemplate } from './PlotlyChartBase';
import type { ModelClusterRow, MetricsFilters } from '../../../types/metrics';

interface FrequencyDeltaChartProps {
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

export function FrequencyDeltaChart({
  data,
  filters,
  topClusters,
  showCI = false,
  height = 400
}: FrequencyDeltaChartProps) {
  
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

    // Apply significance filter if enabled
    let finalData = filteredData;
    if (filters.significanceOnly) {
      finalData = filteredData.filter(row => row.proportion_delta_significant === true);
    }

    if (!finalData.length && filters.significanceOnly) {
      return []; // Will show "no significant differences" message
    }

    // Group by cluster (data is already sorted and filtered by MetricsMainContent)
    const clusterGroups = finalData.reduce((groups, row) => {
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
      for (const row of finalData) {
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
        return {
          cluster,
          proportionDelta: row?.proportion_delta || 0,
          ciLower: showCI ? row?.proportion_delta_ci_lower : undefined,
          ciUpper: showCI ? row?.proportion_delta_ci_upper : undefined,
          size: row?.size || 0,
          significant: row?.proportion_delta_significant === true
        };
      });

      const deltas = modelData.map(d => d.proportionDelta);
      const clusterLabels = clustersToShow.map(cluster => truncateLabel(cluster, 20));

      const trace: any = {
        type: 'bar',
        name: model.split('/').pop() || model, // Short model name for legend
        x: clusterLabels,
        y: deltas,
        marker: {
          color: getModelColor(model, allModels),
          opacity: 0.8
        },
        // Values shown in hover only
        hovertemplate: clustersToShow.map((cluster, i) => {
          const delta = deltas[i];
          const significant = modelData[i].significant;
          return createHoverTemplate(
            cluster, 
            delta * 100, 
            'Frequency Δ (%)', 
            1
          ) + 
          `<br>Conversations: ${modelData[i].size}` +
          (significant ? '<br><b>Significant</b>' : '');
        })
      };

      // Add confidence intervals if available
      if (showCI && modelData.some(d => d.ciLower !== undefined && d.ciUpper !== undefined)) {
        // For delta charts, error bars extend from the delta value (which can be negative)
        // Don't use Math.max(0, ...) since delta values and CIs can be negative
        const arrayminus = modelData.map(d => 
          (d.ciLower !== undefined && d.proportionDelta !== undefined) ? 
            Math.abs(d.proportionDelta - d.ciLower) : 0
        );
        const arrayplus = modelData.map(d => 
          (d.ciUpper !== undefined && d.proportionDelta !== undefined) ? 
            Math.abs(d.ciUpper - d.proportionDelta) : 0
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
          No cluster data available for frequency delta analysis.
        </Alert>
      </Box>
    );
  }

  if (!plotData.length) {
    if (filters.significanceOnly) {
      return (
        <Box sx={{ height }}>
          <Alert severity="info">
            No significant frequency differences found. Try disabling the significance filter.
          </Alert>
        </Box>
      );
    }
    return (
      <Box sx={{ height }}>
        <Alert severity="warning">
          No data matches the current filters. Try adjusting your model selection.
        </Alert>
      </Box>
    );
  }

  // Calculate y-axis range to center around zero
  const allValues = plotData.flatMap(trace => trace.y);
  const maxAbs = Math.max(...allValues.map(Math.abs));
  const yRange = [-maxAbs * 1.1, maxAbs * 1.1];

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
        <Typography variant="h6" component="h3">
          Frequency Δ by Cluster
        </Typography>
        <Typography variant="body2" color="text.secondary">
          How much each model over/under-represents vs. average
          {showCI && ' (with confidence intervals)'}
        </Typography>
      </Box>
      
      <PlotlyChartBase
        data={plotData}
        height={height}
        showZeroLine={true}
        yAxisLabel="Frequency Δ (Proportion Delta)"
        layout={{
          barmode: 'group',
          bargap: 0.2,
          bargroupgap: 0.1,
          yaxis: {
            tickformat: '+.1%',
            range: yRange,
            zeroline: true
          },
          xaxis: {
            tickangle: -45
          }
        }}
      />
    </Box>
  );
}

export default FrequencyDeltaChart;
