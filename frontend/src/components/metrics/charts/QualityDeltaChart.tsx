/**
 * QualityDeltaChart - Quality delta by cluster.
 * 
 * Shows grouped bars by model displaying how much better/worse each model
 * performs in each cluster compared to its overall baseline. Includes zero line.
 */

import React, { useMemo } from 'react';
import { Box, Typography, Alert } from '@mui/material';
import { PlotlyChartBase, getModelColor, truncateLabel, createHoverTemplate } from './PlotlyChartBase';
import { getDisplayName, getOriginalMetricName, sanitizeMetricName } from '../utils/metricUtils';
import type { ModelClusterRow, MetricsFilters } from '../../../types/metrics';

interface QualityDeltaChartProps {
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

export function QualityDeltaChart({
  data,
  filters,
  topClusters,
  showCI = false,
  height = 400
}: QualityDeltaChartProps) {
  
  const plotData = useMemo(() => {
    if (!data.length || !filters.qualityMetric) {
      return [];
    }

    // Filter data by selected models
    const filteredData = data.filter(row => 
      filters.selectedModels.length === 0 || filters.selectedModels.includes(row.model)
    );

    if (!filteredData.length) {
      return [];
    }

  // Prepare both name variants for nested vs flattened data
  const originalMetric = getOriginalMetricName(filters.qualityMetric);
  const sanitizedMetric = sanitizeMetricName(filters.qualityMetric);
  
  // Debug logging
  if (import.meta.env.DEV) {
    const sampleRow = filteredData[0];
    console.log('QualityDeltaChart Debug:', {
      qualityMetric: filters.qualityMetric,
      originalMetric,
      hasQualityDeltaObject: sampleRow && 'quality_delta' in sampleRow,
      qualityDeltaKeys: sampleRow?.quality_delta ? Object.keys(sampleRow.quality_delta) : [],
      metricInQualityDelta: sampleRow?.quality_delta && originalMetric in sampleRow.quality_delta
    });
  }

  // Accept nested `quality_delta` object or flattened `quality_delta_<metric>` columns
  const dataWithQualityDelta = filteredData.filter(row => {
    const nested = (row as any)?.quality_delta?.[originalMetric];
    if (typeof nested === 'number' && isFinite(nested)) return true;
    const flat1 = (row as any)[`quality_delta_${filters.qualityMetric}`];
    const flat2 = (row as any)[`quality_delta_${sanitizedMetric}`];
    return [flat1, flat2].some(v => typeof v === 'number' && isFinite(v as number));
  });

    if (!dataWithQualityDelta.length) {
      return [];
    }

    // Apply significance filter if enabled
    let finalData = dataWithQualityDelta;
    if (filters.significanceOnly) {
      finalData = dataWithQualityDelta.filter(row => {
        const significant = row?.quality_delta_significant?.[originalMetric];
        return significant === true;
      });
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
        const nested = row?.quality_delta?.[originalMetric];
        const flat1 = (row as any)?.[`quality_delta_${filters.qualityMetric}`];
        const flat2 = (row as any)?.[`quality_delta_${sanitizedMetric}`];
        const deltaValue = typeof nested === 'number' && isFinite(nested)
          ? nested
          : (typeof flat1 === 'number' && isFinite(flat1) ? flat1 : (typeof flat2 === 'number' && isFinite(flat2) ? flat2 : 0));
        
        return {
          cluster,
          qualityDelta: deltaValue,
          // CI support for both nested and flattened format
          ciLower: showCI ? (
            row?.quality_delta_ci?.[originalMetric]?.lower ||
            (row as any)?.[`quality_delta_${filters.qualityMetric}_ci_lower`] ||
            (row as any)?.[`quality_delta_${sanitizedMetric}_ci_lower`]
          ) : undefined,
          ciUpper: showCI ? (
            row?.quality_delta_ci?.[originalMetric]?.upper ||
            (row as any)?.[`quality_delta_${filters.qualityMetric}_ci_upper`] ||
            (row as any)?.[`quality_delta_${sanitizedMetric}_ci_upper`]
          ) : undefined,
          size: row?.size || 0,
          significant: row?.quality_delta_significant?.[originalMetric] || false
        };
      });

      const deltas = modelData.map(d => d.qualityDelta);
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
            delta, 
            `${getDisplayName(filters.qualityMetric)} Δ`, 
            3
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
          (d.ciLower !== undefined && d.qualityDelta !== undefined) ? 
            Math.abs(d.qualityDelta - d.ciLower) : 0
        );
        const arrayplus = modelData.map(d => 
          (d.ciUpper !== undefined && d.qualityDelta !== undefined) ? 
            Math.abs(d.ciUpper - d.qualityDelta) : 0
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
          No cluster data available for quality delta analysis.
        </Alert>
      </Box>
    );
  }

  if (!filters.qualityMetric) {
    return (
      <Box sx={{ height }}>
        <Alert severity="warning">
          Please select a quality metric to display quality deltas.
        </Alert>
      </Box>
    );
  }

  if (!plotData.length) {
    if (filters.significanceOnly) {
      return (
        <Box sx={{ height }}>
          <Alert severity="info">
            No significant quality differences found for "{getDisplayName(filters.qualityMetric)}". 
            Try disabling the significance filter.
          </Alert>
        </Box>
      );
    }
    return (
      <Box sx={{ height }}>
        <Alert severity="warning">
          No data available for quality metric "{getDisplayName(filters.qualityMetric)}". 
          Try selecting a different metric or adjusting filters.
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
          Quality Δ by Cluster
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {getDisplayName(filters.qualityMetric)} performance vs. model baseline
          {showCI && ' (with confidence intervals)'}
        </Typography>
      </Box>
      
      <PlotlyChartBase
        data={plotData}
        height={height}
        showZeroLine={true}
        yAxisLabel={`${getDisplayName(filters.qualityMetric)} Δ`}
        layout={{
          barmode: 'group',
          bargap: 0.2,
          bargroupgap: 0.1,
          yaxis: {
            tickformat: '+.2f',
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

export default QualityDeltaChart;
