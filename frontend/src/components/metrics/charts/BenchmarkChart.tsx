/**
 * BenchmarkChart - Per-model benchmark metrics bar chart.
 * 
 * Displays horizontal bars showing quality scores for each model across all clusters.
 * Includes optional confidence interval error bars when CI data is available.
 */

import React, { useMemo } from 'react';
import { Box, Typography, Alert } from '@mui/material';
import { PlotlyChartBase, getModelColor } from './PlotlyChartBase';
import { sanitizeMetricName, getDisplayName } from '../utils/metricUtils';
import type { ModelBenchmarkRow, MetricsFilters } from '../../../types/metrics';

interface BenchmarkChartProps {
  /** Benchmark data from backend */
  data: ModelBenchmarkRow[];
  /** Current filters (for selected quality metric) */
  filters: MetricsFilters;
  /** Available quality metrics */
  qualityMetrics: string[];
  /** Whether to show confidence intervals */
  showCI?: boolean;
  /** Chart height */
  height?: number;
}

export function BenchmarkChart({
  data,
  filters,
  qualityMetrics,
  showCI = false,
  height = 400
}: BenchmarkChartProps) {
  
  const plotData = useMemo(() => {
    if (!data.length || !filters.qualityMetric) {
      return [];
    }

    // Try both original and sanitized metric names
    const originalKey = `quality_${filters.qualityMetric}`;
    const sanitizedKey = `quality_${sanitizeMetricName(filters.qualityMetric)}`;
    
    // Determine which key exists in the data  
    const sampleRow = data[0];
    const qualityKey = sampleRow && originalKey in sampleRow ? originalKey : sanitizedKey;
    const ciLowerKey = `${qualityKey}_ci_lower`;
    const ciUpperKey = `${qualityKey}_ci_upper`;
    
    // Filter and sort models by the selected metric (descending)
    const modelsWithScores = data
      .map(row => ({
        model: row.model,
        score: row[qualityKey as keyof ModelBenchmarkRow] as number,
        ciLower: showCI ? (row[ciLowerKey as keyof ModelBenchmarkRow] as number) : undefined,
        ciUpper: showCI ? (row[ciUpperKey as keyof ModelBenchmarkRow] as number) : undefined,
        size: row.size
      }))
      .filter(item => typeof item.score === 'number' && isFinite(item.score))
      .sort((a, b) => b.score - a.score); // Descending order

    if (!modelsWithScores.length) {
      return [];
    }

    const models = modelsWithScores.map(item => item.model);
    const scores = modelsWithScores.map(item => item.score);
    const modelLabels = models.map(model => {
      // Truncate long model names for display
      const shortName = model.split('/').pop() || model;
      return shortName.length > 25 ? shortName.substring(0, 22) + '...' : shortName;
    });

    // Create horizontal bar chart data
    const barData: any = {
      type: 'bar',
      orientation: 'h',
      x: scores,
      y: modelLabels,
      marker: {
        color: models.map(model => getModelColor(model, models)),
        opacity: 0.8
      },
      // Values shown in hover only
      hovertemplate: models.map((model, i) => 
        `<b>${model}</b><br>` +
        `${getDisplayName(filters.qualityMetric)}: ${scores[i].toFixed(3)}<br>` +
        `Conversations: ${modelsWithScores[i].size}<extra></extra>`
      ),
      name: filters.qualityMetric
    };

    // Add confidence interval error bars if available and requested
    if (showCI && modelsWithScores.some(item => 
      item.ciLower !== undefined && item.ciUpper !== undefined
    )) {
      // For horizontal bar chart, use error_x for horizontal error bars
      const arrayminus = scores.map((score, i) => 
        modelsWithScores[i].ciLower !== undefined 
          ? Math.max(0, score - modelsWithScores[i].ciLower!) 
          : 0
      );
      const arrayplus = scores.map((score, i) => 
        modelsWithScores[i].ciUpper !== undefined 
          ? Math.max(0, modelsWithScores[i].ciUpper! - score)
          : 0
      );

      barData.error_x = {
        type: 'data',
        symmetric: false,
        array: arrayplus,
        arrayminus: arrayminus,
        visible: true,
        thickness: 2,
        width: 5,
        color: '#1976d2' // Material UI primary color
      };
    }

    return [barData];
  }, [data, filters.qualityMetric, showCI]);

  // Show error state if no data available
  if (!data.length) {
    return (
      <Box sx={{ height }}>
        <Alert severity="info">
          No benchmark data available. Ensure model scores are computed and available.
        </Alert>
      </Box>
    );
  }

  // Show error if no quality metric selected
  if (!filters.qualityMetric || !qualityMetrics.includes(filters.qualityMetric)) {
    return (
      <Box sx={{ height }}>
        <Alert severity="warning">
          Please select a quality metric. Available metrics: {qualityMetrics.join(', ')}
        </Alert>
      </Box>
    );
  }

  // Show error if no data for selected metric
  if (!plotData.length) {
    return (
      <Box sx={{ height }}>
        <Alert severity="warning">
          No data available for quality metric "{getDisplayName(filters.qualityMetric)}". 
          Try selecting a different metric.
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
        <Typography variant="h6" component="h3">
          Model Benchmark
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {getDisplayName(filters.qualityMetric)} scores across all clusters
          {showCI && ' (with 95% confidence intervals)'}
        </Typography>
      </Box>
      
      <PlotlyChartBase
        data={plotData}
        height={height}
        xAxisLabel={`${getDisplayName(filters.qualityMetric)}`}
        // yAxisLabel="Models"
        layout={{
          margin: { t: 30, r: 60, b: 60, l: 120 }, // More left margin for model names
          showlegend: false, // Single metric, no legend needed
          xaxis: {
            side: 'bottom',
            tickformat: '.3f'
          },
          yaxis: {
            autorange: 'reversed' // Highest scores at top
          }
        }}
      />
    </Box>
  );
}

export default BenchmarkChart;
