/**
 * ClusterPlotsSection - Four types of cluster visualization charts.
 * 
 * Renders the four chart types specified in METRICS_README.md:
 * 1. Frequency (absolute proportion)
 * 2. Frequency Delta (proportion delta with zero line)
 * 3. Quality (absolute quality scores)
 * 4. Quality Delta (quality delta with zero line)
 */

import React from 'react';
import { 
  Box, 
  Typography, 
  Paper,
  Stack,
  Grid
} from '@mui/material';
import { FrequencyChart } from './charts/FrequencyChart';
import { QualityDeltaChart } from './charts/QualityDeltaChart';
import type { 
  ModelClusterRow, 
  MetricsFilters 
} from '../../types/metrics';

interface ClusterPlotsSectionProps {
  data: ModelClusterRow[];
  filters: MetricsFilters;
  qualityMetrics: string[];
  showCI: boolean;
  topClusters: string[];
}

export function ClusterPlotsSection({
  data,
  filters,
  qualityMetrics,
  showCI,
  topClusters
}: ClusterPlotsSectionProps) {

  return (
    <Box sx={{ mb: 4 }}>
      <Typography variant="h6" gutterBottom>
        Cluster Analysis
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Showing top {filters.topN} clusters, sorted by {filters.sortBy.replace('_', ' ')}
        {showCI && ' (with confidence intervals)'}
      </Typography>

      <Stack spacing={4} sx={{ mt: 3, width: '100%' }}>
        {/* Frequency Chart Only */}
        <Box sx={{ width: '100%' }}>
          <Paper elevation={1} sx={{ p: 3, height: 520, width: '100%' }}>
            <FrequencyChart
              data={data}
              filters={filters}
              topClusters={topClusters}
              showCI={showCI}
              height={470}
            />
          </Paper>
        </Box>

        {/* Quality Δ Chart Only */}
        <Box sx={{ width: '100%' }}>
          <Paper elevation={1} sx={{ p: 3, height: 520, width: '100%' }}>
            <QualityDeltaChart
              data={data}
              filters={filters}
                topClusters={topClusters}
                showCI={showCI}
                height={470}
              />
            </Paper>
        </Box>
      </Stack>
    </Box>
  );
}

export default ClusterPlotsSection;