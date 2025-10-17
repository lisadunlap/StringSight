/**
 * ModelCardsSection - Two-column grid of model cards.
 * 
 * Each card shows:
 * - Model name and total battles
 * - Top 5 clusters by proportion_delta  
 * - Color-coded borders (green/red for positive/negative delta)
 * - Significance badges (F for frequency, Q for quality)
 * - Cluster metadata tags as chips
 * - Quality metrics with delta indicators
 */

import React, { useMemo } from 'react';
import { 
  Box, 
  Typography
} from '@mui/material';
import { ModelCardsGrid } from './cards/ModelCardsGrid';
import type { 
  ModelClusterRow, 
  MetricsFilters,
  ModelCardData
} from '../../types/metrics';

interface ModelCardsSectionProps {
  data: ModelClusterRow[];
  filters: MetricsFilters;
  qualityMetrics: string[];
  totalBattles: number;
}

export function ModelCardsSection({
  data,
  filters,
  qualityMetrics,
  totalBattles
}: ModelCardsSectionProps) {
  
  // Group data by model and create card data
  const modelCards: ModelCardData[] = useMemo(() => {
    // Filter by selected models first
    const filteredData = data.filter(row => 
      filters.selectedModels.length === 0 || filters.selectedModels.includes(row.model)
    );

    const modelGroups = filteredData.reduce((groups, row) => {
      if (!groups[row.model]) {
        groups[row.model] = [];
      }
      groups[row.model].push(row);
      return groups;
    }, {} as Record<string, ModelClusterRow[]>);

    return Object.entries(modelGroups).map(([model, clusters]) => {
      // Apply significance filter if enabled
      let clusterCandidates = clusters;
      if (filters.significanceOnly) {
        const qualityDeltaSigKey = `quality_delta_${filters.qualityMetric}_significant`;
        clusterCandidates = clusters.filter(cluster => 
          cluster.proportion_delta_significant === true ||
          cluster[qualityDeltaSigKey as keyof ModelClusterRow] === true
        );
      }

      // Sort clusters by proportion_delta and take top 5
      const topClusters = clusterCandidates
        .sort((a, b) => (b.proportion_delta || 0) - (a.proportion_delta || 0))
        .slice(0, 5);

      return {
        model,
        totalBattles: clusters.reduce((sum, cluster) => sum + (cluster.size || 0), 0),
        clusters,
        topClusters
      };
    }).filter(card => card.topClusters.length > 0); // Only include models with clusters
  }, [data, filters]);

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Top Clusters per Model
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        Most distinctive clusters by frequency delta for each model
        {filters.significanceOnly && ' (significant differences only)'}
      </Typography>

      <Box sx={{ mt: 3 }}>
        <ModelCardsGrid
          cards={modelCards}
          filters={filters}
          qualityMetrics={qualityMetrics}
        />
      </Box>
    </Box>
  );
}

export default ModelCardsSection;