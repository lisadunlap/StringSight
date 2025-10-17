/**
 * ModelCardsGrid - Responsive grid layout for model cards.
 * 
 * Two-column layout on desktop, single column on mobile.
 * Handles the overall grid arrangement and spacing.
 */

import React from 'react';
import { 
  Box,
  Grid
} from '@mui/material';
import { ModelCard } from './ModelCard';
import type { MetricsFilters, ModelCardData } from '../../../types/metrics';

interface ModelCardsGridProps {
  cards: ModelCardData[];
  filters: MetricsFilters;
  qualityMetrics: string[];
}

export function ModelCardsGrid({
  cards,
  filters,
  qualityMetrics
}: ModelCardsGridProps) {
  
  if (cards.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
        No model data available for the selected filters.
      </Box>
    );
  }

  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: {
          xs: '1fr',
          sm: '1fr',
          md: '1fr 1fr'
        },
        gap: 3
      }}
    >
      {cards.map((card) => (
        <Box key={card.model}>
          <ModelCard
            card={card}
            filters={filters}
            qualityMetrics={qualityMetrics}
          />
        </Box>
      ))}
    </Box>
  );
}

export default ModelCardsGrid;
