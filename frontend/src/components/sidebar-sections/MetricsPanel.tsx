/**
 * MetricsPanel - Sidebar panel for metrics filtering and configuration.
 * 
 * This wraps the MetricsControlPanel component and provides filter controls
 * for both the Metrics and Model Cards tabs.
 */

import { Box } from '@mui/material';
import { MetricsControlPanel } from '../metrics/MetricsControlPanel';
import type { MetricsFilters, MetricsSummary } from '../../types/metrics';

interface MetricsPanelProps {
  filters: MetricsFilters;
  onFiltersChange: (filters: MetricsFilters) => void;
  availableModels: string[];
  availableGroups: string[];
  availableQualityMetrics: string[];
  summary?: MetricsSummary;
}

export default function MetricsPanel({
  filters,
  onFiltersChange,
  availableModels,
  availableGroups,
  availableQualityMetrics,
  summary
}: MetricsPanelProps) {
  return (
    <Box sx={{ height: '100%', overflow: 'auto' }}>
      <MetricsControlPanel
        filters={filters}
        onFiltersChange={onFiltersChange}
        availableModels={availableModels}
        availableGroups={availableGroups}
        availableQualityMetrics={availableQualityMetrics}
        summary={summary}
      />
    </Box>
  );
}
