/**
 * BenchmarkSection - Per-model benchmark metrics table.
 *
 * Displays a sortable table showing quality scores for each model across 
 * all clusters. This provides a high-level comparison of model performance.
 */

import { Box } from '@mui/material';
import { BenchmarkTable } from './BenchmarkTable';
import type {
  ModelBenchmarkPayload
} from '../../types/metrics';

interface BenchmarkSectionProps {
  data: ModelBenchmarkPayload;
  qualityMetrics: string[];
}

export function BenchmarkSection({
  data,
  qualityMetrics
}: BenchmarkSectionProps) {
  return (
    <Box sx={{ mb: 4 }}>
      <BenchmarkTable
        data={data.data}
        qualityMetrics={qualityMetrics}
        showCI={true}
      />
    </Box>
  );
}

export default BenchmarkSection;