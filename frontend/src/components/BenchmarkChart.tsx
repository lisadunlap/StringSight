import React from 'react';
import { Box, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Accordion, AccordionSummary, AccordionDetails } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

type BenchmarkPoint = {
  model: string;
  metricKey: string;
  n: number;
  mean: number;
  std: number;
  ciLower: number;
  ciUpper: number;
};

type BenchmarkSummary = {
  points: BenchmarkPoint[];
  metrics: string[];
  models: string[];
};

interface BenchmarkChartProps {
  operationalRows: Array<Record<string, unknown>>;
  method: 'single_model' | 'side_by_side' | 'unknown';
  decimalPrecision?: number;
}

function computeBenchmarkSummaryFromOperational(
  rows: Array<Record<string, unknown>>,
  method: 'single_model' | 'side_by_side' | 'unknown'
): BenchmarkSummary {
  const safeRows = Array.isArray(rows) ? rows : [];

  const observations: Array<{ model: string; metricKey: string; value: number }> = [];

  for (const r of safeRows) {
    if (method === 'single_model') {
      const model = typeof r?.['model'] === 'string' ? (r['model'] as string) : '';
      const score = (r as any)?.score && typeof (r as any).score === 'object' ? ((r as any).score as Record<string, unknown>) : null;
      if (model && score) {
        for (const [k, v] of Object.entries(score)) {
          const num = typeof v === 'number' ? v : typeof v === 'string' ? Number(v) : NaN;
          if (Number.isFinite(num)) observations.push({ model, metricKey: k, value: num });
        }
      }
    } else if (method === 'side_by_side') {
      const modelA = typeof r?.['model_a'] === 'string' ? (r['model_a'] as string) : '';
      const modelB = typeof r?.['model_b'] === 'string' ? (r['model_b'] as string) : '';
      const scoreA = (r as any)?.score_a && typeof (r as any).score_a === 'object' ? ((r as any).score_a as Record<string, unknown>) : null;
      const scoreB = (r as any)?.score_b && typeof (r as any).score_b === 'object' ? ((r as any).score_b as Record<string, unknown>) : null;
      if (modelA && scoreA) {
        for (const [k, v] of Object.entries(scoreA)) {
          const num = typeof v === 'number' ? v : typeof v === 'string' ? Number(v) : NaN;
          if (Number.isFinite(num)) observations.push({ model: modelA, metricKey: k, value: num });
        }
      }
      if (modelB && scoreB) {
        for (const [k, v] of Object.entries(scoreB)) {
          const num = typeof v === 'number' ? v : typeof v === 'string' ? Number(v) : NaN;
          if (Number.isFinite(num)) observations.push({ model: modelB, metricKey: k, value: num });
        }
      }
    }
  }

  const byKey = new Map<string, number[]>();
  const modelsSet = new Set<string>();
  const metricsSet = new Set<string>();
  const groupKey = (m: string, k: string) => `${m}__${k}`;

  for (const o of observations) {
    modelsSet.add(o.model);
    metricsSet.add(o.metricKey);
    const k = groupKey(o.model, o.metricKey);
    const arr = byKey.get(k);
    if (arr) arr.push(o.value); else byKey.set(k, [o.value]);
  }

  const points: BenchmarkPoint[] = [];
  for (const [k, arr] of byKey.entries()) {
    const [model, metricKey] = k.split('__');
    const n = arr.length;
    const mean = arr.reduce((a, b) => a + b, 0) / n;
    const variance = n > 1 ? arr.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / (n - 1) : 0;
    const std = Math.sqrt(variance);
    const ciHalf = n > 1 ? 1.96 * (std / Math.sqrt(n)) : 0;
    points.push({ model, metricKey, n, mean, std, ciLower: mean - ciHalf, ciUpper: mean + ciHalf });
  }

  return {
    points: points.sort((a, b) => a.metricKey.localeCompare(b.metricKey) || a.model.localeCompare(b.model)),
    metrics: Array.from(metricsSet).sort(),
    models: Array.from(modelsSet).sort(),
  };
}

function labelForMetric(metricKey: string): string {
  return metricKey.replace(/_/g, ' ').toUpperCase();
}

export default function BenchmarkChart({ operationalRows, method }: BenchmarkChartProps) {
  const benchmark = React.useMemo(() => computeBenchmarkSummaryFromOperational(operationalRows, method), [operationalRows, method]);

  // Don't render anything if no data
  if (benchmark.metrics.length === 0 || benchmark.models.length === 0) {
    return null;
  }

  // Create a matrix of model -> metric -> point data (including CIs)
  const dataMatrix = React.useMemo(() => {
    const matrix: Record<string, Record<string, BenchmarkPoint | null>> = {};
    
    // Initialize matrix
    benchmark.models.forEach(model => {
      matrix[model] = {};
      benchmark.metrics.forEach(metric => {
        matrix[model][metric] = null;
      });
    });
    
    // Fill in the data
    benchmark.points.forEach(point => {
      matrix[point.model][point.metricKey] = point;
    });
    
    return matrix;
  }, [benchmark]);

  // Find the maximum value for each metric to bold it
  const maxValues = React.useMemo(() => {
    const maxes: Record<string, number> = {};
    
    benchmark.metrics.forEach(metric => {
      let max = -Infinity;
      benchmark.models.forEach(model => {
        const point = dataMatrix[model][metric];
        if (point !== null && point.mean > max) {
          max = point.mean;
        }
      });
      maxes[metric] = max;
    });
    
    return maxes;
  }, [benchmark.metrics, benchmark.models, dataMatrix]);

  return (
    <Box sx={{ mb: 3 }}>
      <Accordion defaultExpanded sx={{ border: '1px solid', borderColor: 'divider' }}>
        <AccordionSummary 
          expandIcon={<ExpandMoreIcon />}
          sx={{ 
            backgroundColor: 'background.paper',
            '&:hover': { backgroundColor: 'action.hover' }
          }}
        >
          <Typography variant="subtitle1" sx={{ fontWeight: 600, fontSize: '1rem' }}>
            Benchmark Overview ({benchmark.models.length} models, {benchmark.metrics.length} metrics)
          </Typography>
        </AccordionSummary>
        <AccordionDetails sx={{ p: 0 }}>
          <TableContainer component={Paper} variant="outlined" sx={{ border: 'none' }}>
            <Table size="small" sx={{ minWidth: 650 }}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 'bold' }}>Model</TableCell>
                  {benchmark.metrics.map(metric => (
                    <TableCell key={metric} align="center" sx={{ fontWeight: 'bold' }}>
                      {labelForMetric(metric)}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {benchmark.models.map(model => (
                  <TableRow key={model} hover>
                    <TableCell sx={{ fontWeight: 500 }}>{model}</TableCell>
                    {benchmark.metrics.map(metric => {
                      const point = dataMatrix[model][metric];
                      const isMax = point !== null && point.mean === maxValues[metric];
                      
                      return (
                        <TableCell 
                          key={metric} 
                          align="center"
                          sx={{ 
                            fontWeight: isMax ? 'bold' : 'normal',
                            color: isMax ? 'primary.main' : 'text.primary',
                            fontFamily: 'monospace',
                            fontSize: '0.875rem'
                          }}
                        >
                          {point !== null ? (
                            `${point.mean.toFixed(4)} [${point.ciLower.toFixed(4)}, ${point.ciUpper.toFixed(4)}]`
                          ) : '—'}
                        </TableCell>
                      );
                    })}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
}
