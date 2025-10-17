/**
 * DataTabBenchmarkTable
 *
 * Adapter to reuse Metrics tab's BenchmarkTable in the Data tab.
 * Converts `operationalRows` into per-model aggregates matching
 * `ModelBenchmarkRow[]` and infers available quality metrics.
 */

import React, { useMemo } from 'react';
import BenchmarkTable from './BenchmarkTable';
import type { ModelBenchmarkRow } from '../../types/metrics';

interface DataTabBenchmarkTableProps {
  operationalRows: Array<Record<string, unknown>>;
  method: 'single_model' | 'side_by_side' | 'unknown';
}

export default function DataTabBenchmarkTable({ operationalRows, method }: DataTabBenchmarkTableProps) {
  const { rows, qualityMetrics } = useMemo(() => {
    const perModel: Record<string, ModelBenchmarkRow> = {};
    const metricSet = new Set<string>();

    const addObservation = (model: string, scoreObj: Record<string, unknown> | null | undefined) => {
      if (!model || !scoreObj || typeof scoreObj !== 'object') return;
      const row = (perModel[model] ||= { model, cluster: 'all_clusters', size: 0, proportion: 1 });
      row.size += 1;
      for (const [k, v] of Object.entries(scoreObj)) {
        const num = typeof v === 'number' ? v : typeof v === 'string' ? Number(v) : NaN;
        if (!Number.isFinite(num)) continue;
        metricSet.add(k);
        const key = `quality_${k}` as keyof ModelBenchmarkRow;
        const accKey = `__acc_${k}`;
        const cntKey = `__cnt_${k}`;
        const sqKey = `__sq_${k}`;
        // Dynamic accumulation keys
        (row as any)[accKey] = ((row as any)[accKey] || 0) + num;
        (row as any)[cntKey] = ((row as any)[cntKey] || 0) + 1;
        (row as any)[sqKey] = ((row as any)[sqKey] || 0) + num * num;
      }
    };

    for (const r of operationalRows || []) {
      if (method === 'single_model') {
        const model = typeof (r as any)?.model === 'string' ? String((r as any).model) : '';
        const score = (r as any)?.score as Record<string, unknown> | null | undefined;
        addObservation(model, score);
      } else if (method === 'side_by_side') {
        const modelA = typeof (r as any)?.model_a === 'string' ? String((r as any).model_a) : '';
        const modelB = typeof (r as any)?.model_b === 'string' ? String((r as any).model_b) : '';
        const scoreA = (r as any)?.score_a as Record<string, unknown> | null | undefined;
        const scoreB = (r as any)?.score_b as Record<string, unknown> | null | undefined;
        addObservation(modelA, scoreA);
        addObservation(modelB, scoreB);
      }
    }

    // Finalize means and confidence intervals per metric (95% CI)
    const rows: ModelBenchmarkRow[] = Object.values(perModel).map((row: any) => {
      metricSet.forEach((k) => {
        const acc = row[`__acc_${k}`];
        const cnt = row[`__cnt_${k}`];
        const sq = row[`__sq_${k}`];
        if (cnt > 0) {
          const mean = acc / cnt;
          row[`quality_${k}`] = mean;
          if (cnt > 1) {
            // sample variance using sum of squares
            const variance = Math.max((sq - cnt * mean * mean) / (cnt - 1), 0);
            const std = Math.sqrt(variance);
            const ciHalf = 1.96 * (std / Math.sqrt(cnt));
            (row as any)[`quality_${k}_ci_lower`] = mean - ciHalf;
            (row as any)[`quality_${k}_ci_upper`] = mean + ciHalf;
          }
        }
        delete row[`__acc_${k}`];
        delete row[`__cnt_${k}`];
        delete row[`__sq_${k}`];
      });
      return row as ModelBenchmarkRow;
    });

    return { rows, qualityMetrics: Array.from(metricSet).sort() };
  }, [operationalRows, method]);

  return (
    <BenchmarkTable data={rows} qualityMetrics={qualityMetrics} showCI={true} />
  );
}


