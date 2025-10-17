/**
 * BenchmarkTable - Sortable table view of benchmark metrics.
 *
 * Displays per-model benchmark metrics in a sortable table with all quality metrics.
 * Complements the BenchmarkChart by providing detailed numeric values.
 */

import React, { useMemo } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Tooltip,
  IconButton
} from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import { useReactTable, getCoreRowModel, getSortedRowModel, flexRender, createColumnHelper } from '@tanstack/react-table';
import type { SortingState } from '@tanstack/react-table';
import type { ModelBenchmarkRow } from '../../types/metrics';
import { sanitizeMetricName, getDisplayName } from './utils/metricUtils';

interface BenchmarkTableProps {
  data: ModelBenchmarkRow[];
  qualityMetrics: string[];
  showCI?: boolean;
}

export function BenchmarkTable({ data, qualityMetrics, showCI = false }: BenchmarkTableProps) {
  const [sorting, setSorting] = React.useState<SortingState>([]);

  const columnHelper = createColumnHelper<ModelBenchmarkRow>();

  const columns = useMemo(() => {
    const cols: any[] = [
      columnHelper.accessor('model', {
        id: 'model',
        header: 'Model',
        cell: info => {
          const fullName = info.getValue();
          const shortName = fullName.split('/').pop() || fullName;
          return (
            <Tooltip title={fullName} placement="top">
              <Typography
                variant="body2"
                sx={{
                  fontWeight: 500,
                  maxWidth: 200,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap'
                }}
              >
                {shortName}
              </Typography>
            </Tooltip>
          );
        },
        enableSorting: true
      }),
      columnHelper.accessor('size', {
        id: 'size',
        header: 'Conversations',
        cell: info => (
          <Typography variant="body2" align="right">
            {info.getValue().toLocaleString()}
          </Typography>
        ),
        enableSorting: true
      })
    ];

    // Add a column for each quality metric (filter out delta metrics)
    qualityMetrics
      .filter(metric => !metric.includes('delta') && !metric.includes('Delta'))
      .forEach(metric => {
      const sanitized = sanitizeMetricName(metric);
      const qualityKey = `quality_${sanitized}` as keyof ModelBenchmarkRow;
      const ciLowerKey = `${qualityKey}_ci_lower` as keyof ModelBenchmarkRow;
      const ciUpperKey = `${qualityKey}_ci_upper` as keyof ModelBenchmarkRow;

      cols.push(
        columnHelper.accessor(qualityKey, {
          id: qualityKey,
          header: getDisplayName(metric),
          cell: info => {
            const value = info.getValue() as number | undefined;
            if (value === undefined || value === null || !isFinite(value)) {
              return <Typography variant="body2" align="right" color="text.disabled">—</Typography>;
            }

            const row = info.row.original;
            const ciLower = row[ciLowerKey] as number | undefined;
            const ciUpper = row[ciUpperKey] as number | undefined;

            const hasCI = ciLower !== undefined && ciUpper !== undefined;

            return (
              <Tooltip
                title={hasCI ? `95% CI: [${ciLower.toFixed(3)}, ${ciUpper.toFixed(3)}]` : ''}
                placement="top"
              >
                <Typography variant="body2" align="right" sx={{ fontWeight: 500 }}>
                  {value.toFixed(3)}
                  {hasCI && (
                    <Typography
                      component="span"
                      variant="caption"
                      color="text.secondary"
                      sx={{ ml: 0.5, fontSize: '0.7rem' }}
                    >
                      ±{((ciUpper - ciLower) / 2).toFixed(3)}
                    </Typography>
                  )}
                </Typography>
              </Tooltip>
            );
          },
          enableSorting: true,
          sortingFn: 'basic'
        })
      );
    });

    return cols;
  }, [qualityMetrics, showCI, columnHelper]);

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel()
  });

  if (!data.length) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="body2" color="text.secondary">
          No benchmark data available
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ mb: 2 }}>
        <Typography variant="h6" component="h3">
          Benchmark Metrics Table
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Click column headers to sort • Hover over values to see confidence intervals
        </Typography>
      </Box>

      <TableContainer component={Paper} variant="outlined">
        <Table size="small" sx={{ minWidth: 650 }}>
          <TableHead>
            {table.getHeaderGroups().map(headerGroup => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map(header => {
                  const isSorted = header.column.getIsSorted();
                  return (
                    <TableCell
                      key={header.id}
                      align={header.id === 'model' ? 'left' : 'right'}
                      sx={{
                        fontWeight: 600,
                        cursor: header.column.getCanSort() ? 'pointer' : 'default',
                        userSelect: 'none',
                        bgcolor: 'background.default',
                        '&:hover': header.column.getCanSort() ? {
                          bgcolor: 'action.hover'
                        } : {}
                      }}
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: header.id === 'model' ? 'flex-start' : 'flex-end', gap: 0.5 }}>
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {isSorted && (
                          <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center' }}>
                            {isSorted === 'asc' ? (
                              <ArrowUpwardIcon sx={{ fontSize: 16 }} />
                            ) : (
                              <ArrowDownwardIcon sx={{ fontSize: 16 }} />
                            )}
                          </Box>
                        )}
                      </Box>
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
          </TableHead>
          <TableBody>
            {table.getRowModel().rows.map(row => (
              <TableRow
                key={row.id}
                sx={{
                  '&:hover': {
                    bgcolor: 'action.hover'
                  }
                }}
              >
                {row.getVisibleCells().map(cell => (
                  <TableCell
                    key={cell.id}
                    align={cell.column.id === 'model' ? 'left' : 'right'}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

export default BenchmarkTable;
