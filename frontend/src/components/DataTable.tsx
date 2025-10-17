import React, { useMemo } from "react";
import { useReactTable, getCoreRowModel, flexRender, createColumnHelper } from "@tanstack/react-table";
import { Box, Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Fade, Pagination } from "@mui/material";
import VisibilityOutlinedIcon from "@mui/icons-material/VisibilityOutlined";
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import FormattedCell from './FormattedCell';

const DataTable = React.memo(function DataTable({
  rows,
  columns,
  responseKeys,
  onView,
  allowedColumns,
  sortColumn,
  sortDirection,
  onSort,
  decimalPrecision = 3,
}: {
  rows: Record<string, any>[];
  columns: string[];
  responseKeys: string[]; // keys where an eye icon should appear
  onView: (row: Record<string, any>) => void;
  allowedColumns?: string[]; // limit visible columns
  sortColumn?: string | null;
  sortDirection?: 'asc' | 'desc' | null;
  onSort?: (column: string) => void;
  decimalPrecision?: number;
}) {
  const columnHelper = createColumnHelper<Record<string, any>>();

  const MAX_LEN = 200;
  const TruncatedCell = React.memo(function TruncatedCell({ text }: { text: string }) {
    const [expanded, setExpanded] = React.useState(false);
    if (!expanded && text.length > MAX_LEN) {
      return (
        <span>
          {text.slice(0, MAX_LEN)}…{' '}
          <Button size="small" variant="text" onClick={() => setExpanded(true)}>Expand</Button>
        </span>
      );
    }
    if (expanded && text.length > MAX_LEN) {
      return (
        <span>
          {text}{' '}
          <Button size="small" variant="text" onClick={() => setExpanded(false)}>Collapse</Button>
        </span>
      );
    }
    return <span>{text}</span>;
  });

  // Animate only on initial mount (first paint) for the first 20 rows
  const animateOnMountRef = React.useRef(true);
  React.useEffect(() => {
    const id = requestAnimationFrame(() => {
      animateOnMountRef.current = false;
    });
    return () => cancelAnimationFrame(id);
  }, []);

  const displayColumns = useMemo(() => {
    const human: Record<string, string> = {
      __index: "INDEX",
      prompt: "PROMPT",
      model: "MODEL",
      model_response: "RESPONSE",
      model_a: "MODEL A",
      model_b: "MODEL B",
      model_a_response: "RESPONSE A",
      model_b_response: "RESPONSE B",
      score: "SCORE",
      score_a: "SCORE A",
      score_b: "SCORE B",
    };

    const baseRaw = allowedColumns && allowedColumns.length > 0
      ? allowedColumns.filter((c) => columns.includes(c))
      : columns;

    // Order: index → prompt → response columns → remaining
    const indexCol = baseRaw.filter((c) => c === '__index');
    const promptFirst = baseRaw.filter((c) => c === 'prompt');
    const resp = baseRaw.filter((c) => responseKeys.includes(c));
    // Remove question_id from remaining (we already show __index)
    const remaining = baseRaw.filter((c) => c !== '__index' && c !== 'prompt' && c !== 'question_id' && !responseKeys.includes(c));
    const base = [...indexCol, ...promptFirst, ...resp, ...remaining];

    const headerLabelFor = (col: string): string => {
      if (human[col]) return human[col];
      if (col.startsWith('score_a_')) {
        return (("A ") + col.slice(8).replace(/_/g, ' ')).toUpperCase();
      }
      if (col.startsWith('score_b_')) {
        return (("B ") + col.slice(8).replace(/_/g, ' ')).toUpperCase();
      }
      if (col.startsWith('score_')) {
        return col.slice(6).replace(/_/g, ' ').toUpperCase();
      }
      return col.toUpperCase();
    };

    return base.map((col) => {
      const isResponse = responseKeys.includes(col);
      return columnHelper.accessor((row) => row[col], {
        id: col,
        header: headerLabelFor(col),
        cell: (info) => {
          if (isResponse) {
            return (
              <Button
                size="small"
                variant="text"
                color="secondary"
                startIcon={<VisibilityOutlinedIcon />}
                onClick={() => onView(info.row.original)}
                sx={{ fontWeight: 600 }}
              >
                View
              </Button>
            );
          }
          const value = info.getValue();
          // Treat nested objects as simple strings (scores should be flattened already)
          if (typeof value === "object" && value !== null) {
            return <span>[object]</span>;
          }
          
          // Check if this is a score column and round to configured decimals
          const isScoreColumn = col.includes('score');
          let displayValue = value;
          if (isScoreColumn && value !== null && value !== undefined && !isNaN(Number(value)) && value !== '') {
            displayValue = Number(value).toFixed(decimalPrecision);
          }
          
          const str = String(displayValue ?? "");
          
          // Check if this is a numeric column (index or numeric value)
          const isNumeric = col === '__index' || (value !== null && value !== undefined && !isNaN(Number(value)) && value !== '');
          
          // Check if this is a prompt column that should use rich formatting
          const isPrompt = col === 'prompt';
          
          return (
            <Box sx={{ textAlign: isNumeric ? 'center' : 'left' }}>
              {isPrompt ? (
                <FormattedCell text={str} isPrompt={true} />
              ) : (
                <TruncatedCell text={str} />
              )}
            </Box>
          );
        },
      });
    });
  }, [columns, allowedColumns, responseKeys, onView, decimalPrecision]);

  // Client-side pagination (100 rows per page)
  const PAGE_SIZE = 100;
  const [page, setPage] = React.useState(1);
  React.useEffect(() => { setPage(1); }, [rows]);
  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const startIdx = (page - 1) * PAGE_SIZE;
  const endIdx = Math.min(rows.length, startIdx + PAGE_SIZE);
  const displayRows = useMemo(() => rows.slice(startIdx, endIdx), [rows, startIdx, endIdx]);

  const table = useReactTable({
    data: displayRows,
    columns: displayColumns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <>
      {rows.length > PAGE_SIZE && (
        <Box sx={{ mb: 1, p: 1, background: '#FFFBEB', color: '#92400E', border: '1px solid #F59E0B', borderRadius: 1, fontSize: 14 }}>
          Showing rows {String(startIdx + 1)}–{String(endIdx)} of {rows.length.toLocaleString()} rows.
        </Box>
      )}
      <TableContainer sx={{ border: '1px solid #E5E7EB', borderRadius: 2, overflow: 'auto', backgroundColor: '#FFFFFF' }}>
        <Table size="small">
        <TableHead sx={{ backgroundColor: '#F3F4F6' }}>
          {table.getHeaderGroups().map((hg) => (
            <TableRow key={hg.id}>
              {hg.headers.map((h) => (
                <TableCell 
                  key={h.id} 
                  sx={{ 
                    color: '#374151', 
                    fontWeight: 700, 
                    fontSize: 12, 
                    letterSpacing: 0.4,
                    cursor: onSort ? 'pointer' : 'default',
                    '&:hover': onSort ? { backgroundColor: '#F9FAFB' } : {}
                  }}
                  onClick={() => onSort && onSort(h.column.id)}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    {h.isPlaceholder ? null : flexRender(h.column.columnDef.header, h.getContext())}
                    {onSort && sortColumn === h.column.id && sortDirection === 'asc' && <ArrowUpwardIcon sx={{ fontSize: 12 }} />}
                    {onSort && sortColumn === h.column.id && sortDirection === 'desc' && <ArrowDownwardIcon sx={{ fontSize: 12 }} />}
                  </Box>
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableHead>
        <TableBody>
          {table.getRowModel().rows.map((r, idx) => {
            const rowEl = (
              <TableRow hover key={r.id}>
                {r.getVisibleCells().map((c) => (
                  <TableCell key={c.id} sx={{ borderBottom: '1px solid #E5E7EB' }}>
                    {flexRender(c.column.columnDef.cell, c.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            );
            if (animateOnMountRef.current && idx < 20) {
              return (
                <Fade in timeout={Math.min(1100 + idx * 220, 5200)} key={`fade-${r.id}`}>
                  {rowEl}
                </Fade>
              );
            }
            return rowEl;
          })}
        </TableBody>
        </Table>
      </TableContainer>
      {rows.length > PAGE_SIZE && (
        <Box sx={{ mt: 1, display: 'flex', justifyContent: 'center' }}>
          <Pagination count={totalPages} page={page} onChange={(_, p) => setPage(p)} size="small" />
        </Box>
      )}
    </>
  );
});

export default DataTable;
