import React from 'react';
import { Box, Typography, Stack, Divider, FormControl, InputLabel, Select, MenuItem } from '@mui/material';

interface DataStatsPanelProps {
  dataOverview: {
    rowCount: string;
    uniquePrompts: string;
    uniqueModels: string;
  } | null;
  method: 'single_model' | 'side_by_side' | 'unknown';
  operationalRows?: Array<Record<string, unknown>>;
  decimalPrecision?: number;
  onDecimalPrecisionChange?: (precision: number) => void;
  uploadedFilename?: string | null;
}


export default function DataStatsPanel({ 
  dataOverview, 
  method, 
  operationalRows = [], 
  decimalPrecision = 2, 
  onDecimalPrecisionChange,
  uploadedFilename 
}: DataStatsPanelProps) {
  // Debug logging
  console.log('[DataStatsPanel] uploadedFilename:', uploadedFilename, 'dataOverview:', dataOverview);
  
  // Show filename even if no data is loaded yet
  if (!dataOverview) {
    return (
      <Stack spacing={3}>
        {uploadedFilename && (
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
              Source File
            </Typography>
            <Box sx={{ 
              p: 2, 
              backgroundColor: 'primary.50', 
              borderRadius: 1,
              border: '1px solid',
              borderColor: 'primary.200'
            }}>
              <Typography 
                variant="body2" 
                sx={{ 
                  fontFamily: 'monospace',
                  wordBreak: 'break-all',
                  color: 'primary.800'
                }}
              >
                {uploadedFilename}
              </Typography>
            </Box>
          </Box>
        )}
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="body2" color="text.secondary">
            {uploadedFilename 
              ? 'Configure column mapping to see dataset statistics.'
              : 'No data loaded yet. Upload a dataset to see statistics.'}
          </Typography>
        </Box>
      </Stack>
    );
  }

  return (
    <Stack spacing={3}>
      {uploadedFilename && (
        <>
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
              Source File
            </Typography>
            <Box sx={{ 
              p: 2, 
              backgroundColor: 'primary.50', 
              borderRadius: 1,
              border: '1px solid',
              borderColor: 'primary.200'
            }}>
              <Typography 
                variant="body2" 
                sx={{ 
                  fontFamily: 'monospace',
                  wordBreak: 'break-all',
                  color: 'primary.800'
                }}
              >
                {uploadedFilename}
              </Typography>
            </Box>
          </Box>
          <Divider />
        </>
      )}
      
      <Box>
        <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
          Dataset Overview
        </Typography>
        <Stack spacing={2}>
          <Box>
            <Typography variant="body2" color="text.secondary">
              Total Rows
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              {dataOverview.rowCount}
            </Typography>
          </Box>
          
          <Box>
            <Typography variant="body2" color="text.secondary">
              Unique Prompts
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              {dataOverview.uniquePrompts}
            </Typography>
          </Box>
          
          <Box>
            <Typography variant="body2" color="text.secondary">
              Unique Models
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              {dataOverview.uniqueModels}
            </Typography>
          </Box>
        </Stack>
      </Box>

      <Divider />

      <Box>
        <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
          Data Format
        </Typography>
        <Box sx={{ 
          p: 2, 
          backgroundColor: 'background.default', 
          borderRadius: 1,
          border: '1px solid',
          borderColor: 'divider'
        }}>
          <Typography variant="body2" color="text.secondary">
            Detection Method
          </Typography>
          <Typography variant="body1" sx={{ fontWeight: 500, textTransform: 'capitalize' }}>
            {method === 'single_model' ? 'Single Model' : 
             method === 'side_by_side' ? 'Side by Side' : 
             'Unknown Format'}
          </Typography>
        </Box>
      </Box>

      <Divider />

      <Box>
        <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
          Display Settings
        </Typography>
        <FormControl size="small" fullWidth>
          <InputLabel>Decimal Precision</InputLabel>
          <Select
            value={decimalPrecision}
            label="Decimal Precision"
            onChange={(e) => onDecimalPrecisionChange?.(Number(e.target.value))}
          >
            <MenuItem value={0}>0 decimals</MenuItem>
            <MenuItem value={1}>1 decimal</MenuItem>
            <MenuItem value={2}>2 decimals</MenuItem>
            <MenuItem value={3}>3 decimals</MenuItem>
            <MenuItem value={4}>4 decimals</MenuItem>
            <MenuItem value={5}>5 decimals</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Box>
        <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
          Use the filters above the data table to explore and narrow down your dataset.
        </Typography>
      </Box>
    </Stack>
  );
}

