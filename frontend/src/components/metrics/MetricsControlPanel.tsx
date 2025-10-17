/**
 * MetricsControlPanel - Left sidebar with filtering and configuration controls.
 * 
 * This component provides all the filter controls:
 * - Model multi-select with global filtering
 * - Group multi-select (based on metadata.group)
 * - Quality metric dropdown (dynamic based on available data)
 * - Top-N slider (1-50, default 15)
 * - Significance toggle (metric-specific logic)
 * - Show CI toggle (when CI columns available)
 * - Advanced sort by dropdown
 */

import {
  Box,
  Paper,
  Typography,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  FormControlLabel,
  Switch,
  Chip,
  Stack,
  Autocomplete,
  TextField,
  Tooltip,
  IconButton
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import TuneIcon from '@mui/icons-material/Tune';
import type { 
  MetricsFilters, 
  MetricsSortOption, 
  MetricsSummary 
} from '../../types/metrics';
import { getDisplayName } from './utils/metricUtils';

interface MetricsControlPanelProps {
  filters: MetricsFilters;
  onFiltersChange: (filters: MetricsFilters) => void;
  availableModels: string[];
  availableGroups: string[];
  availableQualityMetrics: string[];
  summary?: MetricsSummary;
  sx?: object;
}

export function MetricsControlPanel({
  filters,
  onFiltersChange,
  availableModels,
  availableGroups,
  availableQualityMetrics,
  summary,
  sx = {}
}: MetricsControlPanelProps) {
  // Handle filter updates
  const updateFilters = (updates: Partial<MetricsFilters>) => {
    onFiltersChange({ ...filters, ...updates });
  };

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        height: '100%', 
        p: 2, 
        overflow: 'auto',
        bgcolor: 'background.paper',
        ...sx 
      }}
    >
      {/* Header */}
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 2 }}>
        <TuneIcon color="primary" />
        <Typography variant="h6" component="h2">
          Filters & Controls
        </Typography>
      </Stack>

      <Divider sx={{ mb: 2 }} />

      {/* Summary Stats */}
      {summary && (
        <Box sx={{ mb: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
          <Typography variant="caption" color="text.secondary" gutterBottom>
            Dataset Summary
          </Typography>
          <Stack spacing={0.5}>
            <Typography variant="body2">
              {summary.models} models, {summary.clusters} clusters
            </Typography>
            <Typography variant="body2">
              {summary.total_battles.toLocaleString()} battles
            </Typography>
            <Typography variant="body2">
              {summary.quality_metrics} quality metrics
            </Typography>
            {summary.has_confidence_intervals && (
              <Chip 
                label="With Confidence Intervals" 
                size="small" 
                color="info" 
                variant="outlined"
              />
            )}
          </Stack>
        </Box>
      )}

      {/* Model Selection */}
      <FormControl fullWidth sx={{ mb: 2 }}>
        <Autocomplete
          multiple
          options={availableModels}
          value={filters.selectedModels}
          onChange={(_, newValue) => updateFilters({ selectedModels: newValue })}
          renderTags={(value, getTagProps) =>
            value.map((option, index) => {
              const props = getTagProps({ index });
              return (
                <Chip
                  {...props}
                  key={option}
                  label={option.split('/').pop() || option} // Show just model name
                  size="small"
                  variant="outlined"
                />
              );
            })
          }
          renderInput={(params) => (
            <TextField
              {...params}
              label="Models"
              placeholder={filters.selectedModels.length === 0 ? "All models" : ""}
              helperText={`${filters.selectedModels.length || availableModels.length} of ${availableModels.length} selected`}
            />
          )}
          ChipProps={{ size: 'small' }}
        />
      </FormControl>

      {/* Group Selection */}
      <FormControl fullWidth sx={{ mb: 2 }}>
        <Autocomplete
          multiple
          options={availableGroups}
          value={filters.selectedGroups}
          onChange={(_, newValue) => updateFilters({ selectedGroups: newValue })}
          renderTags={(value, getTagProps) =>
            value.map((option, index) => (
              <Chip
                {...getTagProps({ index })}
                key={option}
                label={option}
                size="small"
                variant="outlined"
              />
            ))
          }
          renderInput={(params) => (
            <TextField
              {...params}
              label="Groups"
              placeholder={filters.selectedGroups.length === 0 ? "All groups" : ""}
              helperText="Filter by cluster metadata groups"
            />
          )}
        />
      </FormControl>

      <Divider sx={{ my: 2 }} />

      {/* Quality Metric Selection */}
      {availableQualityMetrics.length > 0 && (
        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel>Quality Metric</InputLabel>
          <Select
            value={filters.qualityMetric}
            label="Quality Metric"
            onChange={(e) => updateFilters({ qualityMetric: e.target.value })}
          >
            {availableQualityMetrics
              .filter(metric => !metric.includes('delta') && !metric.includes('Delta'))
              .map((metric) => (
              <MenuItem key={metric} value={metric}>
                <Stack>
                  <Typography variant="body2">
                    {getDisplayName(metric)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {metric}
                  </Typography>
                </Stack>
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      )}

      {/* Top-N Slider */}
      <Box sx={{ mb: 3 }}>
        <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
          <Typography variant="subtitle2">
            Top N Clusters
          </Typography>
          <Tooltip title="Number of top clusters to display">
            <IconButton size="small">
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Stack>
        <Slider
          value={filters.topN}
          onChange={(_, newValue) => updateFilters({ topN: newValue as number })}
          min={1}
          max={50}
          step={1}
          marks={[
            { value: 1, label: '1' },
            { value: 15, label: '15' },
            { value: 30, label: '30' },
            { value: 50, label: '50' }
          ]}
          valueLabelDisplay="auto"
          sx={{ px: 1 }}
        />
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Significance Toggle */}
      <FormControlLabel
        control={
          <Switch
            checked={filters.significanceOnly}
            onChange={(e) => updateFilters({ significanceOnly: e.target.checked })}
            color="primary"
          />
        }
        label={
          <Stack>
            <Typography variant="body2">
              Significant Only
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Show only statistically significant differences
            </Typography>
          </Stack>
        }
        sx={{ mb: 2 }}
      />

      {/* Show Confidence Intervals Toggle */}
      {summary?.has_confidence_intervals && (
        <FormControlLabel
          control={
            <Switch
              checked={filters.showCI}
              onChange={(e) => updateFilters({ showCI: e.target.checked })}
              color="primary"
            />
          }
          label={
            <Stack>
              <Typography variant="body2">
                Show Confidence Intervals
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Display error bars on charts
              </Typography>
            </Stack>
          }
          sx={{ mb: 2 }}
        />
      )}

      {/* Sort By (Advanced) */}
      <Box sx={{ mt: 3 }}>
        <Typography variant="caption" color="text.secondary" gutterBottom>
          Advanced
        </Typography>
        <FormControl fullWidth size="small">
          <InputLabel>Sort By</InputLabel>
          <Select
            value={filters.sortBy}
            label="Sort By"
            onChange={(e) => updateFilters({ sortBy: e.target.value as MetricsSortOption })}
          >
            <MenuItem value="proportion_desc">Proportion (High to Low)</MenuItem>
            <MenuItem value="proportion_asc">Proportion (Low to High)</MenuItem>
            <MenuItem value="proportion_delta_desc">Proportion Δ (High to Low)</MenuItem>
            <MenuItem value="proportion_delta_asc">Proportion Δ (Low to High)</MenuItem>
            <MenuItem value="quality_desc">Quality (High to Low)</MenuItem>
            <MenuItem value="quality_asc">Quality (Low to High)</MenuItem>
            <MenuItem value="quality_delta_desc">Quality Δ (High to Low)</MenuItem>
            <MenuItem value="quality_delta_asc">Quality Δ (Low to High)</MenuItem>
            <MenuItem value="size_desc">Cluster Size (Large to Small)</MenuItem>
            <MenuItem value="size_asc">Cluster Size (Small to Large)</MenuItem>
          </Select>
        </FormControl>
      </Box>
    </Paper>
  );
}

export default MetricsControlPanel;