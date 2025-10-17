import React from 'react';
import { Box, TextField, Chip, Stack, Autocomplete, Button, FormControlLabel, Switch, Divider } from '@mui/material';

interface Filter {
  column: string;
  values: string[];
  negated: boolean;
}

interface FilterBarProps {
  // Search functionality
  searchValue?: string;
  onSearchChange?: (value: string) => void;
  searchPlaceholder?: string;
  
  // Dynamic column-based filtering (for data tab)
  categoricalColumns?: string[];
  pendingColumn?: string | null;
  pendingValues?: string[];
  pendingNegated?: boolean;
  onPendingColumnChange?: (column: string | null) => void;
  onPendingValuesChange?: (values: string[]) => void;
  onPendingNegatedChange?: (negated: boolean) => void;
  onAddFilter?: () => void;
  filters?: Filter[];
  onRemoveFilter?: (index: number) => void;
  uniqueValuesFor?: (column: string) => string[];
  
  // Fixed column filtering (for properties tab)
  fixedFilters?: {
    label: string;
    options: string[];
    value: string[];
    onChange: (values: string[]) => void;
  }[];
  
  // Results count
  resultCount?: number;
  resultLabel?: string;
  
  // Additional controls (for data tab)
  showGroupBy?: boolean;
  groupByOptions?: string[];
  groupByValue?: string | null;
  onGroupByChange?: (value: string | null) => void;
  
  showCustomCode?: boolean;
  customCodeValue?: string;
  onCustomCodeChange?: (value: string) => void;
  onCustomCodeRun?: () => void;
  onReset?: () => void;
  customCodeError?: string | null;
}

export default function FilterBar({
  searchValue = '',
  onSearchChange,
  searchPlaceholder = 'Search',
  categoricalColumns = [],
  pendingColumn,
  pendingValues = [],
  pendingNegated = false,
  onPendingColumnChange,
  onPendingValuesChange,
  onPendingNegatedChange,
  onAddFilter,
  filters = [],
  onRemoveFilter,
  uniqueValuesFor,
  fixedFilters = [],
  resultCount,
  resultLabel = 'results',
  showGroupBy = false,
  groupByOptions = [],
  groupByValue,
  onGroupByChange,
  showCustomCode = false,
  customCodeValue = '',
  onCustomCodeChange,
  onCustomCodeRun,
  onReset,
  customCodeError
}: FilterBarProps) {
  
  return (
    <Box sx={{ 
      p: 1.5, 
      border: '1px solid #E5E7EB', 
      borderRadius: 2, 
      background: '#FFFFFF', 
      mb: 1,
      width: '100%',
      maxWidth: '100vw',
      overflow: 'hidden'
    }}>
      <Stack direction={{ xs: 'column', lg: 'row' }} spacing={1} alignItems={{ xs: 'stretch', lg: 'center' }}>
        
        {/* Search and Filters Section */}
        <Stack direction="row" spacing={1} alignItems="center" sx={{ flexWrap: 'wrap', flex: 1, minWidth: 0 }}>
          
          {/* Search Bar */}
          {onSearchChange && (
            <TextField 
              size="small" 
              label={searchPlaceholder}
              value={searchValue} 
              onChange={(e) => onSearchChange(e.target.value)} 
              sx={{ minWidth: 200 }} 
            />
          )}
          
          {/* Dynamic Column-based Filters (Data Tab) */}
          {categoricalColumns.length > 0 && (
            <>
              <Autocomplete
                size="small"
                sx={{ minWidth: 180, maxWidth: 220, flex: '0 1 auto' }}
                options={categoricalColumns}
                value={pendingColumn}
                onChange={(_, v) => onPendingColumnChange?.(v)}
                renderInput={(params) => <TextField {...params} label="Add filter (column)" />}
              />
              {pendingColumn && (
                <Autocomplete
                  multiple 
                  size="small"
                  sx={{ minWidth: 200, maxWidth: 300, flex: '0 1 auto' }}
                  options={uniqueValuesFor?.(pendingColumn) || []}
                  value={pendingValues}
                  onChange={(_, v) => onPendingValuesChange?.(v)}
                  renderTags={(value, getTagProps) => value.map((option, index) => (
                    <Chip {...getTagProps({ index })} key={option} label={option} />
                  ))}
                  renderInput={(params) => <TextField {...params} label="Values" />}
                />
              )}
              {pendingColumn && (
                <FormControlLabel 
                  control={<Switch checked={pendingNegated} onChange={(_, c) => onPendingNegatedChange?.(c)} />} 
                  label="NOT" 
                />
              )}
              <Button
                variant="outlined"
                disabled={!pendingColumn || pendingValues.length === 0}
                onClick={onAddFilter}
              >
                Add Filter
              </Button>
              {filters.map((f, i) => (
                <Chip 
                  key={`${f.column}-${i}`} 
                  label={`${f.column}: ${f.negated ? 'NOT ' : ''}${f.values.join(', ')}`} 
                  onDelete={() => onRemoveFilter?.(i)} 
                />
              ))}
            </>
          )}
          
          {/* Fixed Column Filters (Properties Tab) */}
          {fixedFilters.map((filter, index) => (
            <Autocomplete
              key={index}
              multiple
              options={filter.options}
              value={filter.value}
              onChange={(_, v) => filter.onChange(v)}
              renderInput={(params) => <TextField {...params} size="small" label={filter.label} />}
              sx={{ minWidth: 180 }}
            />
          ))}
          

        </Stack>

        {/* Group By Section */}
        {showGroupBy && (
          <>
            <Divider orientation="vertical" flexItem sx={{ display: { xs: 'none', md: 'block' } }} />
            <Stack direction="row" spacing={1} alignItems="center">
              <Autocomplete
                size="small"
                sx={{ minWidth: 160, maxWidth: 220, flex: '0 1 auto' }}
                options={groupByOptions}
                value={groupByValue}
                onChange={(_, v) => onGroupByChange?.(v)}
                renderInput={(params) => <TextField {...params} label="Group by" />}
              />
            </Stack>
          </>
        )}

        {/* Custom Code Section */}
        {showCustomCode && (
          <>
            <Divider orientation="vertical" flexItem sx={{ display: { xs: 'none', md: 'block' } }} />
            <Stack direction="row" spacing={1} alignItems="center" sx={{ flex: 1 }}>
              <TextField 
                size="small" 
                fullWidth 
                placeholder={`pandas expression (returns DataFrame), e.g., df.query("model==\"gpt-4\"")`} 
                value={customCodeValue} 
                onChange={(e) => onCustomCodeChange?.(e.target.value)} 
              />
              <Button variant="outlined" onClick={onCustomCodeRun}>Run</Button>
              <Button variant="text" onClick={onReset}>Reset</Button>
            </Stack>
          </>
        )}
      </Stack>
      
      {/* Custom Code Error */}
      {customCodeError && (
        <Box sx={{ color: '#b91c1c', mt: 1, fontSize: 12 }}>
          {customCodeError}
        </Box>
      )}
    </Box>
  );
}
