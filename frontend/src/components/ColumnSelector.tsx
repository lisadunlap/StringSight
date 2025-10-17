import React, { useState, useEffect } from 'react';
import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Typography,
  Button,
  Alert,
  Divider,
  FormHelperText,
  Checkbox,
  ListItemText
} from '@mui/material';

export interface ColumnMapping {
  promptCol: string;
  responseCols: string[];
  modelCols: string[];
  scoreCols: string[];
  method: 'single_model' | 'side_by_side';
}

interface ColumnSelectorProps {
  columns: string[];
  onMappingChange: (mapping: ColumnMapping) => void;
  onValidationChange: (isValid: boolean, errors: string[]) => void;
  autoDetectedMapping?: ColumnMapping;
}

export function ColumnSelector({ 
  columns, 
  onMappingChange, 
  onValidationChange,
  autoDetectedMapping 
}: ColumnSelectorProps) {
  const [mapping, setMapping] = useState<ColumnMapping>({
    promptCol: '',
    responseCols: [],
    modelCols: [],
    scoreCols: [],
    method: 'single_model'
  });

  const [errors, setErrors] = useState<string[]>([]);
  const [useAutoDetection, setUseAutoDetection] = useState(true);

  // Initialize with auto-detected mapping if available
  useEffect(() => {
    if (autoDetectedMapping && useAutoDetection) {
      setMapping(autoDetectedMapping);
    }
  }, [autoDetectedMapping, useAutoDetection]);

  // Validate mapping and notify parent
  useEffect(() => {
    const newErrors = validateMapping(mapping);
    setErrors(newErrors);
    const isValid = newErrors.length === 0;
    onValidationChange(isValid, newErrors);
    // Do not auto-apply mapping; wait for explicit user action (Done)
  }, [mapping, onValidationChange]);

  const validateMapping = (currentMapping: ColumnMapping): string[] => {
    const errors: string[] = [];
    
    if (!currentMapping.promptCol) {
      errors.push('Please select a prompt column');
    }
    
    if (currentMapping.responseCols.length === 0) {
      errors.push('Please select at least one response column');
    }
    
    if (currentMapping.method === 'side_by_side') {
      if (currentMapping.responseCols.length !== 2) {
        errors.push('Side-by-side comparison requires exactly 2 response columns');
      }
      if (currentMapping.modelCols.length > 0 && currentMapping.modelCols.length !== 2) {
        errors.push('Side-by-side comparison requires exactly 2 model columns (or none)');
      }
    } else {
      if (currentMapping.responseCols.length > 1) {
        errors.push('Single model format should have exactly 1 response column');
      }
      if (currentMapping.modelCols.length > 1) {
        errors.push('Single model format should have at most 1 model column');
      }
    }
    
    return errors;
  };

  const handlePromptChange = (event: any) => {
    setMapping(prev => ({ ...prev, promptCol: event.target.value }));
    setUseAutoDetection(false);
  };

  const handleMethodChange = (event: any) => {
    const newMethod = event.target.value as 'single_model' | 'side_by_side';
    setMapping(prev => ({ 
      ...prev, 
      method: newMethod,
      // Reset response and model cols when method changes
      responseCols: newMethod === 'side_by_side' ? prev.responseCols.slice(0, 2) : prev.responseCols.slice(0, 1),
      modelCols: newMethod === 'side_by_side' ? prev.modelCols.slice(0, 2) : prev.modelCols.slice(0, 1)
    }));
    setUseAutoDetection(false);
  };

  const handleResponseChange = (event: any) => {
    const value = typeof event.target.value === 'string' ? [event.target.value] : event.target.value;
    const maxCols = mapping.method === 'side_by_side' ? 2 : 1;
    setMapping(prev => ({ ...prev, responseCols: value.slice(0, maxCols) }));
    setUseAutoDetection(false);
  };

  const handleModelChange = (event: any) => {
    const value = typeof event.target.value === 'string' ? [event.target.value] : event.target.value;
    const maxCols = mapping.method === 'side_by_side' ? 2 : 1;
    setMapping(prev => ({ ...prev, modelCols: value.slice(0, maxCols) }));
    setUseAutoDetection(false);
  };

  const handleScoreChange = (event: any) => {
    const value = typeof event.target.value === 'string' ? [event.target.value] : event.target.value;
    setMapping(prev => ({ ...prev, scoreCols: value }));
    setUseAutoDetection(false);
  };

  const resetToAutoDetection = () => {
    if (autoDetectedMapping) {
      setMapping(autoDetectedMapping);
      setUseAutoDetection(true);
    }
  };

  const getAvailableColumns = (excludeColumns: string[] = []) => {
    return columns.filter(col => !excludeColumns.includes(col));
  };

  const renderChips = (selectedCols: string[], color: 'primary' | 'secondary' = 'primary', onDelete?: (col: string) => void) => {
    return selectedCols.map(col => (
      <Chip 
        key={col} 
        label={col} 
        size="small" 
        color={color}
        onDelete={onDelete ? () => onDelete(col) : undefined}
        sx={{ mr: 0.5, mb: 0.5 }}
      />
    ));
  };

  return (
    <Box sx={{ p: 3, border: '1px solid #e0e0e0', borderRadius: 2, mb: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Column Mapping</Typography>
        {autoDetectedMapping && !useAutoDetection && (
          <Button 
            size="small" 
            variant="outlined" 
            onClick={resetToAutoDetection}
          >
            Reset to Auto-Detection
          </Button>
        )}
      </Box>

      {autoDetectedMapping && useAutoDetection && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Auto-detected column mapping. You can modify the selections below if needed.
        </Alert>
      )}

      {/* Method Selection */}
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Comparison Method</InputLabel>
        <Select
          value={mapping.method}
          label="Comparison Method"
          onChange={handleMethodChange}
        >
          <MenuItem value="single_model">Single Model Analysis</MenuItem>
          <MenuItem value="side_by_side">Side-by-Side Comparison</MenuItem>
        </Select>
        <FormHelperText>
          {mapping.method === 'single_model' 
            ? 'Analyze responses from individual models'
            : 'Compare responses between two models'
          }
        </FormHelperText>
      </FormControl>

      <Divider sx={{ my: 2 }} />

      {/* Prompt Column */}
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Prompt Column *</InputLabel>
        <Select
          value={mapping.promptCol}
          label="Prompt Column *"
          onChange={handlePromptChange}
        >
          {columns.map(col => (
            <MenuItem key={col} value={col}>{col}</MenuItem>
          ))}
        </Select>
        <FormHelperText>Column containing the input prompts/questions</FormHelperText>
      </FormControl>

      {/* Response Columns */}
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>
          Response Column{mapping.method === 'side_by_side' ? 's' : ''} *
        </InputLabel>
        <Select
          multiple={mapping.method === 'side_by_side'}
          value={mapping.method === 'side_by_side' ? mapping.responseCols : mapping.responseCols[0] || ''}
          label={`Response Column${mapping.method === 'side_by_side' ? 's' : ''} *`}
          onChange={handleResponseChange}
          renderValue={(selected) => (
            <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
              {mapping.method === 'side_by_side' 
                ? renderChips(Array.isArray(selected) ? selected : [selected])
                : selected
              }
            </Box>
          )}
        >
          {getAvailableColumns([mapping.promptCol]).map(col => (
            <MenuItem key={col} value={col}>{col}</MenuItem>
          ))}
        </Select>
        <FormHelperText>
          {mapping.method === 'side_by_side' 
            ? 'Select exactly 2 columns containing model responses (Model A, Model B)'
            : 'Column containing the model response'
          }
        </FormHelperText>
      </FormControl>

      {/* Model Columns */}
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>
          Model Column{mapping.method === 'side_by_side' ? 's' : ''} (Optional)
        </InputLabel>
        <Select
          multiple={mapping.method === 'side_by_side'}
          value={mapping.method === 'side_by_side' ? mapping.modelCols : mapping.modelCols[0] || ''}
          label={`Model Column${mapping.method === 'side_by_side' ? 's' : ''} (Optional)`}
          onChange={handleModelChange}
          renderValue={(selected) => (
            <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
              {mapping.method === 'side_by_side' 
                ? renderChips(Array.isArray(selected) ? selected : [selected], 'secondary')
                : selected
              }
            </Box>
          )}
        >
          {getAvailableColumns([mapping.promptCol, ...mapping.responseCols]).map(col => (
            <MenuItem key={col} value={col}>{col}</MenuItem>
          ))}
        </Select>
        <FormHelperText>
          {mapping.method === 'side_by_side' 
            ? 'Optional: Select 2 columns containing model names (will use "Model A", "Model B" if not specified)'
            : 'Optional: Column containing model names (will use "unknown" if not specified)'
          }
        </FormHelperText>
      </FormControl>

      {/* Score Columns */}
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Score Columns (Optional)</InputLabel>
        <Select
          multiple
          value={mapping.scoreCols}
          label="Score Columns (Optional)"
          onChange={handleScoreChange}
          renderValue={(selected) => (
            <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
              {renderChips(selected, 'secondary', (col) => {
                setMapping(prev => ({ ...prev, scoreCols: prev.scoreCols.filter(c => c !== col) }));
                setUseAutoDetection(false);
              })}
            </Box>
          )}
          MenuProps={{
            PaperProps: { sx: { maxHeight: 360 } }
          }}
        >
          {getAvailableColumns([mapping.promptCol, ...mapping.responseCols, ...mapping.modelCols]).map(col => (
            <MenuItem key={col} value={col}>
              <Checkbox checked={mapping.scoreCols.indexOf(col) > -1} />
              <ListItemText primary={col} />
            </MenuItem>
          ))}
        </Select>
        <FormHelperText>
          Select one or more metrics. Checked items are included. Click Done to apply.
        </FormHelperText>
      </FormControl>

      {/* Validation Errors */}
      {errors.length > 0 && (
        <Alert severity="error" sx={{ mt: 2 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>Please fix the following issues:</Typography>
          <ul style={{ margin: 0, paddingLeft: 20 }}>
            {errors.map((error, index) => (
              <li key={index}>{error}</li>
            ))}
          </ul>
        </Alert>
      )}

      {/* Summary */}
      {errors.length === 0 && mapping.promptCol && mapping.responseCols.length > 0 && (
        <Alert severity="success" sx={{ mt: 2 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>Ready to process:</Typography>
          <Typography variant="body2">
            <strong>Method:</strong> {mapping.method === 'single_model' ? 'Single Model' : 'Side-by-Side'}<br/>
            <strong>Prompt:</strong> {mapping.promptCol}<br/>
            <strong>Response{mapping.method === 'side_by_side' ? 's' : ''}:</strong> {mapping.responseCols.join(', ')}<br/>
            {mapping.modelCols.length > 0 && (
              <>
                <strong>Model{mapping.method === 'side_by_side' ? 's' : ''}:</strong> {mapping.modelCols.join(', ')}<br/>
              </>
            )}
            {mapping.scoreCols.length > 0 && (
              <>
                <strong>Scores:</strong> {mapping.scoreCols.join(', ')}
              </>
            )}
          </Typography>
        </Alert>
      )}

      {/* Persistent footer action */}
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
        <Button
          variant="contained"
          color="primary"
          onClick={() => onMappingChange(mapping)}
          disabled={
            !mapping.promptCol ||
            mapping.responseCols.length === 0 ||
            (mapping.method === 'side_by_side' && mapping.responseCols.length !== 2) ||
            (mapping.method === 'side_by_side' && mapping.modelCols.length > 0 && mapping.modelCols.length !== 2) ||
            (mapping.method === 'single_model' && mapping.responseCols.length !== 1) ||
            (mapping.method === 'single_model' && mapping.modelCols.length > 1) ||
            errors.length > 0
          }
        >
          Done
        </Button>
      </Box>
    </Box>
  );
}
