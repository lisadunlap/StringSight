import React from 'react';
import { Box, Typography, Stack, Chip, Divider } from '@mui/material';

interface PropertyTraceHeaderProps {
  selectedRow: any;
  selectedProperty: any;
  method: 'single_model' | 'side_by_side' | 'unknown';
  evidenceTargetModel?: string;
}

export default function PropertyTraceHeader({
  selectedRow,
  selectedProperty,
  method,
  evidenceTargetModel
}: PropertyTraceHeaderProps) {
  

  // Format property metadata
  const formatPropertyValue = (key: string, value: any): React.ReactNode => {
    if (value === null || value === undefined || value === '') return null;
    
    // Handle arrays (like evidence)
    if (Array.isArray(value)) {
      if (value.length === 0) return null;
      
      // For evidence arrays, show full content
      const isEvidence = key.toLowerCase().includes('evidence');
      const displayValue = isEvidence ? value.join(', ') : (
        value.length > 3 
          ? `${value.slice(0, 3).join(', ')}... (+${value.length - 3} more)`
          : value.join(', ')
      );
      
      return (
        <Box key={key} sx={{ mb: 0.5 }}>
          <Typography variant="caption" sx={{ fontWeight: 600, color: '#64748B' }}>
            {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
          </Typography>
          <Typography variant="body2" sx={{ 
            ml: 1, 
            fontStyle: 'italic',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word'
          }}>
            {displayValue}
          </Typography>
        </Box>
      );
    }
    
    // Handle booleans
    if (typeof value === 'boolean') {
      return (
        <Chip 
          key={key}
          label={`${key.replace(/_/g, ' ')}: ${value ? 'Yes' : 'No'}`}
          size="small"
          sx={{ 
            backgroundColor: value ? '#DCFCE7' : '#FEF2F2', 
            color: value ? '#166534' : '#991B1B',
            fontSize: 11
          }}
        />
      );
    }
    
    // Handle strings and numbers
    const stringValue = String(value);
    
    // Important fields that should show in full (no truncation)
    const importantFields = ['reason', 'evidence', 'explanation', 'justification', 'rationale'];
    const isImportant = importantFields.some(field => key.toLowerCase().includes(field.toLowerCase()));
    
    if (stringValue.length > 100 && !isImportant) {
      return (
        <Box key={key} sx={{ mb: 0.5 }}>
          <Typography variant="caption" sx={{ fontWeight: 600, color: '#64748B' }}>
            {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
          </Typography>
          <Typography variant="body2" sx={{ ml: 1 }}>
            {stringValue.slice(0, 100)}...
          </Typography>
        </Box>
      );
    }
    
    // For important fields or short text, show in full
    if (stringValue.length > 50) {
      return (
        <Box key={key} sx={{ mb: 0.5 }}>
          <Typography variant="caption" sx={{ fontWeight: 600, color: '#64748B' }}>
            {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
          </Typography>
          <Typography variant="body2" sx={{ 
            ml: 1, 
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            maxHeight: isImportant ? 'none' : '120px',
            overflow: isImportant ? 'visible' : 'auto'
          }}>
            {stringValue}
          </Typography>
        </Box>
      );
    }
    
    return (
      <Chip 
        key={key}
        label={`${key.replace(/_/g, ' ')}: ${stringValue}`}
        size="small"
        variant="outlined"
        sx={{ fontSize: 11 }}
      />
    );
  };

  // Get conversation scores
  const getConversationScores = () => {
    let entries: [string, any][] = [];
    if (method === 'single_model') {
      const scores = selectedRow?.score || null;
      entries = scores && typeof scores === 'object' ? Object.entries(scores as Record<string, number>) : [];
    } else if (method === 'side_by_side') {
      const isA = evidenceTargetModel && String(selectedRow?.model_a || '') === String(evidenceTargetModel);
      const isB = evidenceTargetModel && String(selectedRow?.model_b || '') === String(evidenceTargetModel);
      const sa = (selectedRow as any)?.score_a || {};
      const sb = (selectedRow as any)?.score_b || {};
      const chosen = isA ? sa : isB ? sb : sa; // default A
      entries = Object.entries(chosen);
    }
    return entries;
  };

  // Get property metadata (excluding certain fields)
  const getPropertyMetadata = () => {
    if (!selectedProperty) return [];
    
    const excludeKeys = new Set([
      'question_id', 'model', 'model_response', 'property_description', 
      '__index', 'row_index', 'id', 'meta', 'raw_response'
    ]);
    
    return Object.entries(selectedProperty)
      .filter(([key]) => !excludeKeys.has(key))
      .filter(([, value]) => value !== null && value !== undefined && value !== '');
  };

  const conversationScores = getConversationScores();
  const propertyMetadata = getPropertyMetadata();

  return (
    <Box sx={{ mb: 2 }}>
      {/* Original Conversation Header */}
      <Box sx={{ mb: 1.5 }}>
        <Typography variant="subtitle2" sx={{ color: '#334155', mb: 0.5 }}>
          📄 Original Conversation
        </Typography>
        <Typography variant="body2" sx={{ color: '#64748B', mb: 1 }}>
          {method === 'single_model' 
            ? String(selectedRow?.model || 'Unknown Model')
            : `${String(selectedRow?.model_a || 'Model A')} vs ${String(selectedRow?.model_b || 'Model B')}`
          }
        </Typography>
        
        {/* Conversation Scores */}
        {conversationScores.length > 0 && (
          <Stack direction="row" spacing={0.5} sx={{ flexWrap: 'wrap' }}>
            {conversationScores.map(([k, v]) => (
              <Box key={k} sx={{ 
                px: 0.75, py: 0.25, 
                border: '1px solid #E5E7EB', 
                borderRadius: 9999, 
                fontSize: 12, 
                color: '#334155', 
                background: '#F8FAFC' 
              }}>
                {k}: {typeof v === 'number' ? v.toFixed(2) : String(v)}
              </Box>
            ))}
          </Stack>
        )}
      </Box>

      <Divider sx={{ my: 1 }} />

      {/* Property Information */}
      <Box>
        <Typography variant="subtitle2" sx={{ color: '#334155', mb: 0.5 }}>
          🏷️ Property Information
        </Typography>
        
        {/* Property Description */}
        {selectedProperty?.property_description && (
          <Typography variant="body2" sx={{ mb: 1, fontStyle: 'italic', color: '#64748B' }}>
            "{selectedProperty.property_description}"
          </Typography>
        )}
        
        {/* Property Metadata */}
        {propertyMetadata.length > 0 && (
          <Stack spacing={0.5} sx={{ mb: 1 }}>
            {propertyMetadata.slice(0, 8).map(([key, value]) => formatPropertyValue(key, value))}
            {propertyMetadata.length > 8 && (
              <Typography variant="caption" sx={{ color: '#64748B', fontStyle: 'italic' }}>
                ... and {propertyMetadata.length - 8} more properties
              </Typography>
            )}
          </Stack>
        )}
      </Box>
    </Box>
  );
}
