import React from 'react';
import { Box, Typography, Card, CardContent, Chip, Stack } from '@mui/material';
import ResponseContent from './ResponseContent';
import type { HighlightRange } from './ResponseContent';

interface ModelResponseCardProps {
  modelName: string;
  response: string;
  highlightedRanges?: HighlightRange[];
  metadata?: Record<string, any>;
  variant?: 'default' | 'compact' | 'expanded';
  showModelName?: boolean;
}

/**
 * Component for displaying a single model's response with optional highlighting.
 * Handles both single model and side-by-side model displays.
 */
export default function ModelResponseCard({ 
  modelName, 
  response, 
  highlightedRanges = [], 
  metadata = {},
  variant = 'default',
  showModelName = true
}: ModelResponseCardProps) {
  
  const isCompact = variant === 'compact';
  const isExpanded = variant === 'expanded';
  
  // Extract score if available in metadata
  const score = metadata.score || metadata.scores;
  const hasScore = score && typeof score === 'object' && Object.keys(score).length > 0;
  
  return (
    <Card 
      elevation={isCompact ? 1 : 2}
      sx={{ 
        mb: isCompact ? 1 : 2,
        border: '1px solid #E5E7EB',
        '&:hover': {
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
        }
      }}
    >
      <CardContent sx={{ p: isCompact ? 2 : 3 }}>
        {/* Model name header */}
        {showModelName && (
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography 
              variant={isCompact ? 'subtitle2' : 'h6'} 
              sx={{ 
                fontWeight: 600,
                color: '#1976d2',
                fontFamily: 'monospace'
              }}
            >
              {modelName}
            </Typography>
            
            {/* Score display */}
            {hasScore && (
              <Stack direction="row" spacing={1}>
                {Object.entries(score).map(([key, value]) => (
                  <Chip
                    key={key}
                    label={`${key}: ${typeof value === 'number' ? value.toFixed(2) : value}`}
                    size="small"
                    variant="outlined"
                    sx={{ fontSize: '0.75rem' }}
                  />
                ))}
              </Stack>
            )}
          </Box>
        )}
        
        {/* Response content */}
        <Box 
          sx={{ 
            backgroundColor: '#fafafa',
            border: '1px solid #e0e0e0',
            borderRadius: 1,
            p: isCompact ? 1.5 : 2,
            maxHeight: isExpanded ? 'none' : '400px',
            overflow: isExpanded ? 'visible' : 'auto'
          }}
        >
          <ResponseContent 
            content={response || '(No response)'}
            highlightedRanges={highlightedRanges}
          />
        </Box>
        
        {/* Additional metadata */}
        {isExpanded && metadata && Object.keys(metadata).length > 0 && (
          <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid #e0e0e0' }}>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
              Metadata:
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap">
              {Object.entries(metadata)
                .filter(([key]) => key !== 'score' && key !== 'scores') // Already shown above
                .map(([key, value]) => (
                  <Chip
                    key={key}
                    label={`${key}: ${String(value)}`}
                    size="small"
                    variant="outlined"
                    sx={{ fontSize: '0.7rem', mb: 0.5 }}
                  />
                ))}
            </Stack>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Utility component for side-by-side model comparison
 */
interface SideBySideResponseProps {
  modelA: string;
  modelB: string;
  responseA: string;
  responseB: string;
  highlightedRangesA?: HighlightRange[];
  highlightedRangesB?: HighlightRange[];
  metadataA?: Record<string, any>;
  metadataB?: Record<string, any>;
  variant?: 'default' | 'compact' | 'expanded';
}

export function SideBySideResponse({
  modelA,
  modelB,
  responseA,
  responseB,
  highlightedRangesA = [],
  highlightedRangesB = [],
  metadataA = {},
  metadataB = {},
  variant = 'default'
}: SideBySideResponseProps) {
  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
      <ModelResponseCard
        modelName={modelA}
        response={responseA}
        highlightedRanges={highlightedRangesA}
        metadata={metadataA}
        variant={variant}
      />
      <ModelResponseCard
        modelName={modelB}
        response={responseB}
        highlightedRanges={highlightedRangesB}
        metadata={metadataB}
        variant={variant}
      />
    </Box>
  );
}
