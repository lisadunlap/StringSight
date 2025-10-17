/**
 * SignificanceBadge - Significance indicators for frequency/quality.
 * 
 * Shows compact round pills:
 * - F: Frequency delta significant (pink theme)
 * - Q: Quality delta significant (blue theme)
 */

import React from 'react';
import { 
  Chip,
  useTheme
} from '@mui/material';

interface SignificanceBadgeProps {
  type: 'frequency' | 'quality';
  size?: 'small' | 'medium';
}

export function SignificanceBadge({
  type,
  size = 'small'
}: SignificanceBadgeProps) {
  const theme = useTheme();
  
  const config = type === 'frequency' 
    ? {
        label: 'F',
        color: '#cc6699', // Pink theme
        backgroundColor: '#fce4ec'
      }
    : {
        label: 'Q', 
        color: '#007bff', // Blue theme
        backgroundColor: '#e3f2fd'
      };

  return (
    <Chip
      label={config.label}
      size={size}
      sx={{
        height: 20,
        minWidth: 20,
        fontSize: '0.7rem',
        fontWeight: 600,
        color: config.color,
        backgroundColor: config.backgroundColor,
        border: `1px solid ${config.color}`,
        '& .MuiChip-label': {
          px: 0.5
        }
      }}
    />
  );
}

export default SignificanceBadge;
