import React from 'react';
import { Box, Chip, Typography, Stack } from '@mui/material';
import type { DataOperation } from '../types/operations';
import { getOperationDescription } from '../types/operations';

interface FilterSummaryProps {
  operations: DataOperation[];
  onRemoveOperation: (operationId: string) => void;
}

const FilterSummary: React.FC<FilterSummaryProps> = ({
  operations,
  onRemoveOperation
}) => {
  if (operations.length === 0) {
    return null;
  }

  const getOperationColor = (type: string) => {
    switch (type) {
      case 'filter': return { bg: '#DBEAFE', text: '#1E40AF' };
      case 'custom': return { bg: '#FEF3C7', text: '#92400E' };
      case 'sort': return { bg: '#D1FAE5', text: '#065F46' };
      default: return { bg: '#F3F4F6', text: '#374151' };
    }
  };

  const getOperationTypeLabel = (type: string) => {
    switch (type) {
      case 'filter': return 'Filter';
      case 'custom': return 'Pandas';
      case 'sort': return 'Sort';
      default: return 'Unknown';
    }
  };

  return (
    <Box sx={{
      mb: 2,
      p: 2,
      backgroundColor: '#EBF5FF',
      border: '1px solid #3B82F6',
      borderRadius: 2,
      fontSize: 14
    }}>
      <Stack spacing={1.5}>
        <Typography variant="subtitle2" sx={{ 
          color: '#1E40AF', 
          fontWeight: 600, 
          fontSize: 13, 
          textTransform: 'uppercase',
          letterSpacing: 0.5
        }}>
          Operation Chain ({operations.length})
        </Typography>

        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          {operations.map((operation, index) => {
            const colors = getOperationColor(operation.type);
            return (
              <Chip
                key={operation.id}
                label={`${index + 1}. ${getOperationTypeLabel(operation.type)}: ${getOperationDescription(operation)}`}
                onDelete={() => onRemoveOperation(operation.id)}
                size="small"
                sx={{
                  backgroundColor: colors.bg,
                  color: colors.text,
                  fontFamily: operation.type === 'custom' ? 'monospace' : 'inherit',
                  '& .MuiChip-deleteIcon': {
                    color: colors.text,
                    '&:hover': {
                      opacity: 0.8
                    }
                  }
                }}
              />
            );
          })}
        </Stack>
      </Stack>
    </Box>
  );
};

export default FilterSummary;
