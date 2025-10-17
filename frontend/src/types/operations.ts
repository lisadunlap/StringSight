// Data operation types for tracking data provenance

export interface BaseOperation {
  id: string;
  timestamp: number;
}

export interface FilterOperation extends BaseOperation {
  type: 'filter';
  column: string;
  values: string[];
  negated: boolean;
}

export interface CustomCodeOperation extends BaseOperation {
  type: 'custom';
  code: string;
}

export interface SortOperation extends BaseOperation {
  type: 'sort';
  column: string;
  direction: 'asc' | 'desc';
}

export type DataOperation = FilterOperation | CustomCodeOperation | SortOperation;

export interface OperationChain {
  operations: DataOperation[];
}

// Helper functions for operation management
export const createFilterOperation = (
  column: string, 
  values: string[], 
  negated: boolean = false
): FilterOperation => ({
  id: `filter_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
  type: 'filter',
  column,
  values,
  negated,
  timestamp: Date.now()
});

export const createCustomCodeOperation = (code: string): CustomCodeOperation => ({
  id: `custom_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
  type: 'custom',
  code,
  timestamp: Date.now()
});

export const createSortOperation = (
  column: string, 
  direction: 'asc' | 'desc'
): SortOperation => ({
  id: `sort_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
  type: 'sort',
  column,
  direction,
  timestamp: Date.now()
});

// Operation display helpers
export const getOperationDescription = (operation: DataOperation): string => {
  switch (operation.type) {
    case 'filter':
      return `${operation.column}: ${operation.negated ? 'NOT ' : ''}${operation.values.join(', ')}`;
    case 'custom':
      return operation.code.length > 50 ? `${operation.code.slice(0, 50)}...` : operation.code;
    case 'sort':
      return `Sort by ${operation.column} (${operation.direction})`;
    default:
      return 'Unknown operation';
  }
};
