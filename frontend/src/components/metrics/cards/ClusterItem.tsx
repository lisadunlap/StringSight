/**
 * ClusterItem - Rich cluster display within model cards.
 * 
 * Features:
 * - Color-coded left border based on proportion delta
 * - Full cluster label with hover truncation
 * - Frequency stats (percentage, delta, count)
 * - Significance badges (F/Q)
 * - Quality metrics with delta indicators
 * - Metadata tags as chips
 */

import React from 'react';
import { 
  Box, 
  Typography, 
  Stack,
  useTheme,
  Tooltip,
  Collapse,
  IconButton
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';
const Plot = createPlotlyComponent(Plotly);
import { SignificanceBadge } from './SignificanceBadge';
import { TagChips } from './TagChips';
import type { ModelClusterRow } from '../../../types/metrics';

interface ClusterItemProps {
  cluster: ModelClusterRow;
  rank: number;
  qualityMetric: string;
}

export function ClusterItem({
  cluster,
  rank,
  qualityMetric
}: ClusterItemProps) {
  const theme = useTheme();
  const [expanded, setExpanded] = React.useState(false);
  
  const proportionDelta = cluster.proportion_delta || 0;
  const isPositiveDelta = proportionDelta > 0;

  // Color coding based on metadata tags
  const getColorFromMetadata = () => {
    if (!cluster.metadata) return theme.palette.grey[300];

    // Convert metadata object to searchable string
    const metadataStr = JSON.stringify(cluster.metadata).toLowerCase();
    if (metadataStr.includes('negative') || metadataStr.includes('critical')) {
      return theme.palette.error.main;
    }
    if (metadataStr.includes('positive')) {
      return theme.palette.success.main;
    }
    return theme.palette.grey[300];
  };

  const borderColor = getColorFromMetadata();
  
  // Quality metrics - extract all quality delta metrics
  const qualityDeltaMetrics: Record<string, number> = {};
  Object.keys(cluster).forEach(key => {
    if (key.startsWith('quality_delta_') && !key.endsWith('_significant')) {
      const metricName = key.replace('quality_delta_', '');
      const value = cluster[key as keyof ModelClusterRow];
      if (typeof value === 'number') {
        qualityDeltaMetrics[metricName] = value;
      }
    }
  });
  
  // Current selected quality metric
  const qualityKey = `quality_${qualityMetric}`;
  const qualityDeltaKey = `quality_delta_${qualityMetric}`;
  const qualityValue = cluster[qualityKey as keyof ModelClusterRow] as number;
  const qualityDelta = cluster[qualityDeltaKey as keyof ModelClusterRow] as number;
  
  // Significance flags
  const freqSignificant = cluster.proportion_delta_significant === true;
  const qualitySignificant = cluster[`${qualityDeltaKey}_significant` as keyof ModelClusterRow] === true;
  
  // Display full cluster name and allow wrapping
  const displayLabel = cluster.cluster;

  return (
    <Box
      sx={{
        borderLeft: 4,
        borderLeftColor: borderColor,
        pl: 2,
        py: 1,
        backgroundColor: theme.palette.mode === 'dark' ? 'grey.900' : 'grey.50',
        borderRadius: 1,
        '&:hover': {
          backgroundColor: theme.palette.mode === 'dark' ? 'grey.800' : 'grey.100'
        }
      }}
    >
      <Stack spacing={1}>
        {/* Cluster Label with Expand Button */}
        <Stack direction="row" spacing={1} sx={{ alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <Tooltip title={cluster.cluster} placement="top-start">
            <Typography 
              variant="body2" 
              sx={{ 
                fontWeight: 500,
                lineHeight: 1.3,
                cursor: 'help',
                flex: 1
              }}
            >
              #{rank} {displayLabel}
            </Typography>
          </Tooltip>
          {Object.keys(qualityDeltaMetrics).length > 0 && (
            <IconButton
              size="small"
              onClick={() => setExpanded(!expanded)}
              sx={{
                transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s'
              }}
            >
              <ExpandMoreIcon fontSize="small" />
            </IconButton>
          )}
        </Stack>
        
        {/* Frequency Stats and Badges Row */}
        <Stack direction="row" spacing={0.75} sx={{ alignItems: 'center', flexWrap: 'wrap', rowGap: 0.5 }}>
          <Typography variant="caption" color="text.secondary" sx={{ mr: 0.5 }}>
            {((cluster.proportion || 0) * 100).toFixed(1)}%
            {proportionDelta !== 0 && (
              <Box component="span" sx={{
                color: isPositiveDelta ? 'success.main' : 'error.main',
                fontWeight: 500
              }}>
                {' '}({isPositiveDelta ? '+' : ''}{(proportionDelta * 100).toFixed(1)}%)
              </Box>
            )}
            {' '}frequency ({cluster.size || 0} out of {Math.round((cluster.size || 0) / ((cluster.proportion || 0.001) || 0.001))} total)
          </Typography>
          {freqSignificant && <SignificanceBadge type="frequency" />}
          {qualitySignificant && <SignificanceBadge type="quality" />}
        </Stack>
        
        {/* Quality Metrics */}
        {qualityValue !== undefined && (
          <Typography variant="caption" color="text.secondary">
            {qualityMetric}: {qualityValue.toFixed(2)}
            {qualityDelta !== undefined && qualityDelta !== 0 && (
              <Box component="span" sx={{ 
                color: qualityDelta > 0 ? 'success.main' : 'error.main',
                fontWeight: 500,
                ml: 0.5
              }}>
                (Δ{qualityDelta > 0 ? '+' : ''}{qualityDelta.toFixed(2)})
              </Box>
            )}
          </Typography>
        )}
        
        {/* Metadata Tags */}
        {cluster.metadata && (
          <TagChips metadata={cluster.metadata} />
        )}
        
        {/* Expandable Quality Delta Chart */}
        <Collapse in={expanded}>
          {Object.keys(qualityDeltaMetrics).length > 0 && (
            <Box sx={{ mt: 1, pt: 1, borderTop: `1px solid ${theme.palette.divider}` }}>
              <Typography variant="caption" sx={{ fontWeight: 500, color: 'text.secondary', mb: 0.5, display: 'block' }}>
                Quality Delta by Metric
              </Typography>
              <Plot
                data={[{
                  type: 'bar',
                  x: Object.keys(qualityDeltaMetrics),
                  y: Object.values(qualityDeltaMetrics),
                  marker: { 
                    color: Object.values(qualityDeltaMetrics).map(v => 
                      v > 0 ? '#10B981' : v < 0 ? '#EF4444' : '#6B7280'
                    )
                  },
                  hovertemplate: '%{x}: %{y:.3f}<extra></extra>',
                  text: Object.values(qualityDeltaMetrics).map(v => v.toFixed(3)),
                  textposition: 'outside',
                  cliponaxis: false
                }]}
                layout={{
                  height: 200,
                  margin: { l: 50, r: 10, t: 10, b: 60 },
                  xaxis: { tickangle: -30, automargin: true },
                  yaxis: { title: { text: 'Quality Δ' }, tickformat: '.3f', zeroline: true },
                  showlegend: false,
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent'
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            </Box>
          )}
        </Collapse>
      </Stack>
    </Box>
  );
}

export default ClusterItem;
