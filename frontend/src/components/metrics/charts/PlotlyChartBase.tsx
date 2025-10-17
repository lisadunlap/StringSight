/**
 * PlotlyChartBase - Shared Plotly configuration and styling for metrics charts.
 * 
 * This component provides consistent styling, colors, and behavior across all
 * metrics charts while allowing customization for specific chart types.
 */

import { Box, useTheme } from '@mui/material';
// @ts-ignore - Plotly types issue
import Plotly from 'plotly.js-dist-min';
// @ts-ignore - React-plotly types issue  
import createPlotlyComponent from 'react-plotly.js/factory';

const Plot = createPlotlyComponent(Plotly);

export interface PlotlyChartBaseProps {
  /** Chart data in Plotly format */
  data: Plotly.Data[];
  /** Chart layout configuration */
  layout?: Partial<Plotly.Layout>;
  /** Chart configuration options */
  config?: Partial<Plotly.Config>;
  /** Chart height in pixels */
  height?: number;
  /** Whether to show a zero line (for delta charts) */
  showZeroLine?: boolean;
  /** Chart title */
  title?: string;
  /** X-axis label */
  xAxisLabel?: string;
  /** Y-axis label */
  yAxisLabel?: string;
  /** Loading state */
  loading?: boolean;
}

/**
 * Standard color palette for models - consistent across all charts
 */
export const MODEL_COLORS = [
  '#1f77b4', // blue
  '#ff7f0e', // orange  
  '#2ca02c', // green
  '#d62728', // red
  '#9467bd', // purple
  '#8c564b', // brown
  '#e377c2', // pink
  '#7f7f7f', // gray
  '#bcbd22', // olive
  '#17becf', // cyan
  '#aec7e8', // light blue
  '#ffbb78', // light orange
  '#98df8a', // light green
  '#ff9896', // light red
  '#c5b0d5', // light purple
];

/**
 * Get consistent color for a model name
 */
export function getModelColor(modelName: string, allModels: string[]): string {
  const index = allModels.indexOf(modelName);
  return MODEL_COLORS[index % MODEL_COLORS.length];
}

export function PlotlyChartBase({
  data,
  layout = {},
  config = {},
  height = 400,
  showZeroLine = false,
  title,
  xAxisLabel,
  yAxisLabel,
  loading = false
}: PlotlyChartBaseProps) {
  const theme = useTheme();
  
  // Default layout configuration matching Material-UI theme
  const defaultLayout: Partial<Plotly.Layout> = {
    height,
    margin: { t: title ? 80 : 50, r: 30, b: 80, l: 80 },
    plot_bgcolor: 'transparent',
    paper_bgcolor: 'transparent',
    font: {
      family: theme.typography.fontFamily,
      size: 12,
      color: theme.palette.text.primary
    },
    title: title ? {
      text: title,
      font: {
        size: 16,
        color: theme.palette.text.primary
      },
      x: 0.02,
      xanchor: 'left'
    } : undefined,
    xaxis: {
      title: xAxisLabel ? {
        text: xAxisLabel,
        font: { size: 12 }
      } : undefined,
      gridcolor: theme.palette.mode === 'dark' ? '#333' : '#eee',
      linecolor: theme.palette.divider,
      tickfont: { size: 10 },
      automargin: true
    },
    yaxis: {
      title: yAxisLabel ? {
        text: yAxisLabel,
        font: { size: 12 }
      } : undefined,
      gridcolor: theme.palette.mode === 'dark' ? '#333' : '#eee',
      linecolor: theme.palette.divider,
      tickfont: { size: 10 },
      automargin: true,
      ...(showZeroLine ? {
        zeroline: true,
        zerolinecolor: theme.palette.text.secondary,
        zerolinewidth: 1
      } : {})
    },
    legend: {
      orientation: 'h' as const,
      x: 0,
      y: 1.02,
      xanchor: 'left',
      yanchor: 'bottom',
      font: { size: 10 },
      bgcolor: 'transparent'
    },
    showlegend: data.length > 1,
    hovermode: 'closest' as const,
    hoverlabel: {
      bgcolor: theme.palette.mode === 'dark' ? '#333' : 'white',
      bordercolor: theme.palette.divider,
      font: {
        family: theme.typography.fontFamily,
        size: 12,
        color: theme.palette.text.primary
      },
      align: 'left' as const
    }
  };

  // Default config for consistent behavior
  const defaultConfig: Partial<Plotly.Config> = {
    displayModeBar: true,
    modeBarButtonsToRemove: [
      'pan2d',
      'select2d', 
      'lasso2d',
      'autoScale2d',
      'toggleSpikelines',
      'hoverCompareCartesian',
      'hoverClosestCartesian'
    ],
    displaylogo: false,
    responsive: true,
    ...config
  };

  // Merge layouts
  const finalLayout = {
    ...defaultLayout,
    ...layout,
    // Deep merge axis configs
    xaxis: { ...defaultLayout.xaxis, ...layout.xaxis },
    yaxis: { ...defaultLayout.yaxis, ...layout.yaxis }
  };

  if (loading) {
    return (
      <Box 
        sx={{ 
          height, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          color: 'text.secondary' 
        }}
      >
        Loading chart...
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', height, minWidth: 0 }}>
      <Plot
        data={data}
        layout={finalLayout}
        config={defaultConfig}
        style={{ width: '100%', height: '100%', minWidth: 0 }}
        useResizeHandler={true}
      />
    </Box>
  );
}

/**
 * Utility function to truncate cluster labels for x-axis display
 */
export function truncateLabel(label: string, maxLength: number = 20): string {
  if (label.length <= maxLength) return label;
  return label.substring(0, maxLength - 3) + '...';
}

/**
 * Utility function to format hover text with full cluster name
 * Wraps long text at approximately 50 characters per line
 */
export function createHoverTemplate(
  fullClusterName: string,
  value: number,
  metric: string,
  decimals: number = 3
): string {
  const wrappedName = wrapText(fullClusterName, 50);
  return `<b>${wrappedName}</b><br>${metric}: ${value.toFixed(decimals)}<extra></extra>`;
}

/**
 * Wraps text at word boundaries to fit within maxCharsPerLine
 */
function wrapText(text: string, maxCharsPerLine: number = 50): string {
  const words = text.split(' ');
  const lines: string[] = [];
  let currentLine = '';

  for (const word of words) {
    if (currentLine.length === 0) {
      currentLine = word;
    } else if (currentLine.length + word.length + 1 <= maxCharsPerLine) {
      currentLine += ' ' + word;
    } else {
      lines.push(currentLine);
      currentLine = word;
    }
  }

  if (currentLine.length > 0) {
    lines.push(currentLine);
  }

  return lines.join('<br>');
}

export default PlotlyChartBase;
