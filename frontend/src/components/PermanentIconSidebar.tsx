import React from 'react';
import { Box, IconButton, Tooltip } from '@mui/material';
import {
  TableView as DataViewIcon,
  FindInPage as PropertyExtractionIcon,
  ScatterPlot as ClusteringIcon,
  Analytics as MetricsIcon
} from '@mui/icons-material';

export type SidebarSection = 'data' | 'extraction' | 'clustering' | 'metrics';

interface PermanentIconSidebarProps {
  activeSection: SidebarSection;
  onSectionChange: (section: SidebarSection) => void;
}

interface IconButtonItemProps {
  icon: React.ReactNode;
  tooltip: string;
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
}

function IconButtonItem({ icon, tooltip, active, onClick, disabled = false }: IconButtonItemProps) {
  return (
    <Tooltip title={tooltip} placement="right">
      <Box sx={{ mb: 1 }}>
        <IconButton
          onClick={onClick}
          disabled={disabled}
          sx={{
            width: 48,
            height: 48,
            borderRadius: 2,
            backgroundColor: active ? 'primary.main' : 'transparent',
            color: active ? 'primary.contrastText' : 'text.secondary',
            '&:hover': {
              backgroundColor: active ? 'primary.dark' : 'action.hover',
            },
            '&.Mui-disabled': {
              color: 'action.disabled',
            },
            transition: 'all 0.2s ease-in-out',
          }}
        >
          {icon}
        </IconButton>
      </Box>
    </Tooltip>
  );
}

export default function PermanentIconSidebar({ activeSection, onSectionChange }: PermanentIconSidebarProps) {
  return (
    <Box
      sx={{
        width: 60,
        height: '100vh',
        backgroundColor: 'background.paper',
        borderRight: '1px solid',
        borderColor: 'divider',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        py: 2,
        position: 'fixed',
        left: 0,
        top: (theme) => theme.mixins.toolbar.minHeight, // Start below header
        bottom: 0,
        zIndex: 1200,
      }}
    >
      <IconButtonItem
        icon={<DataViewIcon />}
        tooltip="Data Viewing"
        active={activeSection === 'data'}
        onClick={() => onSectionChange('data')}
      />
      
      <IconButtonItem
        icon={<PropertyExtractionIcon />}
        tooltip="Property Extraction"
        active={activeSection === 'extraction'}
        onClick={() => onSectionChange('extraction')}
      />
      
      <IconButtonItem
        icon={<ClusteringIcon />}
        tooltip="Clustering"
        active={activeSection === 'clustering'}
        onClick={() => onSectionChange('clustering')}
      />
      
      <IconButtonItem
        icon={<MetricsIcon />}
        tooltip="Metrics"
        active={activeSection === 'metrics'}
        onClick={() => onSectionChange('metrics')}
      />
    </Box>
  );
}

