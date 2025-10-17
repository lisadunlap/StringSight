import React from 'react';
import { Box, IconButton, Typography, Drawer } from '@mui/material';
import { ChevronLeft as CollapseIcon } from '@mui/icons-material';
import { type SidebarSection } from './PermanentIconSidebar';

interface ExpandedSidebarProps {
  activeSection: SidebarSection;
  expanded: boolean;
  onToggleExpanded: () => void;
  children: React.ReactNode;
}

const sectionTitles: Record<SidebarSection, string> = {
  data: 'Data Statistics',
  extraction: 'Property Extraction',
  clustering: 'Clustering Analysis',
  metrics: 'Metrics Dashboard'
};

export default function ExpandedSidebar({ 
  activeSection, 
  expanded, 
  onToggleExpanded, 
  children 
}: ExpandedSidebarProps) {
  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={expanded}
      sx={{
        width: expanded ? 400 : 0,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 400,
          boxSizing: 'border-box',
          left: 60, // Account for permanent icon sidebar
          top: 64, // Start below header (standard AppBar height)
          height: 'calc(100vh - 64px)',
          borderRight: '1px solid',
          borderColor: 'divider',
          backgroundColor: 'background.paper',
        },
      }}
    >
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* Header */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            p: 2,
            borderBottom: '1px solid',
            borderColor: 'divider',
            minHeight: 64,
          }}
        >
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            {sectionTitles[activeSection]}
          </Typography>
          <IconButton
            aria-label="Collapse panel"
            onClick={onToggleExpanded}
            sx={{
              '&:hover': {
                backgroundColor: 'action.hover',
              },
            }}
            title="Collapse panel"
          >
            <CollapseIcon />
          </IconButton>
        </Box>

        {/* Content */}
        <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
          {children}
        </Box>
      </Box>
    </Drawer>
  );
}
