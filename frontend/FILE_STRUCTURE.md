# Frontend File Structure Guide

This document provides an overview of all files in the StringSight frontend, organized by purpose and location.

## Core Application Files

### Entry Points
- [index.html](index.html) - Main HTML entry point for the Vite app
- [src/main.tsx](src/main.tsx) - React application initialization with providers (React Query, MUI Theme)
- [src/App.tsx](src/App.tsx) - Root application component with routing and state management

### Configuration
- [package.json](package.json) - NPM dependencies and scripts
- [package-lock.json](package-lock.json) - Locked dependency versions
- [vite.config.ts](vite.config.ts) - Vite bundler configuration
- [tsconfig.json](tsconfig.json) - TypeScript configuration for the project
- [tsconfig.app.json](tsconfig.app.json) - TypeScript configuration for app source files
- [tsconfig.node.json](tsconfig.node.json) - TypeScript configuration for Node.js files

### Styling
- [src/App.css](src/App.css) - Application-specific styles
- [src/index.css](src/index.css) - Global styles and CSS resets
- [src/theme.ts](src/theme.ts) - Material-UI theme configuration

## Library Files (`src/lib/`)

Core utilities for data processing and API communication:

- [api.ts](src/lib/api.ts) - All backend API calls and endpoints including:
  - File upload and validation
  - DataFrame operations (select, groupby, custom)
  - Property extraction (single, batch, async jobs)
  - Clustering API
  - Results loading
- [normalize.ts](src/lib/normalize.ts) - Data normalization (flattens nested score objects into columns)
- [parse.ts](src/lib/parse.ts) - File parsing for CSV, JSON, and JSONL formats
- [traces.ts](src/lib/traces.ts) - Conversation trace formatting and OpenAI message format conversion

## Type Definitions (`src/types/`)

TypeScript interfaces and type definitions:

- [metrics.ts](src/types/metrics.ts) - Complete metrics data types including:
  - API response payloads
  - Model cluster and benchmark data structures
  - Filters and sorting options
  - Chart configurations
- [operations.ts](src/types/operations.ts) - Data operation types for tracking data provenance (filters, sorts, custom code)
- [react-plotly-js.d.ts](src/types/react-plotly-js.d.ts) - TypeScript definitions for Plotly
- [vite-env.d.ts](src/vite-env.d.ts) - Vite environment type definitions

## Custom Hooks (`src/hooks/`)

React hooks for data fetching and state management:

- [metrics/useMetricsData.tsx](src/hooks/metrics/useMetricsData.tsx) - React Query hook for loading metrics data with automatic fallback

## Components (`src/components/`)

### Main Tab Components
- [ClustersTab.tsx](src/components/ClustersTab.tsx) (741 lines) - Displays clusters with frequency/quality charts, filtering, and sorting
- [PropertiesTab.tsx](src/components/PropertiesTab.tsx) (502 lines) - Shows extracted properties with filtering, grouping, and custom code operations
- [MetricsContent.tsx](src/components/MetricsContent.tsx) (387 lines) - Main metrics visualization with model comparison

### Sidebar Components
- [ControlSidebar.tsx](src/components/ControlSidebar.tsx) (342 lines) - Main control panel for the application
- [PermanentIconSidebar.tsx](src/components/PermanentIconSidebar.tsx) (104 lines) - Icon-based navigation sidebar
- [ExpandedSidebar.tsx](src/components/ExpandedSidebar.tsx) (83 lines) - Expandable sidebar with detailed controls

#### Sidebar Sections (`sidebar-sections/`)
- [PropertyExtractionPanel.tsx](src/components/sidebar-sections/PropertyExtractionPanel.tsx) (475 lines) - Controls for running property extraction with model settings
- [MetricsPanel.tsx](src/components/sidebar-sections/MetricsPanel.tsx) (243 lines) - Metrics filtering and configuration panel
- [ClusteringPanel.tsx](src/components/sidebar-sections/ClusteringPanel.tsx) (191 lines) - Clustering configuration (embedding model, min cluster size, grouping)
- [DataStatsPanel.tsx](src/components/sidebar-sections/DataStatsPanel.tsx) (125 lines) - Dataset statistics and information display

### Data Display Components
- [DataTable.tsx](src/components/DataTable.tsx) (239 lines) - Reusable table component with sorting and viewing
- [FilterBar.tsx](src/components/FilterBar.tsx) (220 lines) - Filter creation UI with column/value selection
- [FilterSummary.tsx](src/components/FilterSummary.tsx) (86 lines) - Displays active filters as chips
- [ColumnSelector.tsx](src/components/ColumnSelector.tsx) (363 lines) - Multi-select component for choosing visible columns
- [FormattedCell.tsx](src/components/FormattedCell.tsx) (129 lines) - Cell formatting with expand/collapse for long content

### Conversation Display Components
- [ConversationTrace.tsx](src/components/ConversationTrace.tsx) (193 lines) - Displays conversation messages in chat format
- [SideBySideTrace.tsx](src/components/SideBySideTrace.tsx) (41 lines) - Shows side-by-side model comparisons
- [PropertyTraceHeader.tsx](src/components/PropertyTraceHeader.tsx) (215 lines) - Header for property trace display with metadata

### File Browsing Components
- [ServerFileBrowser.tsx](src/components/ServerFileBrowser.tsx) (245 lines) - Browse and load files from the server
- [ServerFolderBrowser.tsx](src/components/ServerFolderBrowser.tsx) (230 lines) - Navigate server folder structure
- [RemoteBrowserDialog.tsx](src/components/RemoteBrowserDialog.tsx) (103 lines) - Dialog for remote file browsing

### Card Components (`components/cards/`)
Display cards for various data types:

- [ClusterCard.tsx](src/components/cards/ClusterCard.tsx) (55 lines) - Individual cluster display card
- [PropertyCard.tsx](src/components/cards/PropertyCard.tsx) (132 lines) - Property extraction result card
- [ModelResponseCard.tsx](src/components/cards/ModelResponseCard.tsx) (165 lines) - Model response display with metadata
- [ResponseContent.tsx](src/components/cards/ResponseContent.tsx) (185 lines) - Formats and displays response content (text, JSON, markdown)
- [CardTestPage.tsx](src/components/cards/CardTestPage.tsx) (198 lines) - Test page for card components

### Metrics Components (`components/metrics/`)

Main metrics functionality with detailed visualizations:

- [MetricsTab.tsx](src/components/metrics/MetricsTab.tsx) (342 lines) - Main metrics tab with all sections
- [MetricsMainContent.tsx](src/components/metrics/MetricsMainContent.tsx) (239 lines) - Main content area for metrics
- [MetricsControlPanel.tsx](src/components/metrics/MetricsControlPanel.tsx) (381 lines) - Control panel for metrics filtering and configuration
- [BenchmarkSection.tsx](src/components/metrics/BenchmarkSection.tsx) (49 lines) - Benchmark data display section
- [BenchmarkTable.tsx](src/components/metrics/BenchmarkTable.tsx) (232 lines) - Table for benchmark metrics
- [DataTabBenchmarkTable.tsx](src/components/metrics/DataTabBenchmarkTable.tsx) (92 lines) - Simplified benchmark table for data tab
- [ClusterPlotsSection.tsx](src/components/metrics/ClusterPlotsSection.tsx) (110 lines) - Section for cluster plot visualizations
- [ModelCardsSection.tsx](src/components/metrics/ModelCardsSection.tsx) (99 lines) - Section displaying model cards
- [index.ts](src/components/metrics/index.ts) - Barrel export for metrics components
- [types.ts](src/components/metrics/types.ts) - Local type definitions for metrics components
- [utils/metricUtils.ts](src/components/metrics/utils/metricUtils.ts) - Utility functions for metrics calculations

#### Metrics Cards (`components/metrics/cards/`)
- [ModelCard.tsx](src/components/metrics/cards/ModelCard.tsx) - Individual model performance card
- [ModelCardsGrid.tsx](src/components/metrics/cards/ModelCardsGrid.tsx) - Grid layout for model cards
- [ClusterItem.tsx](src/components/metrics/cards/ClusterItem.tsx) - Individual cluster item in model card
- [SignificanceBadge.tsx](src/components/metrics/cards/SignificanceBadge.tsx) - Badge showing statistical significance
- [TagChips.tsx](src/components/metrics/cards/TagChips.tsx) - Chip display for cluster tags/metadata

#### Metrics Charts (`components/metrics/charts/`)
Plotly-based chart components:

- [PlotlyChartBase.tsx](src/components/metrics/charts/PlotlyChartBase.tsx) - Base component for all Plotly charts
- [FrequencyChart.tsx](src/components/metrics/charts/FrequencyChart.tsx) - Absolute frequency/proportion visualization
- [FrequencyDeltaChart.tsx](src/components/metrics/charts/FrequencyDeltaChart.tsx) - Frequency change visualization with zero line
- [QualityChart.tsx](src/components/metrics/charts/QualityChart.tsx) - Absolute quality metric visualization
- [QualityDeltaChart.tsx](src/components/metrics/charts/QualityDeltaChart.tsx) - Quality metric change visualization
- [BenchmarkChart.tsx](src/components/metrics/charts/BenchmarkChart.tsx) - Overall benchmark comparison chart

### Legacy/Standalone Components
- [BenchmarkChart.tsx](src/components/BenchmarkChart.tsx) (207 lines) - Older benchmark chart component (may be superseded by metrics/charts version)

## Documentation

- [README.md](README.md) - Frontend setup, architecture, and usage guide
- [METRICS_README.md](METRICS_README.md) - Detailed documentation for the metrics functionality

## Data Flow

1. **File Loading**: Files are uploaded via [ServerFileBrowser](src/components/ServerFileBrowser.tsx) or drag-and-drop
2. **Parsing**: Data is parsed in [parse.ts](src/lib/parse.ts) and validated via [api.ts](src/lib/api.ts)
3. **Normalization**: Score objects are flattened by [normalize.ts](src/lib/normalize.ts)
4. **Display**: Data flows to tab components ([ClustersTab](src/components/ClustersTab.tsx), [PropertiesTab](src/components/PropertiesTab.tsx), [MetricsContent](src/components/MetricsContent.tsx))
5. **Operations**: Users can filter, sort, and transform data via sidebar panels
6. **API Calls**: All backend communication goes through [api.ts](src/lib/api.ts)

## Key Features by File

### Property Extraction
- Configuration: [PropertyExtractionPanel.tsx](src/components/sidebar-sections/PropertyExtractionPanel.tsx)
- API: [api.ts](src/lib/api.ts) (`extractSingle`, `extractBatch`, `extractJob*`)
- Display: [PropertiesTab.tsx](src/components/PropertiesTab.tsx)

### Clustering
- Configuration: [ClusteringPanel.tsx](src/components/sidebar-sections/ClusteringPanel.tsx)
- API: [api.ts](src/lib/api.ts) (`runClustering`, `recomputeClusterMetrics`)
- Display: [ClustersTab.tsx](src/components/ClustersTab.tsx)

### Metrics & Benchmarking
- Configuration: [MetricsPanel.tsx](src/components/sidebar-sections/MetricsPanel.tsx)
- Data Types: [types/metrics.ts](src/types/metrics.ts)
- Display: [MetricsTab.tsx](src/components/metrics/MetricsTab.tsx)
- Charts: [components/metrics/charts/](src/components/metrics/charts/)

### Conversation Viewing
- Single Model: [ConversationTrace.tsx](src/components/ConversationTrace.tsx)
- Side-by-Side: [SideBySideTrace.tsx](src/components/SideBySideTrace.tsx)
- Format Handling: [traces.ts](src/lib/traces.ts)

## Component Size Reference

**Large Components (300+ lines)**
- ClustersTab (741), PropertiesTab (502), PropertyExtractionPanel (475), MetricsContent (387), ColumnSelector (363), MetricsControlPanel (381), ControlSidebar (342), MetricsTab (342)

**Medium Components (200-300 lines)**
- MetricsPanel (243), MetricsMainContent (239), DataTable (239), BenchmarkTable (232), ServerFolderBrowser (230), ServerFileBrowser (245), FilterBar (220), PropertyTraceHeader (215), BenchmarkChart (207)

**Small Components (<200 lines)**
- All other components listed above
