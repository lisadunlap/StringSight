import React, { useState, useCallback, useMemo, useRef, Component } from "react";
import { Box, AppBar, Toolbar, Typography, Container, Button, Drawer, Stack, Accordion, AccordionSummary, AccordionDetails, Pagination, Tabs, Tab, LinearProgress } from "@mui/material";
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import VisibilityOutlinedIcon from '@mui/icons-material/VisibilityOutlined';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import { detectAndValidate, dfGroupPreview, dfGroupRows, dfCustom, resultsLoad, readPath } from "./lib/api";
import { flattenScores } from "./lib/normalize";
import { parseFile } from "./lib/parse";
import { detectMethodFromColumns, ensureOpenAIFormat } from "./lib/traces";
import DataTable from "./components/DataTable";
import ConversationTrace from "./components/ConversationTrace";
import SideBySideTrace from "./components/SideBySideTrace";
import FormattedCell from "./components/FormattedCell";
import FilterSummary from "./components/FilterSummary";
import PropertyTraceHeader from "./components/PropertyTraceHeader";
// import BenchmarkChart from "./components/BenchmarkChart";
import DataTabBenchmarkTable from "./components/metrics/DataTabBenchmarkTable";

import PropertiesTab from "./components/PropertiesTab";
import FilterBar from "./components/FilterBar";
import PermanentIconSidebar, { type SidebarSection } from "./components/PermanentIconSidebar";
import ExpandedSidebar from "./components/ExpandedSidebar";
import DataStatsPanel from "./components/sidebar-sections/DataStatsPanel";
import PropertyExtractionPanel from "./components/sidebar-sections/PropertyExtractionPanel";
import ClusteringPanel from "./components/sidebar-sections/ClusteringPanel";
import { recomputeClusterMetrics } from "./lib/api";
import ClustersTab from "./components/ClustersTab";
import MetricsPanel from "./components/sidebar-sections/MetricsPanel";
import type { MetricsFilters, MetricsSummary } from "./types/metrics";
import { ColumnSelector, type ColumnMapping } from "./components/ColumnSelector";
import CardTestPage from "./components/cards/CardTestPage";
import { MetricsTab } from "./components/metrics/MetricsTab";
import ServerFolderBrowser from "./components/ServerFolderBrowser";
import ServerFileBrowser from "./components/ServerFileBrowser";
import type { DataOperation } from "./types/operations";
import { createFilterOperation, createCustomCodeOperation, createSortOperation } from "./types/operations";




/**
 * Pick a single-model assistant response from a results conversation object.
 * Accepts multiple common aliases. Returns either a string, an array of {role, content}, or empty string.
 */
function pickSingleResponse(c: any): any {
  if (c && Object.prototype.hasOwnProperty.call(c, 'responses')) return c.responses;
  if (c && Object.prototype.hasOwnProperty.call(c, 'response')) return c.response;
  if (c && Object.prototype.hasOwnProperty.call(c, 'model_response')) return c.model_response;
  // Prefer full chat history if provided
  if (Array.isArray(c?.messages)) return c.messages;
  // Common aliases
  if (c && Object.prototype.hasOwnProperty.call(c, 'assistant')) return c.assistant;
  if (c && Object.prototype.hasOwnProperty.call(c, 'output')) return c.output;
  if (c && Object.prototype.hasOwnProperty.call(c, 'completion')) return c.completion;
  if (c && Object.prototype.hasOwnProperty.call(c, 'text')) return c.text;
  return '';
}

/**
 * Pick pair of assistant responses for side-by-side results.
 * Returns a tuple [respA, respB], each can be string or array of {role, content}.
 */
function pickPairResponses(c: any): [any, any] {
  if (Array.isArray(c?.responses) && c.responses.length >= 2) return [c.responses[0], c.responses[1]];
  // Prefer explicit messages arrays per side if present
  if (Array.isArray((c as any)?.messages_a) && Array.isArray((c as any)?.messages_b)) return [(c as any).messages_a, (c as any).messages_b];
  // Common aliases for A/B single strings
  const a = (c as any)?.response_a ?? (c as any)?.assistant_a ?? (c as any)?.output_a ?? (c as any)?.completion_a ?? (c as any)?.text_a ?? '';
  const b = (c as any)?.response_b ?? (c as any)?.assistant_b ?? (c as any)?.output_b ?? (c as any)?.completion_b ?? (c as any)?.text_b ?? '';
  return [a, b];
}

class ErrorBoundary extends Component<{children: React.ReactNode}, {hasError: boolean, error?: Error}> {
  constructor(props: {children: React.ReactNode}) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h4" color="error" gutterBottom>
            Something went wrong
          </Typography>
          <Typography variant="body1" sx={{ mb: 2 }}>
            {this.state.error?.message || 'An unexpected error occurred'}
          </Typography>
          <Button variant="contained" onClick={() => window.location.reload()}>
            Reload Page
          </Button>
        </Box>
      );
    }

    return <>{this.props.children}</>;
  }
}

function App() {
  // Data management layers as suggested
  const [originalRows, setOriginalRows] = useState<Record<string, any>[]>([]); // Raw uploaded data
  const [operationalRows, setOperationalRows] = useState<Record<string, any>[]>([]); // Cleaned, filtered columns
  const [currentRows, setCurrentRows] = useState<Record<string, any>[]>([]); // With filters applied
  
  const [method, setMethod] = useState<"single_model" | "side_by_side" | "unknown">("unknown");
  const [drawerOpen, setDrawerOpen] = useState(false);

  const [selectedTrace, setSelectedTrace] = useState<any>(null);
  const [selectedRow, setSelectedRow] = useState<Record<string, any> | null>(null);
  const [selectedRowForExtraction, setSelectedRowForExtraction] = useState<Record<string, any> | null>(null);
  const [selectedEvidence, setSelectedEvidence] = useState<string[] | null>(null);
  const [evidenceTargetModel, setEvidenceTargetModel] = useState<string | undefined>(undefined);
  const [selectedProperty, setSelectedProperty] = useState<any | null>(null); // Track the property when viewing from properties table
  const [propertiesByKey, setPropertiesByKey] = useState<Map<string, any[]>>(new Map());
  const [propertiesRows, setPropertiesRows] = useState<any[]>([]);
  const [activeSection, setActiveSection] = useState<SidebarSection>('data');
  const [sidebarExpanded, setSidebarExpanded] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'table'|'properties'|'clusters'|'metrics'|'test'>('table');
  const [hasViewedClusters, setHasViewedClusters] = useState<boolean>(false);
  // Debug toggles
  const [debugCacheHits, setDebugCacheHits] = useState<boolean>(false);
  // Results loading indicator
  const [isLoadingResults, setIsLoadingResults] = useState<boolean>(false);
  const [resultsLoadingMessage, setResultsLoadingMessage] = useState<string>('');
  const [resultsError, setResultsError] = useState<string | null>(null);
  // -------- Clustering State ---------
  const [clusters, setClusters] = useState<any[]>([]);
  const [totalConversationsByModel, setTotalConversationsByModel] = useState<Record<string, number> | null>(null);
  const [totalUniqueConversations, setTotalUniqueConversations] = useState<number | null>(null);
  // Results mode (when loading full_dataset.json)
  const [isResultsMode, setIsResultsMode] = useState<boolean>(false);
  const [resultsMetrics, setResultsMetrics] = useState<{ model_cluster_scores?: any; cluster_scores?: any; model_scores?: any } | null>(null);

  // -------- Display Settings ---------
  const [decimalPrecision, setDecimalPrecision] = useState<number>(2);

  // -------- Metrics Tab State ---------
  const [metricsFilters, setMetricsFilters] = useState<MetricsFilters>({
    selectedModels: [],
    selectedGroups: [],
    qualityMetric: '',
    sortBy: 'proportion_delta_desc',
    topN: 15,
    significanceOnly: false,
    showCI: false,
  });

  // Metrics data for sidebar
  const [metricsAvailableModels, setMetricsAvailableModels] = useState<string[]>([]);
  const [metricsAvailableGroups, setMetricsAvailableGroups] = useState<string[]>([]);
  const [metricsAvailableQualityMetrics, setMetricsAvailableQualityMetrics] = useState<string[]>([]);
  const [metricsSummary, setMetricsSummary] = useState<MetricsSummary | null>(null);

  // Reset metrics filters when new data is loaded
  React.useEffect(() => {
    if (resultsMetrics) {
      // Reset filters to defaults when new data is loaded
      setMetricsFilters({
        selectedModels: [],
        selectedGroups: [],
        qualityMetric: '',
        sortBy: 'proportion_delta_desc',
        topN: 15,
        significanceOnly: false,
        showCI: false,
      });
      // Clear previous metrics metadata
      setMetricsAvailableModels([]);
      setMetricsAvailableGroups([]);
      setMetricsAvailableQualityMetrics([]);
      setMetricsSummary(null);
    }
  }, [resultsMetrics]);

  // Auto-switch to Clustering section only the first time clusters appear
  const hasAutoSwitchedToClustersRef = useRef<boolean>(false);
  React.useEffect(() => {
    if (!isResultsMode && !hasAutoSwitchedToClustersRef.current && clusters && clusters.length > 0) {
      setActiveSection('clustering');
      setActiveTab('clusters');
      hasAutoSwitchedToClustersRef.current = true;
    }
  }, [clusters, isResultsMode]);

  // (reserved) keying strategies for recomputes can be added later when needed

  // Remove auto-recompute; metrics will be explicitly recomputed by the ClustersTab when filters change
  const [batchRunning, setBatchRunning] = useState<boolean>(false);
  const [batchProgress, setBatchProgress] = useState<number>(0);
  const [batchState, setBatchState] = useState<string | null>(null);
  
  // Flexible column mapping state
  const [showColumnSelector, setShowColumnSelector] = useState(false);
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [autoDetectedMapping, setAutoDetectedMapping] = useState<ColumnMapping | null>(null);
  const [columnMapping, setColumnMapping] = useState<ColumnMapping | null>(null); // reserved for future use
  const [mappingValid, setMappingValid] = useState(false);
  const [mappingErrors, setMappingErrors] = useState<string[]>([]); // reserved for future validation UI
  const [filterNotice, setFilterNotice] = useState<string | null>(null);
  // Server-side folder browser state
  const [showFolderBrowser, setShowFolderBrowser] = useState(false);
  // Server-side file browser state
  const [showFileBrowser, setShowFileBrowser] = useState(false);
  // Local folder input ref for results loading (set webkitdirectory via ref to avoid TS complaints)
  const resultsFolderInputRef = useRef<HTMLInputElement | null>(null);
  React.useEffect(() => {
    if (resultsFolderInputRef.current) {
      try {
        resultsFolderInputRef.current.setAttribute('webkitdirectory', '');
        resultsFolderInputRef.current.setAttribute('directory', '');
        resultsFolderInputRef.current.setAttribute('multiple', '');
      } catch {}
    }
  }, []);

  // Reset UI, panels, and tabs when a brand new source is loaded
  const resetUiStateForNewSource = React.useCallback((mode: 'file' | 'results') => {
    // Core data and mapping
    setOriginalRows([]);
    setOperationalRows([]);
    setCurrentRows([]);
    setAvailableColumns([]);
    setAutoDetectedMapping(null);
    setColumnMapping(null);
    setFilterNotice(null);
    setMethod('unknown');
    setIsResultsMode(mode === 'results');
    setResultsMetrics(null);

    // Panels, tabs, and sidebar
    setActiveSection('data');
    setActiveTab('table');
    setHasViewedClusters(false);
    setSidebarExpanded(false);

    // Drawer and selections
    setDrawerOpen(false);
    setSelectedTrace(null);
    setSelectedRow(null);
    setSelectedRowForExtraction(null);
    setSelectedEvidence(null);
    setEvidenceTargetModel(undefined);
    setSelectedProperty(null);

    // Properties and clusters
    setPropertiesByKey(new Map());
    setPropertiesRows([]);
    setClusters([]);

    // Operations, filters, grouping, sorting
    setOperationChain([]);
    setGroupBy(null);
    setGroupPreview([]);
    setExpandedGroup(null);
    setGroupRows([]);
    setGroupTotal(0);
    setGroupPagination(new Map());
    setSortColumn(null);
    setSortDirection(null);
    setDataSearchQuery('');

    // Custom code sandbox
    setCustomCode("");
    setCustomError(null);
  }, []);

  async function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    // Assume a completely different source; reset panels and tabs
    resetUiStateForNewSource('file');
    setIsLoadingResults(true);
    setResultsLoadingMessage('Parsing file and preparing data...');
    
    // Parse the file
    const { rows, columns } = await parseFile(file);
    
    // Store raw data and columns
    setOriginalRows(rows);
    setAvailableColumns(columns);
    setFilterNotice(null);
    
    // Try to auto-detect column mapping using legacy method as fallback
    const legacyDetected = detectMethodFromColumns(columns);
    
    // Create auto-detected mapping based on legacy detection and available columns
    const autoMapping: ColumnMapping = {
      promptCol: columns.find(c => c.toLowerCase() === 'prompt') || '',
      responseCols: legacyDetected === 'side_by_side' 
        ? columns.filter(c => c.includes('model_a_response') || c.includes('model_b_response'))
        : columns.filter(c => c.includes('model_response')),
      modelCols: legacyDetected === 'side_by_side'
        ? columns.filter(c => (c.includes('model_a') || c.includes('model_b')) && !c.includes('response'))
        : columns.filter(c => c.toLowerCase() === 'model'),
      scoreCols: columns.filter(c => c.toLowerCase().includes('score')),
      method: legacyDetected === 'unknown' ? 'single_model' : legacyDetected
    };
    
    setAutoDetectedMapping(autoMapping);
    
    // Always show the column selector after upload (clear, explicit flow for new users)
    setShowColumnSelector(true);
    setMethod('unknown');
    setOperationalRows([]);
    setCurrentRows([]);
    
    try {
      await detectAndValidate(file); // optional backend validation
    } catch (_) {}
    finally {
      setIsLoadingResults(false);
      setResultsLoadingMessage('');
    }
  }

  // Load previously computed results from a selected folder (expects full_dataset.json)
  async function onResultsFolderChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    // Assume a completely different source; reset panels and tabs
    resetUiStateForNewSource('results');
    setIsLoadingResults(true);
    setResultsLoadingMessage('Reading results folder...');

    const fileList = Array.from(files);
    // Locate full_dataset.json anywhere within the selected folder
    const fullDataset = fileList.find((f) => f.name === 'full_dataset.json' || (f as any).webkitRelativePath?.endsWith('/full_dataset.json'));
    if (!fullDataset) {
      console.warn('[ResultsLoader] full_dataset.json not found in selected folder');
      return;
    }

    // Parse dataset
    let data: any;
    try {
      const text = await fullDataset.text();
      data = JSON.parse(text);
    } catch (err) {
      console.error('[ResultsLoader] Failed parsing full_dataset.json', err);
      setIsLoadingResults(false);
      setResultsLoadingMessage('');
      return;
    }

    const conversations: any[] = Array.isArray(data?.conversations) ? data.conversations : [];
    const properties: any[] = Array.isArray(data?.properties) ? data.properties : [];
    const clustersRaw: any[] = Array.isArray(data?.clusters) ? data.clusters : [];

    // Detect method from conversations
    const detectedMethod: 'single_model' | 'side_by_side' = conversations.some((c) => Array.isArray(c?.model)) ? 'side_by_side' : 'single_model';

    // Build operational rows from conversations
    const operational: any[] = conversations.map((c, idx) => {
      const qid = String(c?.question_id ?? idx);
      const prompt = c?.prompt ?? '';
      const meta = (c?.meta && typeof c.meta === 'object') ? c.meta : {};
      // Keep processed (operational) rows minimal for backend compatibility and deterministic UI:
      // Only include: __index, question_id, prompt, model(s), response(s), and score dict(s).
      const row: any = { __index: idx, question_id: qid, prompt };
      if (Array.isArray(c?.model)) {
        // side-by-side
        row.model_a = c.model?.[0] ?? 'model_a';
        row.model_b = c.model?.[1] ?? 'model_b';
        const [respA, respB] = pickPairResponses(c);
        row.model_a_response = respA ?? '';
        row.model_b_response = respB ?? '';
        const scores = c?.scores ?? [{}, {}];
        if (Array.isArray(scores)) {
          row.score_a = (scores?.[0] && typeof scores[0] === 'object') ? scores[0] : {};
          row.score_b = (scores?.[1] && typeof scores[1] === 'object') ? scores[1] : {};
        }
      } else {
        // single model
        row.model = c?.model ?? 'model';
        row.model_response = pickSingleResponse(c);
        const score = (c?.scores && typeof c.scores === 'object' && !Array.isArray(c.scores)) ? c.scores : {};
        if (Object.keys(score).length > 0) row.score = score;
      }
      // Intentionally do NOT copy additional metadata into the processed dataframe.
      // This ensures predictable grouping/aggregation and keeps the operational dataset lean.
      return row;
    });

    // Build a mapping from question_id to __index for property linking
    const qidToIndex = new Map<string, number>();
    operational.forEach((r: any) => { qidToIndex.set(String(r.question_id), Number(r.__index)); });

    // Prepare properties rows with row_index to aid lookup
    const propsWithIndex = properties.map((p: any) => ({ ...p, row_index: qidToIndex.get(String(p?.question_id)) }));

    // Set state: method, operational, current (flattened), properties, clusters
    setMethod(detectedMethod);
    setOperationalRows(operational);
    const { rows: flattenedRows } = flattenScores(operational, detectedMethod);
    setCurrentRows(flattenedRows);
    setPropertiesRows(propsWithIndex);
    setClusters(clustersRaw);
    setIsResultsMode(true);
    // Collapse the control panel and switch to Metrics for an overview-first experience
    setSidebarExpanded(false);
    setActiveSection('data');

    // Enrich clusters with metrics meta so plots and summaries render immediately
    try {
      setResultsLoadingMessage('Computing per-model proportions and quality...');
      const res = await recomputeClusterMetrics({
        clusters: clustersRaw,
        properties: propsWithIndex,
        operationalRows: operational,
      });
      if (res?.clusters) setClusters(res.clusters);
    } catch (err) {
      console.warn('[ResultsLoader] Failed to enrich clusters; showing raw clusters', err);
    }

    // Cache functional metrics if present for future Metrics tab
    try {
      const findFile = (name: string) => fileList.find((f) => f.name === name || (f as any).webkitRelativePath?.endsWith(`/${name}`));
      const readJson = async (f?: File) => (f ? JSON.parse(await f.text()) : undefined);
      
      // Try to load from both old JSON format and new JSONL format
      let modelCluster, clusterScores, modelScores;
      
      // First try old JSON format
      modelCluster = await readJson(findFile('model_cluster_scores.json') as File);
      clusterScores = await readJson(findFile('cluster_scores.json') as File);
      modelScores = await readJson(findFile('model_scores.json') as File);
      
      // If JSON files don't exist, try JSONL format (convert to array format for compatibility)
      if (!modelCluster) {
        const jsonlFile = findFile('model_cluster_scores_df.jsonl');
        if (jsonlFile) {
          const text = await jsonlFile.text();
          modelCluster = text.trim().split('\n').map(line => JSON.parse(line));
        }
      }
      
      if (!modelScores) {
        const jsonlFile = findFile('model_scores_df.jsonl');
        if (jsonlFile) {
          const text = await jsonlFile.text();
          modelScores = text.trim().split('\n').map(line => JSON.parse(line));
        }
      }
      
      if (!clusterScores) {
        const jsonlFile = findFile('cluster_scores_df.jsonl');
        if (jsonlFile) {
          const text = await jsonlFile.text();
          clusterScores = text.trim().split('\n').map(line => JSON.parse(line));
        }
      }
      
      if (modelCluster || clusterScores || modelScores) {
        console.log('📊 Loaded metrics data:', { 
          modelCluster: modelCluster?.length || 0, 
          clusterScores: clusterScores?.length || 0, 
          modelScores: modelScores?.length || 0 
        });
        setResultsMetrics({ model_cluster_scores: modelCluster, cluster_scores: clusterScores, model_scores: modelScores });
      } else {
        console.warn('📊 No metrics files found in results folder');
        setResultsMetrics(null);
      }
    } catch (e) {
      console.error('📊 Error loading metrics:', e);
      // Non-fatal: metrics are optional
      setResultsMetrics(null);
    }
    finally {
      setIsLoadingResults(false);
      setResultsLoadingMessage('');
    }
  }

  // Load a single data file from server-side path (for Load File button)
  async function onServerFileSelected(path: string) {
    console.log('[ServerFileLoader] Loading file from path:', path);
    
    // Reset UI state for new file source
    resetUiStateForNewSource('file');
    setIsLoadingResults(true);
    setResultsLoadingMessage('Loading file from server...');

    try {
      // Load data using the server API
      const data = await readPath(path);
      console.log('[ServerFileLoader] Loaded data:', data);

      const rows = data.preview || [];
      const columns = data.columns || [];
      const detectedMethod = data.method as 'single_model' | 'side_by_side';

      // Store raw data and columns
      setOriginalRows(rows);
      setAvailableColumns(columns);
      setFilterNotice(null);

      // Create auto-detected mapping
      const autoMapping: ColumnMapping = {
        promptCol: columns.find(c => c.toLowerCase() === 'prompt') || '',
        responseCols: detectedMethod === 'side_by_side' 
          ? columns.filter(c => c.includes('model_a_response') || c.includes('model_b_response'))
          : columns.filter(c => c.includes('model_response')),
        modelCols: detectedMethod === 'side_by_side'
          ? columns.filter(c => (c.includes('model_a') || c.includes('model_b')) && !c.includes('response'))
          : columns.filter(c => c.toLowerCase() === 'model'),
        scoreCols: columns.filter(c => c.toLowerCase().includes('score')),
        method: detectedMethod
      };
      
      setAutoDetectedMapping(autoMapping);
      
      // Always show the column selector after upload (clear, explicit flow for new users)
      setShowColumnSelector(true);
      setMethod('unknown');
      setOperationalRows([]);
      setCurrentRows([]);

      console.log(`[ServerFileLoader] Successfully loaded ${rows.length} rows from ${path}`);

    } catch (err) {
      console.error('[ServerFileLoader] Failed to load file:', err);
      // Show error to user (could add error state here)
    }
    finally {
      setIsLoadingResults(false);
      setResultsLoadingMessage('');
    }
  }

  // Load results from server-side path
  async function onServerPathSelected(path: string) {
    console.log('[ServerResultsLoader] Loading results from path:', path);
    
    // Reset UI state for new results source
    resetUiStateForNewSource('results');
    setIsLoadingResults(true);
    setResultsLoadingMessage('Loading results from server...');

    try {
      // Load data using the server API with initial limits to speed up loading
      // Load first 10k conversations and all properties for responsive initial load
      const data = await resultsLoad(path, {
        max_conversations: 10000,
        max_properties: undefined // Load all properties (they're smaller)
      });
      setResultsLoadingMessage('Parsing conversations, properties, and clusters...');
      console.log('[ServerResultsLoader] Loaded data:', data);

      const conversations: any[] = Array.isArray(data?.conversations) ? data.conversations : [];
      const properties: any[] = Array.isArray(data?.properties) ? data.properties : [];
      const clustersRaw: any[] = Array.isArray(data?.clusters) ? data.clusters : [];

      // Detect method from conversations
      const detectedMethod: 'single_model' | 'side_by_side' = conversations.some((c) => Array.isArray(c?.model)) ? 'side_by_side' : 'single_model';

      // Build operational rows from conversations (same logic as local loader)
      const operational: any[] = conversations.map((c, idx) => {
        const qid = String(c?.question_id ?? idx);
        const prompt = c?.prompt ?? '';
        const meta = (c?.meta && typeof c.meta === 'object') ? c.meta : {};
        const row: any = { __index: idx, question_id: qid, prompt };
        
        if (Array.isArray(c?.model)) {
          // side-by-side
          row.model_a = c.model?.[0] ?? 'model_a';
          row.model_b = c.model?.[1] ?? 'model_b';
          const [respA, respB] = pickPairResponses(c);
          row.model_a_response = respA ?? '';
          row.model_b_response = respB ?? '';
          const scores = c?.scores ?? [{}, {}];
          if (Array.isArray(scores)) {
            row.score_a = (scores?.[0] && typeof scores[0] === 'object') ? scores[0] : {};
            row.score_b = (scores?.[1] && typeof scores[1] === 'object') ? scores[1] : {};
          }
        } else {
          // single_model (mirror local results loader schema)
          row.model = c?.model ?? 'model';
          // Accept multiple common aliases for assistant response
          row.model_response = pickSingleResponse(c);
          const score = (c?.scores && typeof c.scores === 'object' && !Array.isArray(c.scores)) ? c.scores : (c?.score && typeof c.score === 'object' ? c.score : {});
          if (Object.keys(score).length > 0) (row as any).score = score;
        }
        
        return row;
      });

      console.log(`[ServerResultsLoader] ${operational.length} operational rows built`);
      setResultsLoadingMessage('Indexing and linking properties...');

      // Build a mapping from question_id to __index for property linking (mirror local loader)
      const qidToIndex = new Map<string, number>();
      operational.forEach((r: any) => { qidToIndex.set(String(r?.question_id), Number(r?.__index)); });
      // Prepare properties with row_index to aid metrics recompute
      const propsWithIndex = properties.map((p: any) => ({ ...p, row_index: qidToIndex.get(String(p?.question_id)) }));

      // Set state (and ensure flattened view for table)
      setOperationalRows(operational);
      const { rows: flattenedRows } = flattenScores(operational, detectedMethod);
      setCurrentRows(flattenedRows);
      setMethod(detectedMethod);

      // Parse properties
      const propMap = new Map<string, any[]>();
      properties.forEach((prop) => {
        const propType = prop?.property_type || 'unknown';
        if (!propMap.has(propType)) propMap.set(propType, []);
        propMap.get(propType)!.push(prop);
      });
      setPropertiesByKey(propMap);
      setPropertiesRows(properties);

      // Set clusters if available
      if (clustersRaw.length > 0) {
        setClusters(clustersRaw);
      }

      // Enrich clusters with metrics meta so plots and summaries render immediately (mirror local loader)
      try {
        setResultsLoadingMessage('Computing per-model proportions and quality...');
        const res = await recomputeClusterMetrics({
          clusters: clustersRaw,
          properties: propsWithIndex,
          operationalRows: operational,
        });
        if (res?.clusters) setClusters(res.clusters);
      } catch (err) {
        console.warn('[ServerResultsLoader] Failed to enrich clusters; showing raw clusters', err);
      }

      // Load metrics if available
      try {
        setResultsLoadingMessage('Loading metrics artifacts...');
        const metricsData = {
          model_cluster_scores: data.model_cluster_scores || [],
          cluster_scores: data.cluster_scores || [],
          model_scores: data.model_scores || []
        };
        setResultsMetrics(metricsData);
      } catch (e) {
        console.error('📊 Error loading metrics:', e);
        setResultsMetrics(null);
      }

      // Show notification if data was truncated
      if (data.conversations.length >= 10000) {
        console.warn('[ServerResultsLoader] Loaded first 10k conversations. Full dataset may be larger.');
      }

      console.log(`[ServerResultsLoader] Successfully loaded results from ${path}`);

      // Close sidebar when data is successfully loaded
      setSidebarExpanded(false);

    } catch (err) {
      console.error('[ServerResultsLoader] Failed to load results:', err);
      setResultsError(`Failed to load results: ${String((err as any)?.message || err)}`);
    }
    finally {
      setIsLoadingResults(false);
      setResultsLoadingMessage('');
    }
  }

  // New function to process data with flexible column mapping
  function processDataWithMapping(rows: Record<string, any>[], mapping: ColumnMapping) {
    setMethod(mapping.method);
    setColumnMapping(mapping);
    setFilterNotice(null);

    // Create operational data using user-specified columns (first build standardized + score dicts)
    const mappedRows = rows.map((row, index) => {
      // Keep processed rows minimal: only core identifiers, models, responses, and score dicts
      const opRow: Record<string, any> = { __index: index };

      // Ensure every operational row has a stable question_id used by backend joins
      // Match backend data_objects convention: use the row index as string
      opRow.question_id = String(index);
      
      // Map prompt column
      if (mapping.promptCol && row[mapping.promptCol] !== undefined) {
        opRow.prompt = row[mapping.promptCol];
      }
      
      // Map response columns
      if (mapping.method === 'single_model' && mapping.responseCols[0]) {
        opRow.model_response = row[mapping.responseCols[0]];
      } else if (mapping.method === 'side_by_side') {
        if (mapping.responseCols[0]) opRow.model_a_response = row[mapping.responseCols[0]];
        if (mapping.responseCols[1]) opRow.model_b_response = row[mapping.responseCols[1]];
      }
      
      // Map model columns
      if (mapping.method === 'single_model' && mapping.modelCols[0]) {
        opRow.model = row[mapping.modelCols[0]];
      } else if (mapping.method === 'side_by_side') {
        if (mapping.modelCols[0]) opRow.model_a = row[mapping.modelCols[0]];
        if (mapping.modelCols[1]) opRow.model_b = row[mapping.modelCols[1]];
      }
      
      // Build score dictionaries per selected columns (backend contract)
      if (mapping.scoreCols.length > 0) {
        const toNumber = (v: any): number | undefined => {
          if (typeof v === 'number') return Number.isFinite(v) ? v : undefined;
          if (typeof v === 'string' && v.trim() !== '') {
            const n = Number(v);
            return Number.isFinite(n) ? n : undefined;
          }
          return undefined;
        };
        
        const parseMaybeJsonDict = (v: any): Record<string, any> | null => {
          if (v && typeof v === 'object' && !Array.isArray(v)) return v as Record<string, any>;
          if (typeof v === 'string') {
            const s = v.trim();
            if (s.startsWith('{') && s.endsWith('}')) {
              try { return JSON.parse(s); } catch (_) { return null; }
            }
          }
          return null;
        };

        if (mapping.method === 'single_model') {
          const scoreDict: Record<string, number> = {};
          for (const col of mapping.scoreCols) {
            const raw = row[col];
            const asDict = parseMaybeJsonDict(raw);
            if (asDict) {
              for (const [k, v] of Object.entries(asDict)) {
                const num = toNumber(v);
                if (num !== undefined) scoreDict[k] = num;
              }
            } else {
              const key = col.replace(/^(score_)?/i, '').replace(/_?score$/i, '') || 'value';
              const num = toNumber(raw);
              if (num !== undefined) scoreDict[key] = num;
            }
          }
          if (Object.keys(scoreDict).length > 0) (opRow as any).score = scoreDict;
        } else {
          const scoreADict: Record<string, number> = {};
          const scoreBDict: Record<string, number> = {};
          for (const col of mapping.scoreCols) {
            const raw = row[col];
            const asDict = parseMaybeJsonDict(raw);
            const target = col.toLowerCase().includes('_b') ? scoreBDict
                          : col.toLowerCase().includes('_a') ? scoreADict
                          : null; // apply to both if null
            if (asDict) {
              for (const [k, v] of Object.entries(asDict)) {
                const num = toNumber(v);
                if (num !== undefined) {
                  if (target === scoreADict) scoreADict[k] = num;
                  else if (target === scoreBDict) scoreBDict[k] = num;
                  else { scoreADict[k] = num; scoreBDict[k] = num; }
                }
              }
            } else {
              const base = col.replace(/^(score_)?/i, '').replace(/_?score$/i, '').replace(/_a$/i, '').replace(/_b$/i, '') || 'value';
              const num = toNumber(raw);
              if (num !== undefined) {
                if (target === scoreADict) scoreADict[base] = num;
                else if (target === scoreBDict) scoreBDict[base] = num;
                else { scoreADict[base] = num; scoreBDict[base] = num; }
              }
            }
          }
          if (Object.keys(scoreADict).length > 0) (opRow as any).score_a = scoreADict;
          if (Object.keys(scoreBDict).length > 0) (opRow as any).score_b = scoreBDict;
        }
      }

      // Do not include any other columns to keep the operational data clean
      return opRow;
    });

    // Filter out rows where any selected score is missing
    let filteredCount = 0;
    const scoreCols = mapping.scoreCols || [];
    const hasScoresSelected = scoreCols.length > 0;
    const isMissing = (v: any) => v === undefined || v === null || (typeof v === 'string' && v.trim() === '') || (typeof v === 'number' && Number.isNaN(v)) || (v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length === 0);
    const rowsAfterFilter = !hasScoresSelected ? mappedRows : mappedRows.filter((_, idx) => {
      const original = rows[idx];
      const missingAny = scoreCols.some(col => isMissing(original[col]));
      if (missingAny) filteredCount += 1;
      return !missingAny;
    });

    if (hasScoresSelected && filteredCount > 0) {
      setFilterNotice(`Filtered out ${filteredCount} row(s) due to missing values in selected score columns: ${scoreCols.join(', ')}`);
    }

    // Keep operational rows with consolidated score objects (backend format)
    console.log('🔧 DEBUG: Setting operationalRows:', {
      count: rowsAfterFilter.length,
      sampleRow: rowsAfterFilter[0],
      method: mapping.method,
      hasScores: rowsAfterFilter[0]?.score || rowsAfterFilter[0]?.score_a || rowsAfterFilter[0]?.score_b
    });
    setOperationalRows(rowsAfterFilter);
    
    // Create display rows with flattened scores for UI table
    console.log('📊 DEBUG: Before flattening:', {
      inputRows: rowsAfterFilter.length,
      sampleInputRow: rowsAfterFilter[0],
      method: mapping.method
    });
    const { rows: flattenedRows, columns: flattenedColumns } = flattenScores(rowsAfterFilter, mapping.method);
    console.log('📊 DEBUG: After flattening:', {
      outputRows: flattenedRows.length,
      sampleOutputRow: flattenedRows[0],
      columns: flattenedColumns,
      scoreColumns: flattenedColumns.filter(c => c.startsWith('score_'))
    });
    setCurrentRows(flattenedRows);
  }

  // Handle column mapping changes from the selector
  const handleMappingChange = useCallback((mapping: ColumnMapping) => {
    if (mappingValid && originalRows.length > 0) {
      processDataWithMapping(originalRows, mapping);
      setShowColumnSelector(false);
      setSidebarExpanded(false); // Close sidebar when data is loaded
    }
  }, [mappingValid, originalRows]);

  // Handle validation changes from the selector
  const handleValidationChange = useCallback((isValid: boolean, errors: string[]) => {
    setMappingValid(isValid);
    setMappingErrors(errors);
  }, []);

  const onView = useCallback((row: Record<string, any>, preserveEvidence = false) => {
    if (method === "single_model") {
      const messages = ensureOpenAIFormat(String(row?.["prompt"] ?? ""), row?.["model_response"]);
      setSelectedTrace({ type: "single", messages });
    } else if (method === "side_by_side") {
      const prompt = String(row?.["prompt"] ?? "");
      const messagesA = ensureOpenAIFormat(prompt, row?.["model_a_response"]);
      const messagesB = ensureOpenAIFormat(prompt, row?.["model_b_response"]);
      setSelectedTrace({
        type: "sbs",
        messagesA,
        messagesB,
        modelA: String(row?.["model_a"] ?? "Model A"),
        modelB: String(row?.["model_b"] ?? "Model B"),
      });
    }
    setSelectedRow(row);
    setDrawerOpen(true);
    if (!preserveEvidence) {
    setSelectedEvidence(null);
    setEvidenceTargetModel(undefined);
      setSelectedProperty(null); // Clear property context when viewing from main table
    }
  }, [method]);

  const responseKeys = useMemo(() => 
    method === "single_model"
      ? ["model_response"]
      : method === "side_by_side"
        ? ["model_a_response", "model_b_response"]
        : [],
    [method]
  );

  // Keep clusters tab mounted after first visit to avoid re-mount plot cost
  React.useEffect(() => {
    if (activeTab === 'clusters' && !hasViewedClusters) setHasViewedClusters(true);
  }, [activeTab, hasViewedClusters]);

  // Get allowed columns from current (display) data with proper ordering
  const allowedColumns = useMemo(() => {
    if (currentRows.length === 0) return [];
    const allColumns = Object.keys(currentRows[0]);
    
    // Order: index → prompt → response columns → remaining
    const indexCol = allColumns.filter((c) => c === '__index');
    const promptFirst = allColumns.filter((c) => c === 'prompt');
    const resp = allColumns.filter((c) => responseKeys.includes(c));
    const remaining = allColumns.filter((c) => c !== '__index' && c !== 'prompt' && !responseKeys.includes(c));
    
    const result = [...indexCol, ...promptFirst, ...resp, ...remaining];
    console.log('🔗 DEBUG allowedColumns (from currentRows - FIXED):', {
      currentRowsCount: currentRows.length,
      allColumns,
      result,
      scoreColumns: allColumns.filter(c => c.startsWith('score_')),
      sampleCurrentRow: currentRows[0]
    });
    return result;
  }, [currentRows, responseKeys]);

  // -------- Data Operations Chain ---------
  const [operationChain, setOperationChain] = useState<DataOperation[]>([]);
  const [pendingColumn, setPendingColumn] = useState<string | null>(null);
  const [pendingValues, setPendingValues] = useState<string[]>([]);
  const [pendingNegated, setPendingNegated] = useState<boolean>(false);
  const [dataSearchQuery, setDataSearchQuery] = useState<string>('');
  
  // Legacy filter interface for compatibility
  type Filter = { column: string; values: string[]; negated: boolean };
  const filters: Filter[] = operationChain
    .filter(op => op.type === 'filter')
    .map(op => op as any);

  const categoricalColumns = useMemo(() => {
    if (operationalRows.length === 0) return [] as string[];
    const cols = new Set<string>();
    for (const c of allowedColumns) {
      // Skip index column - it's not categorical
      if (c === '__index') continue;
      const uniq = new Set(operationalRows.slice(0, 500).map(r => r?.[c])).size;
      if (uniq > 0 && uniq <= 50) cols.add(c);
    }
    return Array.from(cols);
  }, [operationalRows, allowedColumns]);

  const numericCols = useMemo(() => {
    if (currentRows.length === 0) return [] as string[];
    // Consider a column numeric if the first few non-null values can be parsed as finite numbers
    const sample = currentRows.slice(0, 25);
    const isNumericColumn = (col: string) => {
      let seen = 0;
      let numeric = 0;
      for (const r of sample) {
        const v = (r as any)[col];
        if (v === null || v === undefined || v === '') continue;
        seen += 1;
        const n = Number(v);
        if (Number.isFinite(n)) numeric += 1;
        if (seen >= 5) break; // small check window
      }
      return seen > 0 && numeric === seen;
    };
    return allowedColumns.filter(c => c === '__index' || isNumericColumn(c));
  }, [currentRows, allowedColumns]);

  const uniqueValuesFor = useMemo(() => {
    const cache = new Map<string, string[]>();
    return (col: string) => {
      if (cache.has(col)) {
        if (debugCacheHits) console.debug(`[Cache] uniqueValuesFor hit: ${col}`);
        return cache.get(col)!;
      }
      const s = new Set<string>();
      operationalRows.forEach(r => { const v = r?.[col]; if (v !== undefined && v !== null) s.add(String(v)); });
      const result = Array.from(s).sort();
      cache.set(col, result);
      if (debugCacheHits) console.debug(`[Cache] uniqueValuesFor miss: ${col} → computed ${result.length} values`);
      return result;
    };
  }, [operationalRows, debugCacheHits]);

  // Apply the entire operation chain to operational data
  const applyOperationChain = useCallback(async (operations: DataOperation[]) => {
    console.log('🔄 Applying operation chain:', operations);
    
    // Start from operational data (with score dicts)
    let opData = [...operationalRows];
    
    // Apply non-sort operations on operational data
    for (const operation of operations) {
      if (operation.type === 'filter') {
        const filterOp = operation as any;
        opData = opData.filter(row => {
          const rowValue = String(row[filterOp.column] || '');
          const matchesValues = filterOp.values.includes(rowValue);
          return filterOp.negated ? !matchesValues : matchesValues;
        });
        console.log(`🔍 Filter ${filterOp.column}: ${opData.length} rows`);
      } else if (operation.type === 'custom') {
        const customOp = operation as any;
        try {
          const res = await dfCustom({ rows: opData, code: customOp.code });
          if (res.error) {
            console.error('Custom operation failed:', res.error);
          } else {
            opData = res.rows || opData;
            console.log(`🐍 Custom code: ${opData.length} rows`);
          }
        } catch (e) {
          console.error('Custom operation error:', e);
        }
      }
    }
    
    // Always flatten for UI display
    const { rows: flattenedRows } = flattenScores(opData, method);
    let displayData = flattenedRows;
    
    // Apply sort operations on flattened data (so score_* columns sort correctly)
    const sortOp = operations.find(op => op.type === 'sort') as any | undefined;
    if (sortOp) {
      displayData = [...displayData].sort((a, b) => {
        let aVal = a[sortOp.column];
        let bVal = b[sortOp.column];
        
        if (aVal == null && bVal == null) return 0;
        if (aVal == null) return sortOp.direction === 'asc' ? 1 : -1;
        if (bVal == null) return sortOp.direction === 'asc' ? -1 : 1;
        
        const aNum = Number(aVal);
        const bNum = Number(bVal);
        if (!isNaN(aNum) && !isNaN(bNum)) {
          const diff = aNum - bNum;
          return sortOp.direction === 'asc' ? diff : -diff;
        } else {
          const comp = String(aVal).toLowerCase().localeCompare(String(bVal).toLowerCase());
          return sortOp.direction === 'asc' ? comp : -comp;
        }
      });
      console.log(`🔄 Sort ${sortOp.column} ${sortOp.direction}: ${displayData.length} rows`);
    }
    
    console.log(`Final result (flattened): ${displayData.length} rows`);
    setCurrentRows(displayData);
  }, [operationalRows, method]);

  // Legacy wrapper for backward compatibility
  const applyFilters = useCallback(async (newFilters: Filter[]) => {
    const filterOps = newFilters.map(f => createFilterOperation(f.column, f.values, f.negated));
    const nonFilterOps = operationChain.filter(op => op.type !== 'filter');
    const newChain = [...filterOps, ...nonFilterOps];
    setOperationChain(newChain);
    await applyOperationChain(newChain);
  }, [operationChain, applyOperationChain]);

  const resetAll = useCallback(() => {
    const { rows: flattened } = flattenScores(operationalRows, method);
    setCurrentRows(flattened);
    setOperationChain([]);
    setGroupBy(null);
    setGroupPreview([]);
    setExpandedGroup(null);
    setCustomCode("");
    setCustomError(null);
    setSortColumn(null);
    setSortDirection(null);
  }, [operationalRows, method]);

  // -------- GroupBy State ---------
  const [groupBy, setGroupBy] = useState<string | null>(null);
  const [groupPreview, setGroupPreview] = useState<{ value: any; count: number; means: Record<string, number> }[]>([]);
  const [expandedGroup, setExpandedGroup] = useState<any | null>(null); // reserved for grouped row details
  const [groupPage, setGroupPage] = useState<number>(1); // reserved for grouped row pagination
  const [groupRows, setGroupRows] = useState<Record<string, any>[]>([]); // reserved for grouped row details
  const [groupTotal, setGroupTotal] = useState<number>(0); // reserved for grouped row details
  const [groupPagination, setGroupPagination] = useState<Map<string, number>>(new Map());

  // -------- Sorting State ---------
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc' | null>(null);

  // Sort function
  const handleSort = useCallback((column: string) => {
    let newDirection: 'asc' | 'desc' | null = 'asc';
    
    if (sortColumn === column) {
      if (sortDirection === 'asc') {
        newDirection = 'desc';
      } else if (sortDirection === 'desc') {
        newDirection = null;
      }
    }
    
    setSortColumn(newDirection ? column : null);
    setSortDirection(newDirection);
    
    // Update operation chain
    const nonSortOps = operationChain.filter(op => op.type !== 'sort');
    const newChain = newDirection 
      ? [...nonSortOps, createSortOperation(column, newDirection)]
      : nonSortOps;
    
    setOperationChain(newChain);
    void applyOperationChain(newChain);
  }, [sortColumn, sortDirection, operationChain, applyOperationChain]);

  // Auto-select first row for extraction when data changes
  React.useEffect(() => {
    if (currentRows.length > 0 && !selectedRowForExtraction) {
      setSelectedRowForExtraction(currentRows[0]);
    }
  }, [currentRows, selectedRowForExtraction]);

  // Apply sorting + search to currentRows with performance optimization
  const sortedRows = useMemo(() => {
    console.log('📊 DEBUG sortedRows generation:', {
      currentRowsCount: currentRows.length,
      sampleCurrentRow: currentRows[0],
      currentRowKeys: currentRows[0] ? Object.keys(currentRows[0]) : [],
      scoreColumns: currentRows[0] ? Object.keys(currentRows[0]).filter(k => k.startsWith('score_')) : []
    });
    
    // Apply search filter first
    let filteredRows = currentRows;
    if (dataSearchQuery.trim()) {
      const query = dataSearchQuery.toLowerCase().trim();
      filteredRows = currentRows.filter(row => {
        // Search across all text columns
        return allowedColumns.some(col => {
          const value = row[col];
          return value != null && String(value).toLowerCase().includes(query);
        });
      });
    }
    
    if (!sortColumn || !sortDirection) {
      console.log('📊 DEBUG sortedRows result (no sorting):', {
        resultCount: filteredRows.length,
        sampleResult: filteredRows[0]
      });
      return filteredRows;
    }
    
    // Use faster array copy and optimize comparison
    const result = filteredRows.length > 5000 ? filteredRows : filteredRows.slice();
    
    // Pre-determine if column is numeric for better performance
    const isNumericColumn = filteredRows.length > 0 && 
      filteredRows.slice(0, 10).every(row => {
        const val = row[sortColumn];
        return val == null || !isNaN(Number(val));
      });
    
    // For very large datasets, use localeCompare only for short strings to reduce CPU
    result.sort((a, b) => {
      let aVal = a[sortColumn];
      let bVal = b[sortColumn];
      
      // Handle null/undefined values
      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return sortDirection === 'asc' ? 1 : -1;
      if (bVal == null) return sortDirection === 'asc' ? -1 : 1;
      
      if (isNumericColumn) {
        // Numeric comparison
        aVal = Number(aVal);
        bVal = Number(bVal);
        const diff = aVal - bVal;
        return sortDirection === 'asc' ? diff : -diff;
      } else {
        // String comparison with cached lowercase
        const aStr = String(aVal);
        const bStr = String(bVal);
        // Avoid lowercasing very long strings
        const aCmp = aStr.length < 256 ? aStr.toLowerCase() : aStr;
        const bCmp = bStr.length < 256 ? bStr.toLowerCase() : bStr;
        const comp = aCmp < bCmp ? -1 : aCmp > bCmp ? 1 : 0;
        return sortDirection === 'asc' ? comp : -comp;
      }
    });
    
    return result;
  }, [currentRows, sortColumn, sortDirection, dataSearchQuery, allowedColumns]);

  // Memoize expensive data overview calculations using sorted data
  const dataOverview = useMemo(() => {
    if (sortedRows.length === 0) return null;
    const uniquePrompts = new Set(sortedRows.map(r => r?.prompt)).size;
    let uniqueModels = 0;
    if (method === 'single_model') {
      uniqueModels = new Set(sortedRows.map(r => r?.model)).size;
    } else if (method === 'side_by_side') {
      uniqueModels = new Set([
        ...sortedRows.map(r => r?.model_a || ''),
        ...sortedRows.map(r => r?.model_b || '')
      ]).size;
    }
    return {
      rowCount: sortedRows.length.toLocaleString(),
      uniquePrompts: uniquePrompts.toLocaleString(),
      uniqueModels: uniqueModels.toLocaleString(),
    };
  }, [sortedRows, method]);

  // Truncated Cell component for grouped view
  const TruncatedCell = React.memo(function TruncatedCell({ text }: { text: string }) {
    const [expanded, setExpanded] = React.useState(false);
    const MAX_LEN = 200;
    if (!expanded && text.length > MAX_LEN) {
      return (
        <span>
          {text.slice(0, MAX_LEN)}…{' '}
          <Button size="small" variant="text" onClick={() => setExpanded(true)}>Expand</Button>
        </span>
      );
    }
    if (expanded && text.length > MAX_LEN) {
      return (
        <span>
          {text}{' '}
          <Button size="small" variant="text" onClick={() => setExpanded(false)}>Collapse</Button>
        </span>
      );
    }
    return <span>{text}</span>;
  });

  // -------- Custom Code ---------
  const [customCode, setCustomCode] = useState<string>("");
  const [customError, setCustomError] = useState<string | null>(null);
  
  const handleCustomCodeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setCustomCode(e.target.value);
  }, []);
  
  const runCustom = useCallback(async () => {
    if (!customCode.trim()) return;
    
    try {
      // Add custom operation to chain
      const customOp = createCustomCodeOperation(customCode);
      const newChain = [...operationChain, customOp];
      setOperationChain(newChain);
      setCustomError(null);
      await applyOperationChain(newChain);
      setCustomCode(""); // Clear after successful application
    } catch (e: any) {
      console.error('runCustom error:', e);
      setCustomError(String(e?.message || e));
    }
  }, [customCode, operationChain, applyOperationChain]);

  // Operation management callbacks
  const removeOperation = useCallback((operationId: string) => {
    const newChain = operationChain.filter(op => op.id !== operationId);
    setOperationChain(newChain);
    
    // Update UI state for removed operations
    const removedOp = operationChain.find(op => op.id === operationId);
    if (removedOp?.type === 'sort') {
      setSortColumn(null);
      setSortDirection(null);
    }
    
    void applyOperationChain(newChain);
  }, [operationChain, applyOperationChain]);

  // Legacy filter removal for backward compatibility
  const removeFilter = useCallback((index: number) => {
    const filterOps = operationChain.filter(op => op.type === 'filter');
    if (index < filterOps.length) {
      removeOperation(filterOps[index].id);
    }
  }, [operationChain, removeOperation]);

  const refreshGroupPreview = useCallback(async (by: string) => {
    console.log('🟡 refreshGroupPreview called with:', by);
    console.log('🟡 current rows length:', currentRows.length, 'numericCols:', numericCols);
    
    // Local-first groupby (like filters)
    const grouped = new Map<any, any[]>();
    currentRows.forEach(row => {
      const key = row[by];
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key)!.push(row);
    });
    
    const localGroups = Array.from(grouped.entries()).map(([value, rows]) => {
      const count = rows.length;
      const means: Record<string, number> = {};
      numericCols.forEach(col => {
        const nums = rows.map(r => Number(r[col])).filter(n => !isNaN(n));
        if (nums.length > 0) {
          means[col] = nums.reduce((sum, n) => sum + n, 0) / nums.length;
        }
      });
      return { value, count, means };
    });
    
    console.log('🟡 Local groupby result:', localGroups);
    setGroupPreview(localGroups);
    
    // Optional backend validation (fire-and-forget). We intentionally do NOT overwrite the local
    // preview to avoid flicker when backend lacks flattened numeric columns. Keep for logging only.
    try {
      const canValidate = operationalRows.length > 0 && Object.prototype.hasOwnProperty.call(operationalRows[0] || {}, by);
      if (canValidate) {
        void dfGroupPreview({ rows: operationalRows, by }).then((res) => {
          console.log('🟡 dfGroupPreview (ignored for UI) response:', res);
        }).catch((e) => {
          console.log('🟡 Backend validation failed (ignored):', e);
        });
      }
    } catch (_) {
      // no-op
    }
  }, [currentRows, operationalRows, numericCols]);

  // Memoized table content to keep hook order stable
  const tableContent = useMemo(() => {
    if (activeTab !== 'table') return null;
    if (operationalRows.length === 0) return null;
    if (groupBy && groupPreview.length > 0) {
      const groupedRowsMap = new Map<any, any[]>();
      sortedRows.forEach((row) => {
        const key = row[groupBy];
        if (!groupedRowsMap.has(key)) groupedRowsMap.set(key, []);
        groupedRowsMap.get(key)!.push(row);
      });
      return (
        <>
        {/* Benchmark Metrics Table (same style as Metrics tab) */}
        <DataTabBenchmarkTable operationalRows={operationalRows} method={method} />
        
        {/* Keep FilterBar visible in grouped mode */}
        <Box sx={{ mt: 2 }}>
        <FilterBar
          searchValue={dataSearchQuery}
          onSearchChange={setDataSearchQuery}
          searchPlaceholder="Search data..."
          categoricalColumns={categoricalColumns}
          pendingColumn={pendingColumn}
          pendingValues={pendingValues}
          pendingNegated={pendingNegated}
          onPendingColumnChange={(column) => { 
            setPendingColumn(column); 
            setPendingValues([]); 
            setPendingNegated(false); 
          }}
          onPendingValuesChange={setPendingValues}
          onPendingNegatedChange={setPendingNegated}
          onAddFilter={() => {
            if (!pendingColumn || pendingValues.length === 0) return;
            const next = [...filters, { column: pendingColumn, values: pendingValues, negated: pendingNegated }];
            setPendingColumn(null); 
            setPendingValues([]); 
            setPendingNegated(false);
            void applyFilters(next);
          }}
          filters={filters}
          onRemoveFilter={removeFilter}
          uniqueValuesFor={uniqueValuesFor}
          resultCount={sortedRows.length}
          resultLabel="rows"
          showGroupBy={true}
          groupByOptions={allowedColumns}
          groupByValue={groupBy}
          onGroupByChange={(v) => { 
            console.log('🔵 GroupBy onChange triggered with value:', v);
            setGroupBy(v); 
            setExpandedGroup(null); 
            setGroupRows([]); 
            if (v) {
              console.log('🔵 Calling refreshGroupPreview with:', v);
              refreshGroupPreview(v);
            } else {
              console.log('🔵 Clearing group preview');
              setGroupPreview([]);
            }
          }}
          showCustomCode={true}
          customCodeValue={customCode}
          onCustomCodeChange={(value) => handleCustomCodeChange({ target: { value } } as React.ChangeEvent<HTMLInputElement>)}
          onCustomCodeRun={runCustom}
          onReset={resetAll}
          customCodeError={customError}
        />
        </Box>
        <Box sx={{ border: '1px solid #E5E7EB', borderRadius: 2, overflow: 'auto', backgroundColor: '#FFFFFF' }}>
          <Box sx={{ backgroundColor: '#F3F4F6', p: 2, borderBottom: '1px solid #E5E7EB', position: 'sticky', top: 0, zIndex: 1 }}>
            <Box sx={{ display: 'grid', gridTemplateColumns: `auto 2fr 1fr 1fr repeat(${Math.max(allowedColumns.length - 3, 0)}, 1fr)`, gap: 2, alignItems: 'center', minWidth: 960 }}>
              <Box sx={{ width: 24 }} />
              {allowedColumns.map((col, index) => (
                <Box key={col} sx={{ gridColumn: index === 0 ? 'span 1' : undefined, display: 'flex', alignItems: 'center', gap: 0.5, cursor: 'pointer' }} onClick={() => handleSort(col)}>
                  <Typography variant="subtitle2" sx={{ color: '#374151', fontWeight: 700, fontSize: 12, letterSpacing: 0.4, textTransform: 'uppercase' }}>
                    {col === '__index' ? 'INDEX' :
                     col === 'prompt' ? 'PROMPT' :
                     col === 'model' ? 'MODEL' :
                     col === 'model_response' ? 'RESPONSE' :
                     col === 'model_a' ? 'MODEL A' :
                     col === 'model_b' ? 'MODEL B' :
                     col === 'model_a_response' ? 'RESPONSE A' :
                     col === 'model_b_response' ? 'RESPONSE B' :
                     col.toUpperCase()}
                  </Typography>
                  {sortColumn === col && sortDirection === 'asc' && <ArrowUpwardIcon sx={{ fontSize: 12, color: '#374151' }} />}
                  {sortColumn === col && sortDirection === 'desc' && <ArrowDownwardIcon sx={{ fontSize: 12, color: '#374151' }} />}
                </Box>
              ))}
            </Box>
          </Box>

          {Array.from(groupedRowsMap.entries()).map(([groupValue, rows]) => {
            const groupKey = String(groupValue);
            const currentPage = groupPagination.get(groupKey) || 1;
            const pageSize = 10;
            const paginatedRows = rows.slice((currentPage - 1) * pageSize, currentPage * pageSize);
            const totalPages = Math.ceil(rows.length / pageSize);
            const handlePageChange = (page: number) => {
              setGroupPagination(prev => new Map(prev).set(groupKey, page));
            };
            return (
              <Accordion key={groupKey} sx={{ '&:before': { display: 'none' }, boxShadow: 'none', border: 'none' }}>
                <AccordionSummary expandIcon={null} sx={{ backgroundColor: '#F9FAFB', borderBottom: '1px solid #E5E7EB', minHeight: 48, '&.Mui-expanded': { minHeight: 48 }, '& .MuiAccordionSummary-content': { margin: '12px 0' }, cursor: 'pointer', '&:hover': { backgroundColor: '#F3F4F6' } }}>
                  <Box sx={{ display: 'grid', gridTemplateColumns: `auto 2fr 1fr 1fr repeat(${Math.max(allowedColumns.length - 3, 0)}, 1fr)`, gap: 2, alignItems: 'center', width: '100%', minWidth: 960 }}>
                    <ExpandMoreIcon sx={{ fontSize: 20, color: '#6B7280' }} />
                    {/* Group label spans first three columns */}
                    <Box sx={{ gridColumn: '2 / span 3', display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                        {String(groupValue).length > 50 ? String(groupValue).slice(0, 50) + '...' : String(groupValue)}
                      </Typography>
                      <Typography variant="body2" sx={{ backgroundColor: '#E0E7FF', color: '#3730A3', px: 1.5, py: 0.25, borderRadius: 9999, fontSize: 11, fontWeight: 500, textAlign: 'center', minWidth: 20 }}>
                        {rows.length}
                      </Typography>
                    </Box>
                    {allowedColumns.slice(3).map((col) => {
                      if (responseKeys.includes(col)) return <Box key={col} />;
                      const groupStats = groupPreview.find(g => g.value === groupValue);
                      const mean = groupStats?.means[col];
                      if (typeof mean === 'number') {
                        return (
                          <Typography key={col} variant="body2" sx={{ color: '#6B7280', fontStyle: 'italic', fontSize: 13 }}>
                            avg: {mean.toFixed(decimalPrecision)}
                          </Typography>
                        );
                      }
                      return <Box key={col} />;
                    })}
                  </Box>
                </AccordionSummary>
                <AccordionDetails sx={{ p: 0 }}>
                  <Box>
                    {paginatedRows.map((row: any, idx: number) => (
                      <Box key={idx} sx={{ display: 'grid', gridTemplateColumns: `auto 2fr 1fr 1fr repeat(${Math.max(allowedColumns.length - 3, 0)}, 1fr)`, gap: 2, alignItems: 'center', p: 2, borderBottom: idx < paginatedRows.length - 1 ? '1px solid #E5E7EB' : 'none', minWidth: 960 }}>
                        <Box sx={{ width: 24 }} />
                        {/* First three columns render as-is */}
                        {allowedColumns.slice(0, 3).map(col => (
                          <Box key={col}>
                            {responseKeys.includes(col) ? (
                              <Button size="small" variant="text" color="secondary" startIcon={<VisibilityOutlinedIcon />} onClick={() => onView(row)} sx={{ fontWeight: 600 }}>
                                View
                              </Button>
                            ) : (
                              (() => {
                                const value = row[col];
                                const isNumeric = col === '__index' || (value !== null && value !== undefined && !isNaN(Number(value)) && value !== '');
                                const isPrompt = col === 'prompt';
                                return (
                                  <Typography variant="body2" sx={{ maxWidth: 200, textAlign: isNumeric ? 'center' : 'left' }}>
                                    {isPrompt ? (<FormattedCell text={String(value || '')} isPrompt={true} />) : (<TruncatedCell text={String(value || '')} />)}
                                  </Typography>
                                );
                              })()
                            )}
                          </Box>
                        ))}
                        {/* Remaining columns */}
                        {allowedColumns.slice(3).map(col => (
                          <Box key={col}>
                            {responseKeys.includes(col) ? (
                              <Button size="small" variant="text" color="secondary" startIcon={<VisibilityOutlinedIcon />} onClick={() => onView(row)} sx={{ fontWeight: 600 }}>
                                View
                              </Button>
                            ) : (
                              (() => {
                                const value = row[col];
                                const isNumeric = col === '__index' || (value !== null && value !== undefined && !isNaN(Number(value)) && value !== '');
                                const isPrompt = col === 'prompt';
                                return (
                                  <Typography variant="body2" sx={{ maxWidth: 200, textAlign: isNumeric ? 'center' : 'left' }}>
                                    {isPrompt ? (<FormattedCell text={String(value || '')} isPrompt={true} />) : (<TruncatedCell text={String(value || '')} />)}
                                  </Typography>
                                );
                              })()
                            )}
                          </Box>
                        ))}
                      </Box>
                    ))}
                    {totalPages > 1 && (
                      <Box sx={{ p: 2, display: 'flex', justifyContent: 'center', borderTop: '1px solid #E5E7EB' }}>
                        <Pagination count={totalPages} page={currentPage} onChange={(_, page) => handlePageChange(page)} size="small" />
                      </Box>
                    )}
                  </Box>
                </AccordionDetails>
              </Accordion>
            );
          })}
        </Box>
        </>
      );
    }

    // Normal flat table view when no groupBy
    return (
      <>
        {/* Benchmark Metrics Table (same style as Metrics tab) */}
        <DataTabBenchmarkTable operationalRows={operationalRows} method={method} />
        
        <Box sx={{ mt: 2 }}>
        <FilterBar
          searchValue={dataSearchQuery}
          onSearchChange={setDataSearchQuery}
          searchPlaceholder="Search data..."
          categoricalColumns={categoricalColumns}
          pendingColumn={pendingColumn}
          pendingValues={pendingValues}
          pendingNegated={pendingNegated}
          onPendingColumnChange={(column) => { 
            setPendingColumn(column); 
            setPendingValues([]); 
            setPendingNegated(false); 
          }}
          onPendingValuesChange={setPendingValues}
          onPendingNegatedChange={setPendingNegated}
          onAddFilter={() => {
            if (!pendingColumn || pendingValues.length === 0) return;
            const next = [...filters, { column: pendingColumn, values: pendingValues, negated: pendingNegated }];
            setPendingColumn(null); 
            setPendingValues([]); 
            setPendingNegated(false);
            void applyFilters(next);
          }}
          filters={filters}
          onRemoveFilter={removeFilter}
          uniqueValuesFor={uniqueValuesFor}
          resultCount={sortedRows.length}
          resultLabel="rows"
          showGroupBy={true}
          groupByOptions={allowedColumns}
          groupByValue={groupBy}
          onGroupByChange={(v) => { 
            console.log('🔵 GroupBy onChange triggered with value:', v);
            setGroupBy(v); 
            setExpandedGroup(null); 
            setGroupRows([]); 
            if (v) {
              console.log('🔵 Calling refreshGroupPreview with:', v);
              refreshGroupPreview(v);
            } else {
              console.log('🔵 Clearing group preview');
              setGroupPreview([]);
            }
          }}
          showCustomCode={true}
          customCodeValue={customCode}
          onCustomCodeChange={(value) => handleCustomCodeChange({ target: { value } } as React.ChangeEvent<HTMLInputElement>)}
          onCustomCodeRun={runCustom}
          onReset={resetAll}
          customCodeError={customError}
        />
        </Box>
      <DataTable
        rows={sortedRows}
        columns={allowedColumns}
        responseKeys={responseKeys}
        onView={onView}
        allowedColumns={allowedColumns}
        sortColumn={sortColumn}
        sortDirection={sortDirection}
        onSort={handleSort}
        decimalPrecision={decimalPrecision}
      />
      </>
    );
  }, [activeTab, operationalRows, groupBy, groupPreview, sortedRows, allowedColumns, responseKeys, onView, groupPagination, sortColumn, sortDirection, handleSort, dataSearchQuery, categoricalColumns, pendingColumn, pendingValues, pendingNegated, filters, removeFilter, uniqueValuesFor, refreshGroupPreview, customCode, handleCustomCodeChange, runCustom, resetAll, customError]);

  // Memoized properties content
  const propertiesContent = useMemo(() => {
    if (activeTab !== 'properties') return null;
    if (propertiesRows.length === 0) {
      return (
        <Box sx={{ p: 2, border: '1px solid #E5E7EB', borderRadius: 2, background: '#FFFFFF' }}>
          <Typography variant="body1" color="text.secondary">
            No properties available yet. Run "Extract on selected" or "Run on all traces" from the sidebar.
          </Typography>
        </Box>
      );
    }
    return (
      <PropertiesTab
        rows={propertiesRows}
        originalData={operationalRows}
        onOpenProperty={(prop) => {
          // Use operationalRows (with consolidated score objects) instead of currentRows (flattened)
          // Prefer direct index if present
          const idx = (prop as any).__index ?? (prop as any).row_index;
          let row: any | null = null;
          if (idx != null) {
            row = operationalRows.find(r => Number(r?.__index) === Number(idx)) || null;
          }
          if (!row) {
            // Fallback: match on question_id and model
            const qid = (prop as any).question_id;
            const modelName = String((prop as any).model || '');
            row = operationalRows.find(r => {
              const rq = r?.question_id;
              if (method === 'single_model') {
                return rq === qid && String(r?.model || '') === modelName;
              } else if (method === 'side_by_side') {
                return rq === qid && (String(r?.model_a || '') === modelName || String(r?.model_b || '') === modelName);
              }
              return false;
            }) || null;
          }
          
          if (!row) {
            console.warn('[App] Could not locate row for property', { prop, idx, method });
          } else {
            // Debug: Show ALL columns of the matched operational dataframe row
            console.log('[App] OPERATIONAL DATAFRAME ROW - All columns:', row);
            console.log('[App] OPERATIONAL DATAFRAME ROW - Column names:', Object.keys(row));
          }
          
          // Process evidence
          const rawEvidence = (prop as any).evidence;
          const ev = Array.isArray(rawEvidence)
            ? rawEvidence
            : rawEvidence ? [rawEvidence] : [];
          
          setSelectedEvidence(ev);
          setEvidenceTargetModel((prop as any).model);
          setSelectedProperty(prop);
          
          if (row) {
            onView(row, true);
          }
        }}
      />
    );
  }, [activeTab, propertiesRows, currentRows, operationalRows, method, onView]);



  // reserved: server-validated grouped rows loader (not used in current UI)



  const clearCustomCode = useCallback(() => {
    setCustomCode("");
    setCustomError(null);
  }, []);

  // Memoized callbacks to prevent unnecessary effect triggers in children
  const getPropertiesRowsCb = useCallback(() => propertiesRows, [propertiesRows]);
  const getOperationalRowsCb = useCallback(() => operationalRows, [operationalRows]);
  const onRequestRecomputeCb = useCallback((included_property_ids?: string[]) => {
    (async () => {
      try {
        const res = await recomputeClusterMetrics({
          clusters,
          properties: propertiesRows,
          operationalRows,
          included_property_ids,
        });
        setClusters(res.clusters || []);
      } catch (e) {
        console.error('recompute (filters) failed', e);
      }
    })();
  }, [clusters, propertiesRows, operationalRows]);

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {resultsError && (
        <Box sx={{ position: 'fixed', top: 12, left: '50%', transform: 'translateX(-50%)', zIndex: 1500 }}>
          <Box sx={{
            px: 2, py: 1, borderRadius: 1, border: '1px solid', borderColor: 'error.light',
            backgroundColor: '#FEF2F2', color: '#7F1D1D', display: 'flex', alignItems: 'center', gap: 2
          }}>
            <Typography variant="body2" sx={{ fontWeight: 700 }}>Error</Typography>
            <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>{resultsError}</Typography>
            <Button size="small" variant="outlined" onClick={() => setResultsError(null)} sx={{ ml: 'auto' }}>Dismiss</Button>
          </Box>
        </Box>
      )}
      <AppBar position="fixed">
        <Toolbar sx={{ gap: 2 }}>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>StringSight (or maybe Squiglit)</Typography>
          <Stack direction="row" spacing={1} alignItems="center">
            {isLoadingResults && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: 320 }}>
                <LinearProgress sx={{ flexGrow: 1 }} />
                <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.9)' }}>
                  {resultsLoadingMessage || 'Loading...'}
                </Typography>
              </Box>
            )}
            <Button 
              variant="contained" 
              color="primary"
              onClick={() => setShowFileBrowser(true)}
            >
              Load File
            </Button>
            <Button 
              variant="outlined" 
              color="inherit"
              onClick={() => setShowFolderBrowser(true)}
            >
              Browse Results Folder
            </Button>
            {availableColumns.length > 0 && !showColumnSelector && (
              <Button 
                variant="outlined" 
                onClick={() => setShowColumnSelector(true)}
                size="small"
              >
                Configure Columns
              </Button>
            )}
            <Button 
              variant={debugCacheHits ? "contained" : "outlined"}
              color={debugCacheHits ? "secondary" : "inherit"}
              onClick={() => setDebugCacheHits(v => !v)}
              size="small"
            >
              {debugCacheHits ? 'Debug Cache: On' : 'Debug Cache: Off'}
            </Button>
          </Stack>
        </Toolbar>
      </AppBar>
      {/* offset for fixed AppBar */}
      <Box sx={{ height: (theme) => theme.mixins.toolbar.minHeight }} />
      
      {/* Permanent Icon Sidebar */}
          <PermanentIconSidebar 
        activeSection={activeSection} 
        onSectionChange={(section) => {
          setActiveSection(section);
          setSidebarExpanded(true);
              if (section === 'metrics') {
                // Stay within metrics family; default to Metrics view
                setActiveTab('metrics');
              }
        }} 
      />

      {/* Removed expand chevron button; opening is handled by clicking icon sidebar */}

      {/* Expanded Sidebar */}
      <ExpandedSidebar
        activeSection={activeSection}
        expanded={sidebarExpanded}
        onToggleExpanded={() => setSidebarExpanded(!sidebarExpanded)}
      >
        {activeSection === 'data' && (
          <DataStatsPanel 
            dataOverview={dataOverview} 
            method={method} 
            operationalRows={operationalRows}
            decimalPrecision={decimalPrecision}
            onDecimalPrecisionChange={setDecimalPrecision}
          />
        )}
        {activeSection === 'extraction' && (
          <Box sx={{ position: 'relative' }}>
          <PropertyExtractionPanel
            method={method}
            getSelectedRow={() => {
              // Prioritize the row being viewed in the trace drawer, otherwise use the default selection
              return (drawerOpen && selectedRow) ? selectedRow : selectedRowForExtraction;
            }}
            getAllRows={() => currentRows}
            onPropertiesMerged={(props) => {
              const newMap = new Map(propertiesByKey);
              props.forEach(p => {
                const key = `${p.question_id}-${p.model}`;
                if (!newMap.has(key)) newMap.set(key, []);
                newMap.get(key)!.push(p);
              });
              setPropertiesByKey(newMap);
              
              // Also update propertiesRows for the PropertiesTab
              // Enrich new properties with model_response from operational data
              const enrichedProps = props.map(prop => {
                // Find matching operational row by question_id and model
                const matchingRow = operationalRows.find(opRow => {
                  if (method === 'single_model') {
                    return opRow.question_id === prop.question_id && opRow.model === prop.model;
                  } else if (method === 'side_by_side') {
                    return opRow.question_id === prop.question_id && 
                           (opRow.model_a === prop.model || opRow.model_b === prop.model);
                  }
                  return false;
                });
                
                // Add model_response from the matching operational row
                return {
                  ...prop,
                  model_response: matchingRow?.model_response || 
                                 matchingRow?.model_a_response || 
                                 matchingRow?.model_b_response || 
                                 'No response found'
                };
              });
              
              // Add to existing propertiesRows (for single extraction, we append)
              setPropertiesRows(prevRows => [...prevRows, ...enrichedProps]);
              setActiveTab('properties'); // Switch to properties tab to see results
            }}
            onSelectEvidence={setSelectedEvidence}
            onBatchLoaded={(rows) => {
              // Enrich properties with model_response from operational data
              const enrichedRows = rows.map(prop => {
                // Find matching operational row by question_id and model
                const matchingRow = operationalRows.find(opRow => {
                  if (method === 'single_model') {
                    return opRow.question_id === prop.question_id && opRow.model === prop.model;
                  } else if (method === 'side_by_side') {
                    return opRow.question_id === prop.question_id && 
                           (opRow.model_a === prop.model || opRow.model_b === prop.model);
                  }
                  return false;
                });
                
                // Add model_response from the matching operational row
                return {
                  ...prop,
                  model_response: matchingRow?.model_response || 
                                 matchingRow?.model_a_response || 
                                 matchingRow?.model_b_response || 
                                 'No response found'
                };
              });
              
              setPropertiesRows(enrichedRows);
              setActiveTab('properties'); // Switch to properties tab to see results
            }}
            onBatchStart={() => setBatchRunning(true)}
            onBatchStatus={(progress, state) => {
              setBatchProgress(progress);
              setBatchState(state);
            }}
            onBatchDone={() => setBatchRunning(false)}
            onOpenTrace={(row) => {
              // Format trace data properly based on method (same as onView function)
              if (method === "single_model") {
                const messages = ensureOpenAIFormat(String(row?.["prompt"] ?? ""), row?.["model_response"]);
                setSelectedTrace({ type: "single", messages });
              } else if (method === "side_by_side") {
                const prompt = String(row?.["prompt"] ?? "");
                const messagesA = ensureOpenAIFormat(prompt, row?.["model_a_response"]);
                const messagesB = ensureOpenAIFormat(prompt, row?.["model_b_response"]);
                setSelectedTrace({
                  type: "sbs",
                  messagesA,
                  messagesB,
                  modelA: String(row?.["model_a"] ?? "Model A"),
                  modelB: String(row?.["model_b"] ?? "Model B"),
                });
              }
              setSelectedRow(row);
              setDrawerOpen(true);
              setSelectedEvidence(null);
              setEvidenceTargetModel(undefined);
            }}
            onCloseTrace={() => {
              setDrawerOpen(false);
              setSelectedTrace(null);
              setSelectedRow(null);
              setSelectedEvidence(null);
              setEvidenceTargetModel(undefined);
              setSelectedProperty(null);
            }}
          />
          {isResultsMode && (
            <Box sx={{ position: 'absolute', inset: 0, zIndex: (theme) => theme.zIndex.modal + 1, bgcolor: 'rgba(255,255,255,0.75)', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 1, pointerEvents: 'all' }}>
              <Box sx={{ bgcolor: '#F97316', color: '#FFFFFF', px: 2, py: 1.25, borderRadius: 1, boxShadow: 4, border: '1px solid #EA580C', textAlign: 'center' }}>
                <Typography variant="body2" sx={{ fontWeight: 700 }}>
                  Extraction disabled in results mode. Upload raw data to enable extraction.
                </Typography>
              </Box>
            </Box>
          )}
          </Box>
        )}
        {activeSection === 'clustering' && (
          <Box sx={{ position: 'relative' }}>
          <ClusteringPanel 
            hasAnyProperties={propertiesRows.length > 0}
            getOperationalRows={getOperationalRowsCb}
            getPropertiesRows={getPropertiesRowsCb}
            onClustersUpdated={(data) => {
              setClusters(data.clusters || []);
              setTotalConversationsByModel(data.total_conversations_by_model || null);
              setTotalUniqueConversations(data.total_unique_conversations || null);
            }}
          />
          {isResultsMode && (
            <Box sx={{ position: 'absolute', inset: 0, zIndex: (theme) => theme.zIndex.modal + 1, bgcolor: 'rgba(255,255,255,0.75)', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 1, pointerEvents: 'all' }}>
              <Box sx={{ bgcolor: '#F97316', color: '#FFFFFF', px: 2, py: 1.25, borderRadius: 1, boxShadow: 4, border: '1px solid #EA580C', textAlign: 'center' }}>
                <Typography variant="body2" sx={{ fontWeight: 700 }}>
                  Clustering disabled in results mode. Upload raw data to re-cluster.
                </Typography>
              </Box>
            </Box>
          )}
          </Box>
        )}
        {activeSection === 'metrics' && (
          <MetricsPanel
            filters={metricsFilters}
            onFiltersChange={setMetricsFilters}
            availableModels={metricsAvailableModels}
            availableGroups={metricsAvailableGroups}
            availableQualityMetrics={metricsAvailableQualityMetrics}
            summary={metricsSummary || undefined}
          />
        )}
      </ExpandedSidebar>

            <Container maxWidth={false} sx={{ 
        py: 2, 
        flexGrow: 1, 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'stretch', 
        ml: `${60 + (sidebarExpanded ? 400 : 0)}px`, // Account for both sidebars
        mr: 0, // No right margin
        width: `calc(100vw - ${60 + (sidebarExpanded ? 400 : 0)}px)`, // Constrain width to prevent cutoff
        maxWidth: 'none', // Override default maxWidth
        transition: 'margin-left 450ms ease, width 450ms ease',
        overflow: 'auto' // Ensure proper scroll containment
      }}>
        {/* Left control sidebar is always available (collapsed by default via width + Drawer) */}
        {/* Getting started helper - shown before any upload */}
        {originalRows.length === 0 && !showColumnSelector && !isResultsMode && (
          <Box sx={{ 
            mb: 2, py: 3, px: 2, borderRadius: 2,
            background: '#F8FAFC', color: 'text.secondary'
          }}>
            <Box sx={{ textAlign: 'left', maxWidth: 760 }}>
              <Typography variant="h6" sx={{ mb: 1, color: 'primary.dark' }}>Easily Visualize and Analyze your Model Outputs</Typography>
              <Typography variant="body2" sx={{ color: 'primary.dark' }}>1) Upload your dataset (.jsonl, .json, or .csv)</Typography>
              <Typography variant="body2" sx={{ color: 'primary.dark' }}>2) Select which columns correspond to your prompts, responses, models, and scores</Typography>
              <Typography variant="body2" sx={{ color: 'primary.dark' }}>3) Click Done to load your table and explore</Typography>
            </Box>
          </Box>
        )}
        
        {/* Column Selector - shown when user needs to specify column mapping */}
        {showColumnSelector && (
          <ColumnSelector
            columns={availableColumns}
            onMappingChange={handleMappingChange}
            onValidationChange={handleValidationChange}
            autoDetectedMapping={autoDetectedMapping || undefined}
          />
        )}

        {/* Show filter notice if any rows were dropped due to missing scores */}
        {filterNotice && (
          <Box sx={{ mb: 1, p: 1.5, border: '1px solid #F59E0B', background: '#FFFBEB', color: '#92400E', borderRadius: 1 }}>
            {filterNotice}
          </Box>
        )}

        {dataOverview && (
          <Box sx={{ 
            mb: 2, 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 3
          }}>
            <Box sx={{ color: 'text.secondary' }}>
              <strong>{dataOverview.rowCount}</strong> rows ·{' '}
              <strong>{dataOverview.uniquePrompts}</strong> unique prompts ·{' '}
              <strong>{dataOverview.uniqueModels}</strong> unique models
            </Box>
            {/* Removed hint: Click headers to sort • Use filters to narrow results */}
          </Box>
        )}



        {/* Operation Chain Summary */}
        <FilterSummary
          operations={operationChain}
          onRemoveOperation={removeOperation}
        />

        {/* Tabs for switching between Data, Properties, and Clusters */}
        <Box sx={{ mb: 1 }}>
          <Tabs
            value={activeTab}
            onChange={(_, v) => {
              setActiveTab(v);
              if (v === 'metrics') setActiveSection('metrics');
            }}
            textColor="primary"
            indicatorColor="primary"
          >
            <Tab value="table" label="Data" />
            <Tab value="properties" label={`Properties${propertiesRows.length ? ` (${propertiesRows.length})` : ''}`} />
            <Tab value="clusters" label={`Clusters${clusters.length ? ` (${clusters.length})` : ''}`} />
            <Tab value="metrics" label="Metrics" />
            <Tab value="test" label="Card Test" />
          </Tabs>
        </Box>

        {/* Content based on active tab */}
        {/* Mount table/properties only when active to keep memory low */}
        {activeTab === 'table' ? tableContent : null}
        {activeTab === 'properties' ? propertiesContent : null}
        {(activeTab === 'clusters' || hasViewedClusters) && (
          <Box sx={{ display: activeTab === 'clusters' ? 'block' : 'none' }}>
            <ClustersTab 
              clusters={clusters}
              totalConversationsByModel={totalConversationsByModel}
              totalUniqueConversations={totalUniqueConversations}
              getPropertiesRows={getPropertiesRowsCb}
              onRequestRecompute={onRequestRecomputeCb}
              onOpenPropertyById={(pid) => {
              // Find property row in propertiesRows and open in the right drawer
              const prop = propertiesRows.find((p: any) => String(p.id) === String(pid));
              if (!prop) return;
              
              const idx = (prop as any).__index ?? (prop as any).row_index;
              let row: any | null = null;
              if (idx != null) {
                row = operationalRows.find(r => Number(r?.__index) === Number(idx)) || null;
              }
              if (!row) {
                const qid = (prop as any).question_id;
                const modelName = String((prop as any).model || '');
                row = operationalRows.find(r => {
                  const rq = r?.question_id;
                  if (method === 'single_model') {
                    return rq === qid && String(r?.model || '') === modelName;
                  } else if (method === 'side_by_side') {
                    return rq === qid && (String(r?.model_a || '') === modelName || String(r?.model_b || '') === modelName);
                  }
                  return false;
                }) || null;
              }
              
              if (!row) {
                console.warn('[App] Could not locate row for property', { prop, idx, method });
              }
              
              // Process evidence (same logic as PropertiesTab)
              const rawEvidence = (prop as any).evidence;
              const ev = Array.isArray(rawEvidence)
                ? rawEvidence
                : rawEvidence ? [rawEvidence] : [];
              
              setSelectedEvidence(ev);
              setEvidenceTargetModel((prop as any).model);
              setSelectedProperty(prop);
              
              if (row) {
                onView(row, true);
              }
            }}
            />
          </Box>
        )}
        {activeTab === 'test' ? <CardTestPage /> : null}
        {activeTab === 'metrics' && (
          <Box sx={{ mt: 1 }}>
            {resultsMetrics ? (
              <MetricsTab
                resultsData={resultsMetrics}
                filters={metricsFilters}
                onDataProcessed={(data) => {
                  setMetricsAvailableModels(data.availableModels);
                  setMetricsAvailableGroups(data.availableGroups);
                  setMetricsAvailableQualityMetrics(data.availableQualityMetrics);
                  setMetricsSummary(data.summary);
                  
                  // Auto-select defaults if not set
                  setMetricsFilters(prev => {
                    const updates: Partial<MetricsFilters> = {};
                    
                    if (!prev.qualityMetric && data.availableQualityMetrics.length > 0) {
                      updates.qualityMetric = data.availableQualityMetrics[0];
                    }
                    
                    if (prev.selectedModels.length === 0 && data.availableModels.length > 0) {
                      updates.selectedModels = data.availableModels;
                    }
                    
                    return Object.keys(updates).length > 0 ? { ...prev, ...updates } : prev;
                  });
                }}
                debug={true}
                showBenchmark={true}
                showClusterPlots={true}
                showModelCards={true}
              />
            ) : (
              <Box sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  No Metrics Data Available
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Load results from a folder that contains computed metrics, or run the clustering pipeline to generate metrics.
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </Container>



      <Drawer anchor="right" open={drawerOpen} variant="persistent" sx={{ 
        '& .MuiDrawer-paper': { 
          width: '50vw', 
          maxWidth: 900, 
          p: 2,
          overflow: 'auto',
          height: '100vh'
        } 
      }} ModalProps={{ keepMounted: true }}>
        <>
            {(selectedTrace?.type === "single" || selectedTrace?.type === "sbs") && (
              <>
                {selectedProperty ? (
                  // Enhanced header when viewing from properties table
                  <PropertyTraceHeader
                    selectedRow={selectedRow}
                    selectedProperty={selectedProperty}
                    method={method}
                    evidenceTargetModel={evidenceTargetModel}
                  />
                ) : (
                  // Standard header when viewing from main data table
              <Box sx={{ mb: 1.5 }}>
                <Typography variant="subtitle2" sx={{ color: '#334155' }}>
                  {selectedTrace?.type === 'single' ? (String((selectedRow as any)?.model || '')) : `${String((selectedRow as any)?.model_a || '')} vs ${String((selectedRow as any)?.model_b || '')}`}
                </Typography>
                {/* Compact score chips if present */}
                {(() => {
                  let entries: [string, any][] = [];
                  if (method === 'single_model') {
                    const scores = selectedRow?.score || null;
                    entries = scores && typeof scores === 'object' ? Object.entries(scores as Record<string, number>) : [];
                  } else if (method === 'side_by_side') {
                    // Show scores for the model whose evidence is targeted, else both
                    const isA = evidenceTargetModel && String(selectedRow?.model_a || '') === String(evidenceTargetModel);
                    const isB = evidenceTargetModel && String(selectedRow?.model_b || '') === String(evidenceTargetModel);
                    const sa = (selectedRow as any)?.score_a || {};
                    const sb = (selectedRow as any)?.score_b || {};
                    const chosen = isA ? sa : isB ? sb : sa; // default A
                    entries = Object.entries(chosen);
                  }
                  if (!entries.length) return null;
                  return (
                    <Stack direction="row" spacing={0.5} sx={{ mt: 0.5, flexWrap: 'wrap' }}>
                      {entries.map(([k, v]) => (
                        <Box key={k} sx={{ px: 0.75, py: 0.25, border: '1px solid #E5E7EB', borderRadius: 9999, fontSize: 12, color: '#334155', background: '#F8FAFC' }}>
                          {k}: {typeof v === 'number' ? v.toFixed(decimalPrecision) : String(v)}
                        </Box>
                      ))}
                    </Stack>
                  );
                })()}
              </Box>
                )}
              </>
            )}
            {selectedTrace?.type === "single" && (
              <ConversationTrace
                messages={selectedTrace.messages}
                highlights={selectedEvidence || undefined}
                rawResponse={selectedRow?.model_response}
              />
            )}
            {selectedTrace?.type === "sbs" && (
              <SideBySideTrace
                messagesA={selectedTrace.messagesA}
                messagesB={selectedTrace.messagesB}
                modelA={selectedTrace.modelA}
                modelB={selectedTrace.modelB}
                highlights={selectedEvidence || undefined}
                targetModel={evidenceTargetModel}
                rawResponseA={selectedRow?.model_a_response}
                rawResponseB={selectedRow?.model_b_response}
              />
            )}
          </>
      </Drawer>
      {/* Transparent center overlay to close only the right drawer when clicked */}
      {drawerOpen && (
        <Box
          onClick={() => {
            setDrawerOpen(false);
            setSelectedProperty(null);
          }}
          sx={{
            position: 'fixed',
            left: `${60 + (sidebarExpanded ? 400 : 0)}px`,
            right: '50vw',
            top: (theme) => theme.mixins.toolbar.minHeight,
            bottom: 0,
            zIndex: (theme) => theme.zIndex.drawer - 1,
            background: 'transparent',
          }}
        />
      )}

      {/* Server-side folder browser dialog */}
      <ServerFolderBrowser
        open={showFolderBrowser}
        onClose={() => setShowFolderBrowser(false)}
        onSelectPath={onServerPathSelected}
        title="Browse Results Folder"
        acceptedExtensions={['.json', '.jsonl']}
      />

      {/* Server-side file browser dialog */}
      <ServerFileBrowser
        open={showFileBrowser}
        onClose={() => setShowFileBrowser(false)}
        onSelectPath={onServerFileSelected}
        title="Select File from Server"
        acceptedExtensions={['.json', '.jsonl', '.csv']}
      />
    </Box>
  );
}

export default function AppWithErrorBoundary() {
  return (
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  );
}
