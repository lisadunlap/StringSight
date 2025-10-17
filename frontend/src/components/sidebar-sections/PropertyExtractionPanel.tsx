import React, { useRef } from 'react';
import { 
  Box, 
  Stack, 
  Typography, 
  TextField, 
  Button, 
  Autocomplete, 
  Accordion, 
  AccordionSummary, 
  AccordionDetails, 
  LinearProgress 
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { getPrompts, getPromptText, extractSingle, extractJobStart, extractJobStatus, extractJobResult, extractJobCancel } from '../../lib/api';

type Method = 'single_model' | 'side_by_side' | 'unknown';

interface PropertyExtractionPanelProps {
  method: Method;
  getSelectedRow: () => Record<string, any> | null;
  getAllRows: () => Record<string, any>[];
  onPropertiesMerged: (props: any[]) => void;
  onSelectEvidence: (evidence: string[], targetModel?: string) => void;
  onBatchLoaded: (rows: any[]) => void;
  onBatchStart?: () => void;
  onBatchStatus?: (progress: number, state: string | null) => void;
  onBatchDone?: () => void;
  onOpenTrace?: (row: Record<string, any>) => void; // Add callback to open trace viewer
  onCloseTrace?: () => void; // Add callback to close trace viewer
}

export default function PropertyExtractionPanel({
  method,
  getSelectedRow,
  getAllRows,
  onPropertiesMerged,
  onSelectEvidence,
  onBatchLoaded,
  onBatchStart,
  onBatchStatus,
  onBatchDone,
  onOpenTrace,
  onCloseTrace,
}: PropertyExtractionPanelProps) {
  const resultsRef = useRef<HTMLDivElement>(null);
  const [promptOptions, setPromptOptions] = React.useState<{ name: string; label: string; has_task_description: boolean; preview: string; default_task_description_single?: string | null; default_task_description_sbs?: string | null; }[]>([]);
  const [selectedPrompt, setSelectedPrompt] = React.useState<string>(
    () => localStorage.getItem('stringsight.selectedPrompt') || 'default'
  );
  const [taskDescription, setTaskDescription] = React.useState<string>(
    () => localStorage.getItem('stringsight.taskDescription') || ''
  );
  const [userEdited, setUserEdited] = React.useState<boolean>(
    () => (localStorage.getItem('stringsight.taskDescriptionEdited') === 'true') || false
  );
  const [resolvedPrompt, setResolvedPrompt] = React.useState<string>('');

  const [modelName, setModelName] = React.useState<string>('gpt-4o-mini');
  const [temperature, setTemperature] = React.useState<number>(0.6);
  const [topP, setTopP] = React.useState<number>(0.95);
  const [maxTokens, setMaxTokens] = React.useState<number>(2048);
  const [maxWorkers, setMaxWorkers] = React.useState<number>(16);
  const [sampleSize, setSampleSize] = React.useState<number | null>(null);

  const [busy, setBusy] = React.useState<boolean>(false);
  const [lastExtractProps, setLastExtractProps] = React.useState<any[]>([]);
  const [errorMsg, setErrorMsg] = React.useState<string | null>(null);

  const [jobId, setJobId] = React.useState<string>('');
  const [jobProgress, setJobProgress] = React.useState<number>(0);
  const [jobState, setJobState] = React.useState<string | null>(null);

  const selectedPromptMeta = promptOptions.find(p => p.name === selectedPrompt);
  const canTaskDescribe = selectedPromptMeta?.has_task_description || false;

  // Load prompts on mount
  React.useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const res = await getPrompts();
        const allowed = new Set(['default', 'agent']);
        const filtered = (res.prompts || []).filter((p: any) => allowed.has(p.name));
        if (mounted) setPromptOptions(filtered);
      } catch (e: any) {
        if (mounted) setErrorMsg(`Failed to load prompts: ${String(e?.message || e)}`);
      }
    })();
    return () => { mounted = false; };
  }, []);

  // When promptOptions load or selectedPrompt changes, prefill task description with default if supported and user hasn't edited.
  React.useEffect(() => {
    if (!selectedPromptMeta) return;
    // Persist selected prompt
    localStorage.setItem('stringsight.selectedPrompt', selectedPrompt);
    // If prompt supports task description
    if (selectedPromptMeta.has_task_description) {
      const defaultDesc = method === 'side_by_side' ? (selectedPromptMeta.default_task_description_sbs || '') : (selectedPromptMeta.default_task_description_single || '');
      if (!userEdited) {
        setTaskDescription(defaultDesc);
        localStorage.setItem('stringsight.taskDescription', defaultDesc);
        localStorage.setItem('stringsight.taskDescriptionEdited', 'false');
      }
    } else {
      // Clear task description for prompts that don't support it
      setTaskDescription('');
      localStorage.setItem('stringsight.taskDescription', '');
      localStorage.setItem('stringsight.taskDescriptionEdited', 'false');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPromptMeta?.name, selectedPromptMeta?.has_task_description]);

  // Resolve prompt text when selection changes
  React.useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const effectiveTaskDesc = canTaskDescribe && taskDescription.trim().length > 0 ? taskDescription : undefined;
        const methodParam = method === 'unknown' ? undefined : method;
        const res = await getPromptText(selectedPrompt, effectiveTaskDesc, methodParam);
        if (mounted) setResolvedPrompt(res.text);
      } catch (e: any) {
        if (mounted) {
          setResolvedPrompt('');
          setErrorMsg(`Failed to load prompt '${selectedPrompt}': ${String(e?.message || e)}`);
        }
      }
    })();
    return () => { mounted = false; };
  }, [selectedPrompt, canTaskDescribe, taskDescription]);

  // Highlight task description inside resolved prompt (visual only)
  const highlightedResolvedPrompt = React.useMemo(() => {
    const text = resolvedPrompt || '';
    const needle = canTaskDescribe && taskDescription.trim().length > 0 ? taskDescription.trim() : '';
    if (!needle) return [text];
    const parts = text.split(needle);
    const nodes: React.ReactNode[] = [];
    parts.forEach((part, idx) => {
      nodes.push(part);
      if (idx < parts.length - 1) {
        nodes.push(
          <span key={`hl-${idx}`} style={{ color: '#1D4ED8', fontWeight: 600 }}>{needle}</span>
        );
      }
    });
    return nodes;
  }, [resolvedPrompt, canTaskDescribe, taskDescription]);

  async function runExtractSingle() {
    const row = getSelectedRow();
    const methodValid = method === 'single_model' || method === 'side_by_side';
    console.log('[PropertyExtraction] runExtractSingle called', { row, methodValid, method });
    if (!row || !methodValid) return;
    setBusy(true);
    try {
      setErrorMsg(null);
      const body: any = {
        row,
        method,
        system_prompt: selectedPrompt,
        task_description: canTaskDescribe && taskDescription.trim().length > 0 ? taskDescription : undefined,
        model_name: modelName,
        temperature,
        top_p: topP,
        max_tokens: maxTokens,
        max_workers: maxWorkers,
      };
      console.log('[PropertyExtraction] Calling extractSingle with body:', body);
      const res = await extractSingle({ ...body, return_debug: true });
      console.log('[PropertyExtraction] extractSingle response:', res);
      onPropertiesMerged(res.properties || []);
      setLastExtractProps(res.properties || []);
      
      // Open the trace viewer to show the selected row
      if (onOpenTrace && row) {
        onOpenTrace(row);
      }
      
      // Auto-scroll to results section after extraction completes
      setTimeout(() => {
        if (resultsRef.current) {
          resultsRef.current.scrollIntoView({
            behavior: 'smooth',
            block: 'start',
            inline: 'nearest'
          });
        }
      }, 300); // Small delay to ensure results are rendered
      
      if ((res.failures || []).length > 0) {
        setErrorMsg(`Parsing issues detected (${res.failures.length}). Try a different prompt or check JSON format.`);
      }
    } catch (e: any) {
      console.error('[PropertyExtraction] Error in runExtractSingle:', e);
      setErrorMsg(String(e?.message || e));
    } finally {
      setBusy(false);
    }
  }

  async function runExtractBatch() {
    const rows = getAllRows();
    const methodValid = method === 'single_model' || method === 'side_by_side';
    if (!rows || rows.length === 0 || !methodValid) return;

    // Close the trace viewer to focus on batch progress
    onCloseTrace?.();

    setBusy(true);
    onBatchStart?.();
    try {
      setErrorMsg(null);
      setJobProgress(0);
      setJobState('queued');
      const startRes = await extractJobStart({
        rows,
        method,
        system_prompt: selectedPrompt,
        task_description: canTaskDescribe && taskDescription.trim().length > 0 ? taskDescription : undefined,
        model_name: modelName,
        temperature,
        top_p: topP,
        max_tokens: maxTokens,
        max_workers: maxWorkers,
        sample_size: sampleSize || undefined,
      });
      setJobId(startRes.job_id);
      onBatchStart?.();
      await new Promise<void>((resolve, reject) => {
        const t = setInterval(async () => {
          try {
            const s = await extractJobStatus(startRes.job_id);
            setJobState(s.state);
            setJobProgress(s.progress || 0);
            onBatchStatus?.(s.progress || 0, s.state);
            if (s.state === 'done') {
              clearInterval(t);
              const r = await extractJobResult(startRes.job_id);
              (onBatchLoaded as any)?.(r.properties || []);
              onBatchDone?.();
              resolve();
            } else if (s.state === 'cancelled') {
              clearInterval(t);
              const r = await extractJobResult(startRes.job_id);
              (onBatchLoaded as any)?.(r.properties || []);
              setErrorMsg(`Job cancelled. Retrieved ${r.properties?.length || 0} partial results.`);
              onBatchDone?.();
              resolve();
            } else if (s.state === 'error') {
              clearInterval(t);
              reject(new Error(s.error || 'Job error'));
            }
          } catch (e) {
            clearInterval(t);
            reject(e);
          }
        }, 1000);
      });
    } finally {
      setBusy(false);
    }
  }

  async function handleCancelJob() {
    if (!jobId) return;
    try {
      await extractJobCancel(jobId);
      setJobState('cancelled');
    } catch (e: any) {
      setErrorMsg(`Failed to cancel: ${String(e?.message || e)}`);
    }
  }

  const methodValid = method === 'single_model' || method === 'side_by_side';

  return (
    <Stack spacing={3}>
      <Box>
        <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
          Extraction Prompt
        </Typography>
        <Stack spacing={2}>
          <Autocomplete
            size="small"
            options={promptOptions.map(p => p.name)}
            value={selectedPrompt}
            onChange={(_, v) => {
              if (v) {
                setSelectedPrompt(v);
                localStorage.setItem('stringsight.selectedPrompt', v);
                // Reset edited flag on prompt change
                setUserEdited(false);
                localStorage.setItem('stringsight.taskDescriptionEdited', 'false');
              }
            }}
            renderInput={(params) => <TextField {...params} label="Prompt" />}
          />
          
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            {promptOptions.length} prompts available
          </Typography>
          
          {canTaskDescribe && (
            <Stack spacing={1}>
              <TextField 
                label="Task description" 
                value={taskDescription} 
                onChange={(e) => {
                  const val = e.target.value;
                  setTaskDescription(val);
                  setUserEdited(true);
                  localStorage.setItem('stringsight.taskDescription', val);
                  localStorage.setItem('stringsight.taskDescriptionEdited', 'true');
                }} 
                minRows={3} 
                multiline 
              />
              <Box>
                <Button
                  size="small"
                  variant="text"
                  onClick={() => {
                    const def = method === 'side_by_side' ? (selectedPromptMeta?.default_task_description_sbs || '') : (selectedPromptMeta?.default_task_description_single || '');
                    setTaskDescription(def);
                    setUserEdited(false);
                    localStorage.setItem('stringsight.taskDescription', def);
                    localStorage.setItem('stringsight.taskDescriptionEdited', 'false');
                  }}
                >
                  Reset to default
                </Button>
              </Box>
            </Stack>
          )}

          {/* Resolved prompt preview */}
          <Box sx={{ 
            p: 2, 
            border: '1px dashed', 
            borderColor: 'divider', 
            borderRadius: 1, 
            backgroundColor: 'background.default' 
          }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block' }}>
              Resolved system prompt {canTaskDescribe ? '(task description highlighted in blue)' : ''}
            </Typography>
            <Box sx={{
              p: 1.5,
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 1,
              backgroundColor: '#FFFFFF',
              maxHeight: 280,
              overflow: 'auto',
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \\"Liberation Mono\\", \\"Courier New\\", monospace',
              fontSize: 12,
              whiteSpace: 'pre-wrap',
              lineHeight: 1.6,
            }}>
              {resolvedPrompt ? highlightedResolvedPrompt : 'Loading prompt…'}
            </Box>
          </Box>
        </Stack>
      </Box>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle2">Advanced (LLM settings)</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Stack spacing={2}>
            <TextField 
              size="small" 
              label="Model" 
              value={modelName} 
              onChange={(e) => setModelName(e.target.value)} 
            />
            <TextField 
              size="small" 
              label="Temperature" 
              type="number" 
              inputProps={{ step: 0.1 }} 
              value={temperature} 
              onChange={(e) => setTemperature(Number(e.target.value))} 
            />
            <TextField 
              size="small" 
              label="Top P" 
              type="number" 
              inputProps={{ step: 0.05 }} 
              value={topP} 
              onChange={(e) => setTopP(Number(e.target.value))} 
            />
            <TextField 
              size="small" 
              label="Max tokens" 
              type="number" 
              value={maxTokens} 
              onChange={(e) => setMaxTokens(Number(e.target.value))} 
            />
            <TextField 
              size="small" 
              label="Max workers" 
              type="number" 
              value={maxWorkers} 
              onChange={(e) => setMaxWorkers(Number(e.target.value))} 
            />
            <TextField 
              size="small" 
              label="Sample size (batch only)" 
              type="number" 
              value={sampleSize || ''} 
              onChange={(e) => setSampleSize(e.target.value ? Number(e.target.value) : null)} 
              placeholder="Leave empty for all prompts"
              helperText={sampleSize ? `Will sample ${sampleSize} prompts total` : 'Process all prompts'}
            />
          </Stack>
        </AccordionDetails>
      </Accordion>

      {/* View Selected Response Button */}
      <Button
        variant="outlined"
        onClick={() => {
          const row = getSelectedRow();
          if (onOpenTrace && row) {
            onOpenTrace(row);
          }
        }}
        disabled={!getSelectedRow()}
        fullWidth
        sx={{ mb: 2 }}
      >
        {(() => {
          const row = getSelectedRow();
          if (!row) return 'No Response Selected';
          const index = (row as any)?.__index;
          return index !== undefined ? `View Response (Row ${index})` : 'View Selected Response';
        })()}
      </Button>

      {busy && (
        <Box sx={{ width: '100%', mb: 2 }}>
          <Typography variant="body2" sx={{ color: 'primary.main', mb: 0.5 }}>
            {jobState ? `Batch: ${jobState} • ${Math.round((jobProgress||0)*100)}%` : 'Running extraction…'}
          </Typography>
          {/* Indeterminate until first progress update (> 0), then determinate */}
          <LinearProgress
            variant={(jobProgress||0) > 0 ? 'determinate' : 'indeterminate'}
            value={(jobProgress||0)*100}
          />
          {/* Cancel button for batch jobs */}
          {jobId && jobState && !['done', 'error', 'cancelled'].includes(jobState) && (
            <Button
              size="small"
              variant="outlined"
              color="error"
              onClick={handleCancelJob}
              sx={{ mt: 1, width: '100%' }}
            >
              Cancel Extraction
            </Button>
          )}
        </Box>
      )}

      <Box sx={{ display: 'flex', gap: 1, flexDirection: 'column' }}>
          <Button 
            variant="contained" 
            onClick={runExtractSingle} 
            disabled={busy || !methodValid || !getSelectedRow()}
            sx={{ width: '100%' }}
          >
            {(() => {
              const row = getSelectedRow();
              if (!row) return 'Extract on selected';
              const index = (row as any)?.__index;
              return index !== undefined ? `Extract on Row ${index}` : 'Extract on selected';
            })()}
          </Button>
          <Button 
            variant="outlined" 
            onClick={runExtractBatch} 
            disabled={busy || !methodValid}
            sx={{ width: '100%' }}
          >
            {sampleSize && sampleSize > 0 
              ? `Run on sample (${sampleSize} prompts)` 
              : `Run on all traces (${getAllRows().length})`}
          </Button>
        </Box>

      {errorMsg && (
        <Box sx={{ 
          p: 2, 
          border: '1px solid', 
          borderColor: 'error.main', 
          backgroundColor: 'error.light', 
          color: 'error.contrastText', 
          borderRadius: 1 
        }}>
          <Typography variant="body2">{errorMsg}</Typography>
        </Box>
      )}

      {lastExtractProps.length > 0 && (
        <Box ref={resultsRef} sx={{ 
          p: 2, 
          border: '1px solid', 
          borderColor: 'divider', 
          backgroundColor: 'background.paper', 
          borderRadius: 1 
        }}>
          <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
            Last extraction result
          </Typography>
          <Stack spacing={1}>
            {lastExtractProps.map((p, i) => (
              <Accordion 
                key={i} 
                disableGutters 
                sx={{ 
                  boxShadow: 'none', 
                  border: '1px solid', 
                  borderColor: 'divider', 
                  borderRadius: 1, 
                  backgroundColor: 'background.default' 
                }}
                onChange={(_, expanded) => {
                  if (expanded) {
                    const raw = p?.evidence;
                    let list: string[] = [];
                    if (Array.isArray(raw)) list = raw.map((s: any) => String(s || '').trim()).filter(Boolean);
                    else if (typeof raw === 'string') {
                      const trimmed = raw.trim();
                      const parts = trimmed.split(/"\s*,\s*"|\n|,\s(?=[\w\d])/g).map(s => s.replace(/^"|"$/g, '').trim());
                      list = parts.filter(Boolean);
                    }
                    onSelectEvidence(list, (p as any).model);
                  }
                }}
              >
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {p.property_description || `Property ${i + 1}`}
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ display: 'grid', gridTemplateColumns: '120px 1fr', rowGap: 0.5, columnGap: 1 }}>
                    {Object.entries(p).map(([k, v]) => (
                      <React.Fragment key={k}>
                        <Typography variant="caption" sx={{ color: 'text.secondary' }}>{k}</Typography>
                        <Typography variant="caption" sx={{ color: 'text.primary' }}>
                          {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                        </Typography>
                      </React.Fragment>
                    ))}
                  </Box>
                </AccordionDetails>
              </Accordion>
            ))}
          </Stack>
        </Box>
      )}
    </Stack>
  );
}

