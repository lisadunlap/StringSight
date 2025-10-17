import React from 'react';
import { Drawer, Box, Stack, Typography, Divider, TextField, Button, Autocomplete, Accordion, AccordionSummary, AccordionDetails, FormControlLabel, Switch, LinearProgress } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { getPrompts, getPromptText, extractSingle, extractJobStart, extractJobStatus, extractJobResult } from '../lib/api';

type Method = 'single_model' | 'side_by_side' | 'unknown';

export default function ControlSidebar({
  open,
  onClose,
  method,
  hasAnyProperties,
  getSelectedRow,
  getAllRows,
  onPropertiesMerged,
  onSelectEvidence,
  onBatchLoaded,
  onBatchStart,
  onBatchStatus,
  onBatchDone,
}: {
  open: boolean;
  onClose: () => void;
  method: Method;
  hasAnyProperties: boolean;
  getSelectedRow: () => Record<string, any> | null;
  getAllRows: () => Record<string, any>[];
  onPropertiesMerged: (props: any[]) => void;
  onSelectEvidence: (evidence: string[], targetModel?: string) => void;
  onBatchLoaded: (rows: any[]) => void;
  onBatchStart?: () => void;
  onBatchStatus?: (progress: number, state: string | null) => void;
  onBatchDone?: () => void;
}) {
  const [promptOptions, setPromptOptions] = React.useState<{ name: string; label: string; has_task_description: boolean; preview: string; }[]>([]);
  const [selectedPrompt, setSelectedPrompt] = React.useState<string>('single_model_system_prompt');
  const [taskDescription, setTaskDescription] = React.useState<string>('');
  const [customPrompt, setCustomPrompt] = React.useState<string>('');
  const [useCustom, setUseCustom] = React.useState<boolean>(false);
  const [resolvedPrompt, setResolvedPrompt] = React.useState<string>('');

  const [modelName, setModelName] = React.useState<string>('gpt-4o-mini');
  const [temperature, setTemperature] = React.useState<number>(0.6);
  const [topP, setTopP] = React.useState<number>(0.95);
  const [maxTokens, setMaxTokens] = React.useState<number>(2048);
  const [maxWorkers, setMaxWorkers] = React.useState<number>(16);

  const [busy, setBusy] = React.useState<boolean>(false);
  const [lastExtractProps, setLastExtractProps] = React.useState<any[]>([]);
  const [errorMsg, setErrorMsg] = React.useState<string | null>(null);
  const [jobId, setJobId] = React.useState<string | null>(null);
  const [jobProgress, setJobProgress] = React.useState<number>(0);
  const [jobState, setJobState] = React.useState<string | null>(null);

  // Debug: log when component mounts
  React.useEffect(() => {
    console.log('[ControlSidebar] Component mounted, open:', open);
  }, [open]);

  React.useEffect(() => {
    let mounted = true;
    console.log('[ControlSidebar] Starting to fetch prompts...');
    (async () => {
      try {
        console.log('[ControlSidebar] Calling getPrompts()...');
        const res = await getPrompts();
        console.log('[ControlSidebar] getPrompts() response:', res);
        if (mounted) {
          const list = res.prompts || [];
          console.log('[ControlSidebar] Setting prompt options:', list.length, 'prompts');
          setPromptOptions(list);
          // Ensure selectedPrompt is a valid option
          if (!list.find(p => p.name === selectedPrompt) && list.length > 0) {
            setSelectedPrompt(list[0].name);
          }
        }
      } catch (e: any) {
        console.error('[ControlSidebar] Error fetching prompts:', e);
        if (mounted) {
          setErrorMsg(`API blocked (likely ad blocker). Using fallback prompts.`);
          // Fallback defaults so the dropdown isn't empty
          const fallback = [
            { name: 'default', label: 'Default', has_task_description: true, preview: 'Default task-aware analysis prompt' },
            { name: 'agent', label: 'Agent', has_task_description: true, preview: 'Agent task-aware analysis prompt' },
          ];
          setPromptOptions(fallback);
          setSelectedPrompt('default');
        }
      }
    })();
    return () => { mounted = false; };
  }, []);

  const canTaskDescribe = React.useMemo(() => {
    const p = promptOptions.find(p => p.name === selectedPrompt);
    return Boolean(p?.has_task_description);
  }, [promptOptions, selectedPrompt]);

  React.useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        if (useCustom) {
          if (mounted) setResolvedPrompt(customPrompt);
        } else {
          const res = await getPromptText(selectedPrompt, canTaskDescribe ? taskDescription : undefined);
          if (mounted) setResolvedPrompt(res.text);
        }
      } catch (e: any) {
        if (mounted) {
          setResolvedPrompt('');
          setErrorMsg(`Failed to load prompt '${selectedPrompt}': ${String(e?.message || e)}`);
        }
      }
    })();
    return () => { mounted = false; };
  }, [useCustom, customPrompt, selectedPrompt, canTaskDescribe, taskDescription]);

  async function runExtractSingle() {
    // Use the currently selected row from the main table/view
    const row = getSelectedRow();
    const methodValid = method === 'single_model' || method === 'side_by_side';
    if (!row || !methodValid) return;
    setBusy(true);
    try {
      setErrorMsg(null);
      const body: any = {
        row,
        method,
        system_prompt: useCustom ? customPrompt : selectedPrompt,
        task_description: canTaskDescribe ? taskDescription : undefined,
        model_name: modelName,
        temperature,
        top_p: topP,
        max_tokens: maxTokens,
        max_workers: maxWorkers,
      };
      const res = await extractSingle({ ...body, return_debug: true });
      onPropertiesMerged(res.properties || []);
      setLastExtractProps(res.properties || []);
      if ((res.failures || []).length > 0) {
        setErrorMsg(`Parsing issues detected (${res.failures.length}). Try a different prompt or check JSON format.`);
      }
    } catch (e: any) {
      setErrorMsg(String(e?.message || e));
    } finally {
      setBusy(false);
    }
  }

  async function runExtractBatch() {
    const rows = getAllRows();
    const methodValid = method === 'single_model' || method === 'side_by_side';
    if (!rows || rows.length === 0 || !methodValid) return;
    setBusy(true);
    onBatchStart?.();
    try {
      setErrorMsg(null);
      setJobProgress(0);
      setJobState('queued');
      const startRes = await extractJobStart({
        rows,
        method,
        system_prompt: useCustom ? customPrompt : selectedPrompt,
        task_description: canTaskDescribe ? taskDescription : undefined,
        model_name: modelName,
        temperature,
        top_p: topP,
        max_tokens: maxTokens,
        max_workers: maxWorkers,
        chunk_size: 250,
      });
      setJobId(startRes.job_id);
      // Notify app to close the response drawer during batch
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
              // Batch rows to parent for Properties tab
              // We do not merge into propertiesByKey here
              (onBatchLoaded as any)?.(r.properties || []);
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

  return (
    <Drawer anchor="left" open={open} onClose={onClose} variant="persistent" sx={{ '& .MuiDrawer-paper': { width: 480, p: 2 } }}>
      <Stack spacing={2}>
        <Stack direction="row" alignItems="center" justifyContent="space-between">
          <Typography variant="h6">Pipeline Controls</Typography>
          <Button size="small" variant="text" onClick={onClose} sx={{ fontWeight: 600 }}>Collapse ◀</Button>
        </Stack>
        <Divider />

        {/* Conversation preview removed; right drawer shows the selected trace */}

        <Typography variant="subtitle2">Extraction Prompt</Typography>
        <FormControlLabel control={<Switch checked={useCustom} onChange={(_, c) => setUseCustom(c)} />} label="Use custom prompt" />
        {!useCustom && (
          <Autocomplete
            size="small"
            options={promptOptions.map(p => p.name)}
            value={selectedPrompt}
            onChange={(_, v) => v && setSelectedPrompt(v)}
            renderInput={(params) => <TextField {...params} label="Prompt" />}
          />
        )}
        <Typography variant="caption" sx={{ color: '#64748B' }}>{promptOptions.length} prompts available</Typography>
        {useCustom && (
          <TextField
            label="Custom system prompt"
            value={customPrompt}
            onChange={(e) => setCustomPrompt(e.target.value)}
            minRows={4}
            multiline
          />
        )}
        {canTaskDescribe && !useCustom && (
          <TextField label="Task description" value={taskDescription} onChange={(e) => setTaskDescription(e.target.value)} minRows={3} multiline />
        )}

        {/* Show resolved system prompt */}
        <Box sx={{ p: 1, border: '1px dashed #CBD5E1', borderRadius: 1, background: '#F8FAFC' }}>
          <Typography variant="caption" sx={{ color: '#64748B' }}>Resolved system prompt</Typography>
          <TextField value={resolvedPrompt || 'Loading prompt…'} multiline minRows={6} maxRows={14} fullWidth size="small" sx={{ mt: 1 }} InputProps={{ readOnly: true }} />
        </Box>

        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>Advanced (LLM settings)</AccordionSummary>
          <AccordionDetails>
            <Stack spacing={1}>
              <TextField size="small" label="Model" value={modelName} onChange={(e) => setModelName(e.target.value)} />
              <TextField size="small" label="Temperature" type="number" inputProps={{ step: 0.1 }} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
              <TextField size="small" label="Top P" type="number" inputProps={{ step: 0.05 }} value={topP} onChange={(e) => setTopP(Number(e.target.value))} />
              <TextField size="small" label="Max tokens" type="number" value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))} />
              <TextField size="small" label="Max workers" type="number" value={maxWorkers} onChange={(e) => setMaxWorkers(Number(e.target.value))} />
            </Stack>
          </AccordionDetails>
        </Accordion>

        {(() => { const methodValid = method === 'single_model' || method === 'side_by_side'; return (
          <Stack direction="row" spacing={1}>
            <Button variant="contained" onClick={runExtractSingle} disabled={busy || !methodValid || !getSelectedRow()}>Extract on selected</Button>
            <Button variant="outlined" onClick={runExtractBatch} disabled={busy || !methodValid}>Run on all traces</Button>
          </Stack>
        ); })()}
        </Stack>
        {busy && (
          <Box sx={{ width: '100%' }}>
            <Typography variant="body2" sx={{ color: '#0EA5E9', mb: 0.5 }}>
              {jobState ? `Batch: ${jobState} • ${Math.round((jobProgress||0)*100)}%` : 'Running extraction…'}
            </Typography>
            <LinearProgress variant={jobState ? 'determinate' : 'indeterminate'} value={(jobProgress||0)*100} />
          </Box>
        )}

        {errorMsg && (
          <Box sx={{ p: 1, border: '1px solid #FCA5A5', background: '#FEF2F2', color: '#991B1B', borderRadius: 1 }}>
            {errorMsg}
          </Box>
        )}

        {lastExtractProps.length > 0 && (
          <Box sx={{ p: 1, border: '1px solid #E5E7EB', background: '#FFFFFF', borderRadius: 1 }}>
            <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Last extraction result</Typography>
            <Stack spacing={1}>
              {lastExtractProps.map((p, i) => (
                <Accordion key={i} disableGutters sx={{ boxShadow: 'none', border: '1px solid #F1F5F9', borderRadius: 1, background: '#F8FAFC' }}
                  onChange={(_, expanded) => {
                    if (expanded) {
                      // Normalize evidence into string[] and notify parent to highlight
                      const raw = p?.evidence;
                      let list: string[] = [];
                      if (Array.isArray(raw)) list = raw.map((s: any) => String(s || '').trim()).filter(Boolean);
                      else if (typeof raw === 'string') {
                        const trimmed = raw.trim();
                        // Split on quote/comma patterns or fallback commas
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
                    <Box sx={{ display: 'grid', gridTemplateColumns: '180px 1fr', rowGap: 0.5, columnGap: 1 }}>
                      {Object.entries(p).map(([k, v]) => (
                        <React.Fragment key={k}>
                          <Typography variant="caption" sx={{ color: '#64748B' }}>{k}</Typography>
                          <Typography variant="caption" sx={{ color: '#0F172A' }}>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</Typography>
                        </React.Fragment>
                      ))}
                    </Box>
                  </AccordionDetails>
                </Accordion>
              ))}
            </Stack>
          </Box>
        )}

        <Divider />
        <Typography variant="subtitle2">Clustering</Typography>
        <Stack spacing={1}>
          <TextField size="small" label="Min cluster size" type="number" disabled={!hasAnyProperties} />
          <TextField size="small" label="Embedding model" disabled={!hasAnyProperties} />
          <FormControlLabel control={<Switch disabled={!hasAnyProperties} />} label="Hierarchical" />
          <FormControlLabel control={<Switch disabled={!hasAnyProperties} />} label="Assign outliers" />
          <Button variant="outlined" disabled={!hasAnyProperties}>Cluster properties</Button>
        </Stack>
      
    </Drawer>
  );
}



