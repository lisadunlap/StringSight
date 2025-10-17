import React from 'react';
import { Box, Typography, Accordion, AccordionSummary, AccordionDetails, Button, Stack, Chip, Tooltip, IconButton, Fade } from '@mui/material';
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';
const Plot = createPlotlyComponent(Plotly);
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import { FormControl, InputLabel, Select, MenuItem, TextField, Checkbox, ListItemText, OutlinedInput } from '@mui/material';

interface ClustersTabProps {
  clusters: any[];
  totalConversationsByModel?: Record<string, number> | null;
  totalUniqueConversations?: number | null;
  onOpenPropertyById: (id: string) => void;
  getPropertiesRows?: () => any[];
  onRequestRecompute?: (included_property_ids?: string[]) => void;
}

function formatPercent(p?: number): string {
  if (typeof p !== 'number' || !isFinite(p)) return '';
  return `${(p * 100).toFixed(1)}%`;
}

function ClustersTab({ clusters, totalConversationsByModel, totalUniqueConversations, onOpenPropertyById, getPropertiesRows, onRequestRecompute }: ClustersTabProps) {
  // Smooth entrance animation on initial mount
  const animateOnMountRef = React.useRef(true);
  React.useEffect(() => {
    const id = requestAnimationFrame(() => {
      animateOnMountRef.current = false;
    });
    return () => cancelAnimationFrame(id);
  }, []);
  const [decimals, setDecimals] = React.useState<number>(3);
  const [modeByCluster, setModeByCluster] = React.useState<Record<string, 'quality' | 'delta'>>({});
  const [search, setSearch] = React.useState<string>('');
  const [selectedModels, setSelectedModels] = React.useState<string[]>([]);
  const [selectedGroups, setSelectedGroups] = React.useState<string[]>([]);
  const [sortBy, setSortBy] = React.useState<'freqAsc' | 'freqDesc' | 'qualAsc' | 'qualDesc'>('freqDesc');
  const debouncedApplyRef = React.useRef<number | null>(null);

  const allModels = React.useMemo<string[]>(() => {
    // Prefer stable model list from properties if available
    try {
      const props = typeof getPropertiesRows === 'function' ? (getPropertiesRows() || []) : [];
      const fromProps = new Set<string>();
      (props || []).forEach((p: any) => {
        const m = p?.model != null ? String(p.model) : '';
        if (m) fromProps.add(m);
      });
      if (fromProps.size > 0) return Array.from(fromProps).sort((a, b) => a.localeCompare(b));
    } catch (_) {}

    // Fallback to deriving from current clusters
    const s = new Set<string>();
    (clusters || []).forEach((c) => {
      const meta = (c && c.meta) || {};
      const byModel = (meta.proportion_by_model && Object.keys(meta.proportion_by_model)) || [];
      const qualityByModel = (meta.quality_by_model && Object.keys(meta.quality_by_model)) || [];
      [...byModel, ...qualityByModel].forEach((m) => { if (m) s.add(String(m)); });
    });
    return Array.from(s).sort((a, b) => a.localeCompare(b));
  }, [clusters, getPropertiesRows]);

  const allGroups = React.useMemo<string[]>(() => {
    const s = new Set<string>();
    (clusters || []).forEach((c) => {
      const meta = (c && c.meta) || {};
      const g = meta.group;
      if (g != null && g !== '') s.add(String(g));
    });
    return Array.from(s).sort((a, b) => a.localeCompare(b));
  }, [clusters]);

  const propertiesById = React.useMemo<Map<string, any>>(() => {
    const map = new Map<string, any>();
    const props = typeof getPropertiesRows === 'function' ? getPropertiesRows() : [];
    (props || []).forEach((p: any) => {
      const id = p?.id != null ? String(p.id) : null;
      if (id) map.set(id, p);
    });
    return map;
  }, [getPropertiesRows]);

  const applyRecompute = React.useCallback(() => {
    if (!onRequestRecompute) return;
    const hasModelFilter = selectedModels.length > 0;
    const hasGroupFilter = selectedGroups.length > 0;
    if (!hasModelFilter && !hasGroupFilter) {
      onRequestRecompute(undefined);
      return;
    }
    const included = new Set<string>();
    (clusters || []).forEach((c) => {
      const meta = (c && c.meta) || {};
      if (hasGroupFilter) {
        const g = meta.group != null ? String(meta.group) : null;
        if (!g || !selectedGroups.includes(g)) return;
      }
      // Always derive from the cluster's property_ids (stable superset),
      // falling back to meta.property_items only if property_ids are missing.
      const idsFromCluster: string[] = Array.isArray((c as any).property_ids)
        ? (c as any).property_ids.map((pid: any) => (pid != null ? String(pid) : '')).filter(Boolean)
        : (Array.isArray((meta as any).property_items)
            ? (meta as any).property_items.map((it: any) => (it?.property_id != null ? String(it.property_id) : '')).filter(Boolean)
            : []);
      idsFromCluster.forEach((pid) => {
        if (!pid) return;
        if (hasModelFilter) {
          const prop = propertiesById.get(pid);
          const model = prop?.model != null ? String(prop.model) : null;
          if (model && selectedModels.includes(model)) included.add(pid);
        } else {
          included.add(pid);
        }
      });
    });
    onRequestRecompute(Array.from(included));
  }, [onRequestRecompute, selectedModels, selectedGroups, clusters, propertiesById]);

  const requestRecomputeDebounced = React.useCallback(() => {
    if (debouncedApplyRef.current) {
      window.clearTimeout(debouncedApplyRef.current);
      debouncedApplyRef.current = null;
    }
    debouncedApplyRef.current = window.setTimeout(() => {
      applyRecompute();
    }, 300);
  }, [applyRecompute]);

  const visibleClusters = React.useMemo(() => {
    const query = search.trim().toLowerCase();
    const hasSearch = query.length > 0;
    const hasModelFilter = selectedModels.length > 0;
    const hasGroupFilter = selectedGroups.length > 0;

    const avgQuality = (c: any): number => {
      const q = (c && c.meta && c.meta.quality) || {};
      const vals = Object.values(q).map((v: any) => Number(v)).filter((v) => Number.isFinite(v));
      if (!vals.length) return -Infinity; // ensure clusters without quality sink when sorting by quality
      const sum = vals.reduce((a, b) => a + b, 0);
      return sum / vals.length;
    };

    let list = (clusters || []).filter((c) => {
      const meta = (c && c.meta) || {};

      // search match: cluster label or any property description
      let matchesSearch = true;
      if (hasSearch) {
        const label = String(c.label || '').toLowerCase();
        const fromMetaItems: string[] = Array.isArray((meta as any).property_items)
          ? (meta as any).property_items.map((it: any) => String(it?.property_description || '').toLowerCase())
          : [];
        const fromDescriptions: string[] = Array.isArray(c.property_descriptions)
          ? c.property_descriptions.map((s: any) => String(s || '').toLowerCase())
          : [];
        matchesSearch = label.includes(query) || fromMetaItems.some((t) => t.includes(query)) || fromDescriptions.some((t) => t.includes(query));
      }

      if (!matchesSearch) return false;

      // model filter: require intersection with models present in cluster meta
      if (hasModelFilter) {
        const modelsInCluster = new Set<string>([
          ...Object.keys(meta.proportion_by_model || {}),
          ...Object.keys(meta.quality_by_model || {}),
        ].map((m) => String(m)));
        const intersects = selectedModels.some((m) => modelsInCluster.has(String(m)));
        if (!intersects) return false;
      }

      // group filter: cluster meta.group must be one of selected
      if (hasGroupFilter) {
        const g = meta.group != null ? String(meta.group) : null;
        if (!g || !selectedGroups.includes(g)) return false;
      }

      return true;
    });

    // sort
    switch (sortBy) {
      case 'qualAsc':
        list = [...list].sort((a, b) => avgQuality(a) - avgQuality(b));
        break;
      case 'qualDesc':
        list = [...list].sort((a, b) => avgQuality(b) - avgQuality(a));
        break;
      case 'freqAsc':
        list = [...list].sort((a, b) => (Number(a.size || 0) - Number(b.size || 0)));
        break;
      case 'freqDesc':
      default:
        list = [...list].sort((a, b) => (Number(b.size || 0) - Number(a.size || 0)));
        break;
    }
    return list;
  }, [clusters, search, selectedModels, selectedGroups, sortBy]);
  if (!clusters || clusters.length === 0) {
    return (
      <Box sx={{ p: 2, border: '1px solid #E5E7EB', borderRadius: 2, background: '#FFFFFF' }}>
        <Typography variant="body1" color="text.secondary">No clusters yet. Run clustering from the sidebar.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ border: '1px solid #E5E7EB', borderRadius: 2, overflow: 'hidden', background: '#FFFFFF' }}>
      {/* Legend / helper text */}
      <Box sx={{ p: 1.5, background: '#F3F4F6', borderBottom: '1px solid #E5E7EB', display: 'flex', alignItems: 'center', gap: 2, justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
          <InfoOutlinedIcon sx={{ fontSize: 16, color: '#6B7280' }} />
          <Typography variant="caption" color="text.secondary">
            Overall proportion = share of all conversations in this cluster. Per-model proportion = share of that model's conversations in this cluster.
          </Typography>
        </Box>
        <Stack direction="row" spacing={1} sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
          <TextField
            size="small"
            placeholder="Search clusters or properties..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            sx={{ minWidth: 260 }}
          />
          <FormControl size="small" sx={{ minWidth: 160 }}>
            <InputLabel id="models-label">Models</InputLabel>
            <Select
              labelId="models-label"
              multiple
              value={selectedModels}
              onChange={(e) => {
                setSelectedModels(typeof e.target.value === 'string' ? e.target.value.split(',') : (e.target.value as string[]));
                requestRecomputeDebounced();
              }}
              input={<OutlinedInput label="Models" />}
              renderValue={(selected) => {
                const count = (selected as string[]).length;
                return count > 0 ? `${count} selected` : 'All';
              }}
              MenuProps={{ PaperProps: { style: { maxHeight: 320 } } }}
            >
              {allModels.map((name) => (
                <MenuItem key={name} value={name}>
                  <Checkbox checked={selectedModels.indexOf(name) > -1} />
                  <ListItemText primary={name} />
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 160 }} disabled={allGroups.length === 0}>
            <InputLabel id="groups-label">Group</InputLabel>
            <Select
              labelId="groups-label"
              multiple
              value={selectedGroups}
              onChange={(e) => {
                setSelectedGroups(typeof e.target.value === 'string' ? e.target.value.split(',') : (e.target.value as string[]));
                requestRecomputeDebounced();
              }}
              input={<OutlinedInput label="Group" />}
              renderValue={(selected) => {
                const count = (selected as string[]).length;
                return count > 0 ? `${count} selected` : 'All';
              }}
              MenuProps={{ PaperProps: { style: { maxHeight: 320 } } }}
            >
              {allGroups.map((g) => (
                <MenuItem key={g} value={g}>
                  <Checkbox checked={selectedGroups.indexOf(g) > -1} />
                  <ListItemText primary={g} />
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 180 }}>
            <InputLabel id="sortby-label">Sort By</InputLabel>
            <Select labelId="sortby-label" value={sortBy} label="Sort By" onChange={(e) => setSortBy(e.target.value as any)}>
              <MenuItem value={'freqAsc'}>Frequency ▲</MenuItem>
              <MenuItem value={'freqDesc'}>Frequency ▼</MenuItem>
              <MenuItem value={'qualAsc'}>Quality ▲</MenuItem>
              <MenuItem value={'qualDesc'}>Quality ▼</MenuItem>
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel id="decimals-label">Decimals</InputLabel>
            <Select labelId="decimals-label" value={decimals} label="Decimals" onChange={(e) => setDecimals(Number(e.target.value))}>
              <MenuItem value={1}>1</MenuItem>
              <MenuItem value={2}>2</MenuItem>
              <MenuItem value={3}>3</MenuItem>
              <MenuItem value={4}>4</MenuItem>
            </Select>
          </FormControl>
        </Stack>
      </Box>
      {(selectedModels.length > 0 || selectedGroups.length > 0 || (search.trim().length > 0)) && (
        <Box sx={{ px: 1.5, py: 1, borderBottom: '1px solid #E5E7EB', display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', background: '#F9FAFB' }}>
          <Typography variant="caption" sx={{ color: '#6B7280' }}>Active filters:</Typography>
          {selectedModels.map((m) => (
            <Chip
              key={`model-${m}`}
              size="small"
              color="primary"
              variant="outlined"
              label={m}
              onDelete={() => { setSelectedModels((prev) => prev.filter((x) => x !== m)); requestRecomputeDebounced(); }}
            />
          ))}
          {selectedGroups.map((g) => {
            // Determine chip color and variant based on group name (same logic as main chips)
            let chipColor: 'default' | 'primary' | 'secondary' | 'error' | 'warning' | 'info' | 'success' = 'secondary';
            let chipVariant: 'filled' | 'outlined' = 'outlined';
            let chipStyle = {};
            
            const groupLower = g.toLowerCase();
            if (groupLower === 'positive') {
              chipColor = 'success';
              chipVariant = 'filled';
            } else if (groupLower === 'negative (critical)') {
              chipColor = 'error';
              chipVariant = 'filled';
            } else if (groupLower === 'negative (non-critical)') {
              chipColor = 'warning';
              chipVariant = 'filled';
            } else if (groupLower === 'style') {
              chipVariant = 'filled';
              chipStyle = { 
                backgroundColor: '#9C27B0', 
                color: 'white',
                '&:hover': { backgroundColor: '#7B1FA2' }
              };
            }
            
            return (
              <Chip
                key={`group-${g}`}
                size="small"
                color={chipColor}
                variant={chipVariant}
                label={g}
                onDelete={() => { setSelectedGroups((prev) => prev.filter((x) => x !== g)); requestRecomputeDebounced(); }}
                sx={Object.keys(chipStyle).length > 0 ? chipStyle : undefined}
              />
            );
          })}
          {search.trim().length > 0 && (
            <Chip
              size="small"
              color="default"
              variant="outlined"
              label={`Search: ${search}`}
              onDelete={() => setSearch('')}
            />
          )}
          <Button size="small" variant="text" onClick={() => { setSelectedModels([]); setSelectedGroups([]); setSearch(''); requestRecomputeDebounced(); }}>Clear all</Button>
        </Box>
      )}
      {visibleClusters.map((c, idx) => {
        const meta = (c && c.meta) || {};
        const overallQuality: Record<string, number> = meta.quality || {};
        const overallQualityDelta: Record<string, number> = meta.quality_delta || {};
        const overallProp: number | undefined = meta.proportion_overall;
        const group: string | undefined = meta.group;
        const perModelProps: Record<string, number> = meta.proportion_by_model || {};
        const perModelQuality: Record<string, Record<string, number>> = meta.quality_by_model || {};

        const accordion = (
          <Accordion key={c.id ?? idx} sx={{ '&:before': { display: 'none' }, boxShadow: 'none', borderBottom: '1px solid #E5E7EB' }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ '& .MuiAccordionSummary-content': { my: 1 } }}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, width: '100%' }}>
                <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 2 }}>
                  <Typography variant="body1" sx={{ fontWeight: 600, color: '#111827' }}>{String(c.label || '')}</Typography>
                  {/* Overall quality metrics (cluster-level) */}
                  {overallQuality && Object.keys(overallQuality).length > 0 && (
                    <Tooltip title="Overall cluster quality across all models">
                      <Stack direction="column" spacing={0.25} sx={{ minWidth: 360, maxWidth: '50%', alignItems: 'flex-end' }}>
                        {Object.entries(overallQuality).map(([k, v]) => {
                          const d = overallQualityDelta && typeof overallQualityDelta[k] === 'number' ? overallQualityDelta[k] : undefined;
                          let deltaColor = '#6B7280';
                          if (typeof d === 'number') {
                            if (d > 0.02) deltaColor = '#16A34A'; // green-600
                            else if (d < -0.02) deltaColor = '#DC2626'; // red-600
                          }
                          return (
                            <Typography key={k} variant="body2" sx={{ color: '#334155', textAlign: 'right' }}>
                              {k}: {typeof v === 'number' ? v.toFixed(decimals) : String(v)}
                              {typeof d === 'number' && (
                                <Box component="span" sx={{ ml: 0.5, color: deltaColor }}>
                                  ({d >= 0 ? '+' : ''}{d.toFixed(decimals)})
                                </Box>
                              )}
                            </Typography>
                          );
                        })}
                      </Stack>
                    </Tooltip>
                  )}
                </Box>
                {/* Chips: size (with overall proportion) and group */}
                <Stack direction="row" spacing={1} sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
                  <Tooltip title="Count of conversations in this cluster (overall proportion across all models)">
                    <Box sx={{ color: '#6B7280', fontSize: 12 }}>
                      {(() => {
                        const clusterSize = c.size ?? 0;
                        const clusterSizeText = clusterSize.toLocaleString();
                        const overallPropText = overallProp !== undefined ? `(${formatPercent(overallProp)})` : '';
                        
                        // Calculate unique conversations percentage if we have the data
                        let conversationCountText = '';
                        if (totalUniqueConversations && totalUniqueConversations > 0) {
                          const conversationPct = (clusterSize / totalUniqueConversations) * 100;
                          conversationCountText = ` • ${clusterSize} conversations (${conversationPct.toFixed(1)}%)`;
                        }
                        
                        return `${clusterSizeText} ${overallPropText}${conversationCountText}`;
                      })()}
                    </Box>
                  </Tooltip>
                  {group && (() => {
                    // Determine chip color and variant based on group name
                    let chipColor: 'default' | 'primary' | 'secondary' | 'error' | 'warning' | 'info' | 'success' = 'default';
                    let chipVariant: 'filled' | 'outlined' = 'outlined';
                    let chipStyle = {};
                    
                    const groupLower = group.toLowerCase();
                    if (groupLower === 'positive') {
                      chipColor = 'success';
                      chipVariant = 'filled';
                    } else if (groupLower === 'negative (critical)') {
                      chipColor = 'error';
                      chipVariant = 'filled';
                    } else if (groupLower === 'negative (non-critical)') {
                      chipColor = 'warning';
                      chipVariant = 'filled';
                    } else if (groupLower === 'style') {
                      chipVariant = 'filled';
                      chipStyle = { 
                        backgroundColor: '#9C27B0', 
                        color: 'white',
                        '&:hover': { backgroundColor: '#7B1FA2' }
                      };
                    }
                    
                    return (
                      <Chip 
                        size="small" 
                        color={chipColor} 
                        variant={chipVariant}
                        label={group}
                        sx={Object.keys(chipStyle).length > 0 ? chipStyle : undefined}
                      />
                    );
                  })()}
                </Stack>
              </Box>
            </AccordionSummary>
            <AccordionDetails sx={{ background: '#FAFAFA' }}>
              <Box sx={{ p: 2 }}>
                {/* Per-model proportions */}
                {perModelProps && Object.keys(perModelProps).length > 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mb: 1 }}>
                      <Typography variant="subtitle2" sx={{ color: '#334155' }}>Per-model proportions</Typography>
                      <Tooltip title="Fraction of each model's conversations that appear in this cluster">
                        <IconButton size="small"><InfoOutlinedIcon sx={{ fontSize: 16 }} /></IconButton>
                      </Tooltip>
                    </Stack>
                    {(() => {
                      // Sort entries: non-zero values first (by descending value), then zero values alphabetically
                      const entries = Object.entries(perModelProps).sort((a, b) => {
                        const aVal = Number(a[1]);
                        const bVal = Number(b[1]);
                        
                        // If both are zero, sort alphabetically
                        if (aVal === 0 && bVal === 0) {
                          return a[0].localeCompare(b[0]);
                        }
                        // If one is zero and one isn't, non-zero comes first
                        if (aVal === 0) return 1;
                        if (bVal === 0) return -1;
                        // Both non-zero, sort by value descending
                        return bVal - aVal;
                      });
                      const x = entries.map(([m]) => m);
                      const y = entries.map(([, v]) => Number(v));
                      return (
                        <Plot
                          data={[{
                            type: 'bar' as const, x, y,
                            marker: { color: '#3B82F6' },
                            hovertemplate: `%{x}: %{y:.${decimals}f}<extra></extra>`,
                            text: y.map(v => v.toFixed(decimals)),
                            textposition: 'outside' as const,
                            cliponaxis: false
                          }]}
                          layout={{
                            height: 320,
                            margin: { l: 50, r: 10, t: 10, b: 110 },
                            xaxis: { tickangle: -30, automargin: true },
                            yaxis: { title: { text: 'Proportion' }, rangemode: 'tozero', tickformat: `.${decimals}f` },
                            showlegend: false,
                            paper_bgcolor: '#FAFAFA',
                            plot_bgcolor: '#FAFAFA'
                          }}
                          config={{ displayModeBar: false, responsive: true }}
                          style={{ width: '100%' }}
                        />
                      );
                    })()}
                  </Box>
                )}

                {/* Quality Δ (overall, by metric) */}
                {overallQualityDelta && Object.keys(overallQualityDelta).length > 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mb: 1 }}>
                      <Typography variant="subtitle2" sx={{ color: '#334155' }}>Quality Δ (overall)</Typography>
                      <Tooltip title="Cluster-level quality delta by metric (positive = better than overall)">
                        <IconButton size="small"><InfoOutlinedIcon sx={{ fontSize: 16 }} /></IconButton>
                      </Tooltip>
                    </Stack>
                    {(() => {
                      const entries = Object.entries(overallQualityDelta);
                      const x = entries.map(([metric]) => metric);
                      const y = entries.map(([, val]) => Number(val));
                      const colors = y.map(v => (v >= 0 ? '#10B981' : '#EF4444'));
                      return (
                        <Plot
                          data={[{
                            type: 'bar' as const,
                            x,
                            y,
                            marker: { color: colors },
                            hovertemplate: `%{x}: %{y:.${decimals}f}<extra></extra>`,
                            text: y.map(v => v.toFixed(decimals)),
                            textposition: 'outside' as const,
                            cliponaxis: false
                          }]}
                          layout={{
                            height: 320,
                            margin: { l: 50, r: 10, t: 10, b: 110 },
                            xaxis: { tickangle: -30, automargin: true },
                            yaxis: { title: { text: 'Quality Δ' }, zeroline: true, zerolinecolor: '#94A3B8', tickformat: `.${decimals}f` },
                            showlegend: false,
                            paper_bgcolor: '#FAFAFA',
                            plot_bgcolor: '#FAFAFA'
                          }}
                          config={{ displayModeBar: false, responsive: true }}
                          style={{ width: '100%' }}
                        />
                      );
                    })()}
                  </Box>
                )}

                {/* Per-model quality delta - grouped by metric */}
                {meta.quality_delta_by_model && Object.keys(meta.quality_delta_by_model).length > 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mb: 1 }}>
                      <Typography variant="subtitle2" sx={{ color: '#334155' }}>Quality Delta per Model</Typography>
                      <Tooltip title="Quality delta for each model, grouped by metric">
                        <IconButton size="small"><InfoOutlinedIcon sx={{ fontSize: 16 }} /></IconButton>
                      </Tooltip>
                    </Stack>
                    {(() => {
                      const qualityDeltaByModel: Record<string, Record<string, number>> = meta.quality_delta_by_model;
                      const models = Object.keys(qualityDeltaByModel);
                      const metricKeys = Array.from(new Set(models.flatMap(m => Object.keys(qualityDeltaByModel[m] || {}))));
                      const palette = ['#3B82F6','#10B981','#F59E0B','#EF4444','#8B5CF6','#14B8A6'];
                      
                      // Create traces where each metric is a separate bar group
                      const traces = metricKeys.map((metric, i) => ({
                        type: 'bar',
                        name: metric,
                        x: models,
                        y: models.map(m => Number((qualityDeltaByModel[m] || {})[metric] || 0)),
                        marker: { color: palette[i % palette.length] },
                        hovertemplate: `${metric} · %{x}: %{y:.${decimals}f}<extra></extra>`,
                        text: models.map(m => Number((qualityDeltaByModel[m] || {})[metric] || 0).toFixed(decimals)),
                        textposition: 'outside',
                        cliponaxis: false
                      }));
                      
                      return (
                        <Plot
                          data={traces}
                          layout={{
                            barmode: 'group',
                            height: 360,
                            margin: { l: 50, r: 10, t: 10, b: 110 },
                            xaxis: { title: { text: 'Model' }, tickangle: -30, automargin: true },
                            yaxis: { title: { text: 'Quality Δ' }, tickformat: `.${decimals}f` },
                            paper_bgcolor: '#FAFAFA',
                            plot_bgcolor: '#FAFAFA',
                            legend: { orientation: 'h', y: -0.3, x: 0.5, xanchor: 'center' }
                          }}
                          config={{ displayModeBar: false, responsive: true }}
                          style={{ width: '100%' }}
                        />
                      );
                    })()}
                  </Box>
                )}

                {/* Per-model quality metrics */}
                {perModelQuality && Object.keys(perModelQuality).length > 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mb: 1 }}>
                      <Typography variant="subtitle2" sx={{ color: '#334155' }}>Per-model quality</Typography>
                      <Tooltip title="Quality metrics computed for each model within this cluster">
                        <IconButton size="small"><InfoOutlinedIcon sx={{ fontSize: 16 }} /></IconButton>
                      </Tooltip>
                    </Stack>
                    {(() => {
                      const clusterKey = String(c.id ?? c.label ?? idx);
                      const mode = modeByCluster[clusterKey] || 'quality';
                      const qualitySource: Record<string, Record<string, number>> = mode === 'delta' ? (meta.quality_delta_by_model || {}) : (perModelQuality);
                      const models = Object.keys(qualitySource);
                      const metricKeys = Array.from(new Set(models.flatMap(m => Object.keys(qualitySource[m] || {}))));
                      const palette = ['#3B82F6','#10B981','#F59E0B','#EF4444','#8B5CF6','#14B8A6'];
                      const traces = models.map((m, i) => ({
                        type: 'bar' as const,
                        name: m,
                        x: metricKeys,
                        y: metricKeys.map(k => Number((qualitySource[m] || {})[k] || 0)),
                        marker: { color: palette[i % palette.length] },
                        hovertemplate: `${m} · %{x}: %{y:.${decimals}f}<extra></extra>`,
                        text: metricKeys.map(k => Number((qualitySource[m] || {})[k] || 0).toFixed(decimals)),
                        textposition: 'outside' as const,
                        cliponaxis: false
                      }));
                      return (
                        <Plot
                          data={traces}
                          layout={{
                            barmode: 'group',
                            height: 360,
                            margin: { l: 50, r: 10, t: 10, b: 110 },
                            xaxis: { tickangle: -30, automargin: true },
                            yaxis: { title: { text: mode === 'delta' ? 'Quality Δ' : 'Quality' }, rangemode: 'tozero', tickformat: `.${decimals}f` },
                            paper_bgcolor: '#FAFAFA',
                            plot_bgcolor: '#FAFAFA'
                          }}
                          config={{ displayModeBar: false, responsive: true }}
                          style={{ width: '100%' }}
                        />
                      );
                    })()}
                  </Box>
                )}

                <Typography variant="subtitle2" sx={{ mb: 1, color: '#334155' }}>Properties</Typography>
                {(() => {
                  const hasItems = Array.isArray((c.meta && (c.meta as any).property_items)) && (c.meta as any).property_items.length > 0;
                  const items: any[] = hasItems ? (c.meta as any).property_items : [];
                  const showFromItems = hasItems;
                  const filterByModel = selectedModels.length > 0;
                  if (showFromItems) {
                    const filtered = !filterByModel ? items : items.filter((it: any) => {
                      const m = it?.model != null ? String(it.model) : null;
                      return m ? selectedModels.includes(m) : false;
                    });
                    if (filtered.length > 0) {
                      const limited = filtered.slice(0, 50);
                      return (
                        <Box sx={{ 
                          maxHeight: '400px', 
                          overflow: 'auto',
                          border: '1px solid #E5E7EB', 
                          borderRadius: 1, 
                          background: '#FFFFFF'
                        }}>
                          <Stack spacing={1} sx={{ p: 1 }}>
                            {limited.map((item: any, i: number) => (
                              <Box key={`${c.id}-${item.property_id || i}`} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', border: '1px solid #E5E7EB', borderRadius: 1, p: 1, background: '#FAFAFA' }}>
                                <Box sx={{ mr: 2, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                                  <Typography variant="body2">
                                    {String(item.property_description || '')}
                                  </Typography>
                                  {item.model && (
                                    <Typography 
                                      variant="caption" 
                                      sx={{ 
                                        color: '#6B7280', 
                                        fontWeight: 500,
                                        fontSize: '0.75rem',
                                        backgroundColor: '#F3F4F6',
                                        px: 1,
                                        py: 0.25,
                                        borderRadius: 0.75,
                                        border: '1px solid #E5E7EB'
                                      }}
                                    >
                                      {String(item.model)}
                                    </Typography>
                                  )}
                                </Box>
                                <Button size="small" variant="outlined" onClick={() => {
                                  const pid = item.property_id;
                                  if (pid) onOpenPropertyById(String(pid));
                                }}>Open</Button>
                              </Box>
                            ))}
                          </Stack>
                        </Box>
                      );
                    }
                  }
                  const descriptions: string[] = Array.isArray(c.property_descriptions) ? c.property_descriptions : [];
                  const ids: any[] = Array.isArray((c as any).property_ids) ? (c as any).property_ids : [];
                  const tuples = descriptions.map((pd, i) => ({ pd, pid: ids[i] != null ? String(ids[i]) : undefined }));
                  const filteredTuples = !filterByModel ? tuples : tuples.filter(({ pid }) => {
                    if (!pid) return false;
                    const prop = propertiesById.get(String(pid));
                    const m = prop?.model != null ? String(prop.model) : null;
                    return m ? selectedModels.includes(m) : false;
                  });
                  if (filteredTuples.length > 0) {
                    const limitedTuples = filteredTuples.slice(0, 50);
                    return (
                      <Box sx={{ 
                        maxHeight: '400px', 
                        overflow: 'auto',
                        border: '1px solid #E5E7EB', 
                        borderRadius: 1, 
                        background: '#FFFFFF'
                      }}>
                        <Stack spacing={1} sx={{ p: 1 }}>
                          {limitedTuples.map(({ pd, pid }, i) => {
                            const prop = pid ? propertiesById.get(String(pid)) : null;
                            const modelName = prop?.model != null ? String(prop.model) : null;
                            
                            return (
                              <Box key={`${c.id}-${pid || i}`} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', border: '1px solid #E5E7EB', borderRadius: 1, p: 1, background: '#FAFAFA' }}>
                                <Box sx={{ mr: 2, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                                  <Typography variant="body2">
                                    {pd}
                                  </Typography>
                                  {modelName && (
                                    <Typography 
                                      variant="caption" 
                                      sx={{ 
                                        color: '#6B7280', 
                                        fontWeight: 500,
                                        fontSize: '0.75rem',
                                        backgroundColor: '#F3F4F6',
                                        px: 1,
                                        py: 0.25,
                                        borderRadius: 0.75,
                                        border: '1px solid #E5E7EB'
                                      }}
                                    >
                                      {modelName}
                                    </Typography>
                                  )}
                                </Box>
                                <Button size="small" variant="outlined" onClick={() => { if (pid) onOpenPropertyById(String(pid)); }}>Open</Button>
                              </Box>
                            );
                          })}
                        </Stack>
                      </Box>
                    );
                  }
                  return (
                  <Typography variant="body2" color="text.secondary">No properties listed for this cluster.</Typography>
                  );
                })()}
              </Box>
            </AccordionDetails>
          </Accordion>
        );
        if (animateOnMountRef.current && idx < 20) {
          return (
            <Fade in timeout={Math.min(900 + idx * 140, 2600)} key={`fade-${c.id ?? idx}`}>
              {accordion}
            </Fade>
          );
        }
        return accordion;
      })}
    </Box>
  );
}
export default React.memo(ClustersTab);
