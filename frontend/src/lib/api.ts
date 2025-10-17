export type Method = "single_model" | "side_by_side" | null;

export interface DetectResponse {
  method: Method;
  valid: boolean;
  missing: string[];
  row_count: number;
  columns: string[];
  preview: Record<string, any>[];
}

// Prefer same-origin proxy in dev to avoid CORS/ad-blockers: use /api unless explicitly overridden
const API_BASE = (import.meta as any).env?.VITE_API_BASE || (globalThis as any)?.VITE_API_BASE || "/api";
// Debug print once (won't throw in production)
try { console.debug("[stringsight] API_BASE:", API_BASE); } catch {}

export async function detectAndValidate(file: File): Promise<DetectResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/detect-and-validate`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`detect-and-validate failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function readPath(path: string, limit?: number, method?: "single_model" | "side_by_side") {
  const res = await fetch(`${API_BASE}/read-path`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, limit, method }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`read-path failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function listPath(path: string, exts?: string[]) {
  const res = await fetch(`${API_BASE}/list-path`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, exts }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`list-path failed: ${res.status} ${text}`);
  }
  return res.json();
}

export type ResultsLoadResponse = {
  path: string;
  model_cluster_scores: any[];
  cluster_scores: any[];
  model_scores: any[];
  conversations: any[];
  properties: any[];
  clusters: any[];
};

export async function resultsLoad(path: string, options?: { max_conversations?: number; max_properties?: number }) {
  const res = await fetch(`${API_BASE}/results/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      path,
      max_conversations: options?.max_conversations,
      max_properties: options?.max_properties
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`results-load failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<ResultsLoadResponse>;
}

/**
 * Stream properties data progressively using JSONL endpoint.
 * This allows rendering results as they arrive instead of waiting for the full response.
 */
export async function streamProperties(
  path: string,
  offset: number = 0,
  limit: number = 1000,
  onChunk?: (items: any[]) => void
): Promise<any[]> {
  const params = new URLSearchParams({ path, offset: String(offset), limit: String(limit) });
  const res = await fetch(`${API_BASE}/results/stream/properties?${params.toString()}`);

  if (!res.ok) {
    throw new Error(`stream-properties failed: ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) {
    throw new Error('Stream not supported');
  }

  const decoder = new TextDecoder();
  let buffer = '';
  const items: any[] = [];

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      const batch: any[] = [];
      for (const line of lines) {
        if (line.trim()) {
          try {
            const item = JSON.parse(line);
            items.push(item);
            batch.push(item);
          } catch (e) {
            console.warn('Failed to parse JSONL line:', line);
          }
        }
      }

      if (onChunk && batch.length > 0) {
        onChunk(batch);
      }
    }
  } finally {
    reader.releaseLock();
  }

  return items;
}

/**
 * Stream conversations data progressively using JSONL endpoint.
 */
export async function streamConversations(
  path: string,
  offset: number = 0,
  limit: number = 1000,
  onChunk?: (items: any[]) => void
): Promise<any[]> {
  const params = new URLSearchParams({ path, offset: String(offset), limit: String(limit) });
  const res = await fetch(`${API_BASE}/results/stream/conversations?${params.toString()}`);

  if (!res.ok) {
    throw new Error(`stream-conversations failed: ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) {
    throw new Error('Stream not supported');
  }

  const decoder = new TextDecoder();
  let buffer = '';
  const items: any[] = [];

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      const batch: any[] = [];
      for (const line of lines) {
        if (line.trim()) {
          try {
            const item = JSON.parse(line);
            items.push(item);
            batch.push(item);
          } catch (e) {
            console.warn('Failed to parse JSONL line:', line);
          }
        }
      }

      if (onChunk && batch.length > 0) {
        onChunk(batch);
      }
    }
  } finally {
    reader.releaseLock();
  }

  return items;
}

// DataFrame ops
export async function dfSelect(body: { rows: any[]; include?: Record<string, any[]>; exclude?: Record<string, any[]>; }) {
  const res = await fetch(`${API_BASE}/df/select`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function dfGroupPreview(body: { rows: any[]; by: string; numeric_cols?: string[]; }) {
  const url = `${API_BASE}/df/groupby/preview`;
  console.log('🔴 dfGroupPreview: Making request to:', url);
  console.log('🔴 dfGroupPreview: Request body:', { by: body.by, rows_count: body.rows.length, numeric_cols: body.numeric_cols });
  
  const res = await fetch(url, { 
    method: 'POST', 
    headers: { 'Content-Type': 'application/json' }, 
    body: JSON.stringify(body) 
  });
  
  console.log('🔴 dfGroupPreview: Response status:', res.status, res.statusText);
  
  if (!res.ok) {
    const errorText = await res.text();
    console.log('🔴 dfGroupPreview: Error response:', errorText);
    throw new Error(errorText);
  }
  
  const result = await res.json();
  console.log('🔴 dfGroupPreview: Success response:', result);
  return result;
}

export async function dfGroupRows(body: { rows: any[]; by: string; value: any; page?: number; page_size?: number; }) {
  const res = await fetch(`${API_BASE}/df/groupby/rows`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function dfCustom(body: { rows: any[]; code: string; }) {
  const res = await fetch(`${API_BASE}/df/custom`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


// ----------------------------
// Extraction/Prompts endpoints
// ----------------------------

export async function getPrompts(): Promise<{ prompts: { name: string; label: string; has_task_description: boolean; preview: string; default_task_description_single?: string | null; default_task_description_sbs?: string | null; }[] }> {
  const res = await fetch(`${API_BASE}/prompts`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getPromptText(name: string, task_description?: string | null, method?: 'single_model' | 'side_by_side'): Promise<{ name: string; text: string; }> {
  // Build URL safely when API_BASE can be a relative path (e.g., '/api')
  const params = new URLSearchParams();
  params.set('name', name);
  if (task_description) params.set('task_description', task_description);
  if (method) params.set('method', method);
  const url = `${API_BASE}/prompt-text?${params.toString()}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function extractSingle(body: {
  row: Record<string, any>;
  method?: 'single_model' | 'side_by_side';
  system_prompt?: string;
  task_description?: string | null;
  model_name?: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  max_workers?: number;
  include_scores_in_prompt?: boolean;
  use_wandb?: boolean;
  output_dir?: string | null;
  return_debug?: boolean;
}) {
  const res = await fetch(`${API_BASE}/extract/single`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function extractBatch(body: {
  rows: Record<string, any>[];
  method?: 'single_model' | 'side_by_side';
  system_prompt?: string;
  task_description?: string | null;
  model_name?: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  max_workers?: number;
  chunk_size?: number;
  include_scores_in_prompt?: boolean;
  use_wandb?: boolean;
  output_dir?: string | null;
  return_debug?: boolean;
  sample_size?: number;
}) {
  const res = await fetch(`${API_BASE}/extract/batch`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ----------------------------
// Async batch job endpoints
// ----------------------------

export async function extractJobStart(body: {
  rows: Record<string, any>[];
  method?: 'single_model' | 'side_by_side';
  system_prompt?: string;
  task_description?: string | null;
  model_name?: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  max_workers?: number;
  chunk_size?: number;
  include_scores_in_prompt?: boolean;
  use_wandb?: boolean;
  output_dir?: string | null;
  sample_size?: number;
}) {
  const res = await fetch(`${API_BASE}/extract/jobs/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ job_id: string }>;
}

export async function extractJobStatus(job_id: string) {
  const u = `${API_BASE}/extract/jobs/status?job_id=${encodeURIComponent(job_id)}`;
  const res = await fetch(u);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ job_id: string; state: string; progress: number; count_done: number; count_total: number; error?: string }>;
}

export async function extractJobResult(job_id: string) {
  const u = `${API_BASE}/extract/jobs/result?job_id=${encodeURIComponent(job_id)}`;
  const res = await fetch(u);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ properties: any[]; count: number; cancelled?: boolean }>;
}

export async function extractJobCancel(job_id: string) {
  const res = await fetch(`${API_BASE}/extract/jobs/cancel`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ job_id: string; state: string; message: string; properties_count: number }>;
}


// ----------------------------
// Clustering API
// ----------------------------

export async function getEmbeddingModels(): Promise<{ models: string[] }> {
  const res = await fetch(`${API_BASE}/embedding-models`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function runClustering(body: {
  operationalRows: any[];
  properties: any[];
  params: { minClusterSize?: number | null; embeddingModel: string; groupBy?: 'none' | 'category' | 'behavior_type'; summarizationModel?: string; matchingModel?: string };
}) {
  const res = await fetch(`${API_BASE}/cluster/run`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ 
    clusters: any[]; 
    total_conversations_by_model?: Record<string, number>;
    total_unique_conversations?: number;
    metrics?: {
      model_cluster_scores: any[];
      cluster_scores: any[];
      model_scores: any[];
    };
  }>;
}

export async function recomputeClusterMetrics(body: {
  clusters: any[];
  properties: any[];
  operationalRows: any[];
  included_property_ids?: string[];
}) {
  const res = await fetch(`${API_BASE}/cluster/metrics`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ 
    clusters: any[]; 
    total_conversations_by_model?: Record<string, number>;
    total_unique_conversations?: number;
  }>;
}

