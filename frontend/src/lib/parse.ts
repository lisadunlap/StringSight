import Papa from "papaparse";

export type ParsedData = { rows: Record<string, any>[]; columns: string[] };

export async function parseFile(file: File): Promise<ParsedData> {
  const name = file.name.toLowerCase();
  const text = await file.text();
  if (name.endsWith(".jsonl")) {
    const rows = text
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean)
      .map((l) => JSON.parse(l));
    return { rows, columns: inferColumns(rows) };
  }
  if (name.endsWith(".json")) {
    const data = JSON.parse(text);
    const rows = Array.isArray(data) ? data : [data];
    return { rows, columns: inferColumns(rows) };
  }
  if (name.endsWith(".csv")) {
    const res = Papa.parse<Record<string, any>>(text, { header: true, skipEmptyLines: true });
    const rows = res.data as Record<string, any>[];
    return { rows, columns: inferColumns(rows) };
  }
  throw new Error("Unsupported file type. Use JSONL, JSON, or CSV.");
}

export function inferColumns(rows: Record<string, any>[]): string[] {
  const cols = new Set<string>();
  for (const r of rows.slice(0, 100)) {
    Object.keys(r || {}).forEach((k) => cols.add(k));
  }
  return Array.from(cols);
}


