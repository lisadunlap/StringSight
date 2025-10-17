export type Method = "single_model" | "side_by_side" | "unknown";

// Flatten score dictionaries into scalar columns: score_* (single), score_a_*, score_b_* (sbs)
export function flattenScores(rows: Record<string, any>[], method: Method) {
  console.log('🔄 DEBUG flattenScores called:', { rowCount: rows.length, method, sampleRow: rows[0] });
  const out = rows.map((r) => ({ ...r }));

  function flattenField(field: string, prefix: string) {
    console.log(`🔍 DEBUG flattenField called for field "${field}" with prefix "${prefix}"`);
    const keySet = new Set<string>();
    for (const row of out) {
      const val = row[field];
      if (val && typeof val === "object" && !Array.isArray(val)) {
        for (const k of Object.keys(val)) keySet.add(k);
      }
    }
    console.log(`🔍 DEBUG found keys for ${field}:`, Array.from(keySet));
    if (keySet.size === 0) {
      console.log(`⚠️ DEBUG: No keys found for field "${field}", skipping flattening`);
      return;
    }
    for (const row of out) {
      const val = row[field] || {};
      for (const k of keySet) {
        const col = `${prefix}_${k}`;
        const v = val && typeof val === "object" ? val[k] : undefined;
        (row as any)[col] = v;
        console.log(`✅ DEBUG: Created column "${col}" with value:`, v);
      }
      delete (row as any)[field];
    }
    console.log(`🗑️ DEBUG: Deleted original field "${field}"`);
  }

  if (method === "single_model") {
    const hasScoreObjects = out.some((r) => typeof r?.score === "object");
    console.log(`DEBUG single_model: hasScoreObjects=${hasScoreObjects}, sampleScore:`, out[0]?.score);
    if (hasScoreObjects) {
      flattenField("score", "score");
    }
  } else if (method === "side_by_side") {
    const hasScoreAObjects = out.some((r) => typeof r?.score_a === "object");
    const hasScoreBObjects = out.some((r) => typeof r?.score_b === "object");
    console.log(`DEBUG side_by_side: hasScoreAObjects=${hasScoreAObjects}, hasScoreBObjects=${hasScoreBObjects}`);
    console.log(`DEBUG sample scores:`, { score_a: out[0]?.score_a, score_b: out[0]?.score_b });
    if (hasScoreAObjects) {
      flattenField("score_a", "score_a");
    }
    if (hasScoreBObjects) {
      flattenField("score_b", "score_b");
    }
  }

  const columnSet = out.reduce<Set<string>>((set, r) => {
    Object.keys(r).forEach((k) => set.add(k));
    return set;
  }, new Set<string>());
  
  const columns = Array.from(columnSet);

  return { rows: out, columns };
}


