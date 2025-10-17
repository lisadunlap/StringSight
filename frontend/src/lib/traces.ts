export type Role = "user" | "assistant" | "system" | "tool" | string;
export interface Message {
  role: Role;
  content: string | { text?: string; image?: string; tool_calls?: any[] } | any;
  name?: string;
  id?: string;
}

export interface SingleTrace { questionId: string; prompt: string; messages: Message[] }
export interface SideBySideTrace {
  questionId: string;
  prompt: string;
  modelA: string;
  modelB: string;
  messagesA: Message[];
  messagesB: Message[];
}

export function detectMethodFromColumns(cols: string[]): "single_model" | "side_by_side" | "unknown" {
  const single = ["prompt", "model", "model_response"].every((c) => cols.includes(c));
  const sbs = ["prompt", "model_a", "model_b", "model_a_response", "model_b_response"].every((c) => cols.includes(c));
  if (sbs) return "side_by_side";
  if (single) return "single_model";
  return "unknown";
}

export function formatSingleTraceFromRow(row: Record<string, any>): SingleTrace {
  const prompt = String(row["prompt"] ?? "");
  const response = row["model_response"];
  const messages = ensureOpenAIFormat(prompt, response);
  return { questionId: String(row["question_id"] ?? ""), prompt, messages };
}

export function formatSideBySideTraceFromRow(row: Record<string, any>): SideBySideTrace {
  const prompt = String(row["prompt"] ?? "");
  const messagesA = ensureOpenAIFormat(prompt, row["model_a_response"]);
  const messagesB = ensureOpenAIFormat(prompt, row["model_b_response"]);
  return {
    questionId: String(row["question_id"] ?? ""),
    prompt,
    modelA: String(row["model_a"] ?? "Model A"),
    modelB: String(row["model_b"] ?? "Model B"),
    messagesA,
    messagesB,
  };
}

export function ensureOpenAIFormat(prompt: string, response: any): Message[] {
  let parsedResponse = response;

  // If response is already a valid array of messages, return it directly
  if (Array.isArray(response)) {
    const hasValidMessages = response.some((m: any) => m && typeof m.role === 'string' && typeof m.content !== 'undefined');
    if (hasValidMessages) {
      return response.map((m: any) => ({
        role: m.role as Role,
        content: m.content,
        name: m.name,
        id: m.id
      }));
    }
  }

  // Try to parse if it's a string that looks like JSON or Python list
  if (typeof response === 'string' && (response.trim().startsWith('[') || response.trim().startsWith('{'))) {
    try {
      // First try standard JSON parse
      parsedResponse = JSON.parse(response);
    } catch (e) {
      // If that fails, try parsing Python-style repr with safe quote replacement
      // This handles legacy data that was saved with Python string repr
      try {
        // More sophisticated Python literal parsing:
        // First, handle Python-specific values
        let jsonified = response
          .replace(/:\s*None\b/g, ': null')
          .replace(/,\s*None\b/g, ', null')
          .replace(/\[\s*None\b/g, '[null')
          .replace(/:\s*True\b/g, ': true')
          .replace(/,\s*True\b/g, ', true')
          .replace(/:\s*False\b/g, ': false')
          .replace(/,\s*False\b/g, ', false');

        // Now handle quote conversion more carefully
        // Strategy: Use a state machine to track if we're inside a string
        // and only replace quotes outside of strings
        let result = '';
        let inString = false;
        let escapeNext = false;

        for (let i = 0; i < jsonified.length; i++) {
          const char = jsonified[i];
          const prevChar = i > 0 ? jsonified[i - 1] : '';

          if (escapeNext) {
            result += char;
            escapeNext = false;
            continue;
          }

          if (char === '\\') {
            escapeNext = true;
            result += char;
            continue;
          }

          if (char === "'") {
            // Toggle string state and convert to double quote
            inString = !inString;
            result += '"';
          } else {
            result += char;
          }
        }

        parsedResponse = JSON.parse(result);
      } catch (e2) {
        // If both fail, treat as plain string
        console.warn('[traces.ensureOpenAIFormat] Failed to parse response string:', e2);
        console.warn('[traces.ensureOpenAIFormat] Response preview:', response.substring(0, 200));
      }
    }
  }

  // If already an array shaped like OpenAI messages, validate it
  if (Array.isArray(parsedResponse)) {
    const msgs = parsedResponse as any[];
    const hasValidMessages = msgs.some((m) => m && typeof m.role === 'string' && typeof m.content !== 'undefined');
    // If it has messages with roles and content, use as-is after normalization
    if (hasValidMessages) {
      const normalized: Message[] = msgs
        .filter((m) => m && typeof m.role === 'string' && typeof m.content !== 'undefined')
        .map((m) => {
          let content = m.content;
          // If content is an object with a 'text' field that looks like stringified data, try to parse it
          if (typeof content === 'object' && content !== null && content.text && typeof content.text === 'string') {
            if (content.text.trim().startsWith('[') || content.text.trim().startsWith('{')) {
              try {
                const parsed = JSON.parse(content.text);
                // If it's an object or array, keep the text as-is for now
                // (we don't want to double-parse)
              } catch (e) {
                // Not parseable, keep as-is
              }
            }
          }
          return {
            role: m.role as Role,
            content: content,
            name: m.name,
            id: m.id
          };
        });
      return normalized;
    }
    // Fallback to synthesized pair below
    console.warn('[traces.ensureOpenAIFormat] Missing user/assistant in array; falling back to synthesized messages');
  }

  // Common case: response is a string (model output), prompt is user content
  const assistantText = typeof parsedResponse === 'string' ? parsedResponse : String(parsedResponse ?? '');
  const userText = typeof prompt === 'string' ? prompt : String(prompt ?? '');

  // Debug: log if we're falling back to stringification (this indicates a data issue)
  if (typeof parsedResponse !== 'string' && parsedResponse != null) {
    console.warn('[traces.ensureOpenAIFormat] Converting non-string response to string - data may be incorrectly formatted');
    console.warn('[traces.ensureOpenAIFormat] Response type:', typeof parsedResponse);
    console.warn('[traces.ensureOpenAIFormat] Response preview:', JSON.stringify(parsedResponse).substring(0, 200));
  }

  return [
    { role: 'user', content: userText },
    { role: 'assistant', content: assistantText },
  ];
}


