import React from "react";
import type { Message } from "../lib/traces";
import { Box, Typography, Chip, Stack, Accordion, AccordionSummary, AccordionDetails } from "@mui/material";
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

function escapeRegex(s: string): string { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

// Helper to extract text content from various content formats
function getTextContent(content: any): string {
  if (typeof content === 'string') return content;
  if (typeof content === 'object' && content !== null) {
    if (content.text) return String(content.text);
    if (content.content) return String(content.content);
    return JSON.stringify(content, null, 2);
  }
  return String(content ?? '');
}

// Minimal highlighting: escapes and marks exact terms

// Keep rendering simple; no delimiter normalization here

// No evidence fragment splitting; highlights are exact term matches

// No flexible regex; use simple exact highlighting only

function highlightContent(text: string, highlights?: string[]): Array<string | React.ReactNode> {
  if (!highlights || highlights.length === 0) return [text];
  let nodes: Array<string | React.ReactNode> = [text];
  for (const term of highlights) {
    const pattern = escapeRegex(String(term || '').trim());
    if (!pattern) continue;
    const regex = new RegExp(pattern, 'gi');
    const next: Array<string | React.ReactNode> = [];
    for (const node of nodes) {
      if (typeof node !== 'string') { next.push(node); continue; }
      let last = 0; let m: RegExpExecArray | null;
      while ((m = regex.exec(node)) !== null) {
        if (m.index > last) next.push(node.slice(last, m.index));
        next.push(
          <mark key={`${m.index}-${Math.random()}`} style={{ backgroundColor: '#FEF08A', padding: 0 }}>
            {m[0]}
          </mark>
        );
        last = m.index + m[0].length;
        if (regex.lastIndex === m.index) regex.lastIndex++;
      }
      if (last < node.length) next.push(node.slice(last));
    }
    nodes = next;
  }
  return nodes;
}

// Recursively apply highlighting to React children
function applyHighlightToChildren(children: React.ReactNode, highlights?: string[]): React.ReactNode {
  if (!highlights || highlights.length === 0) return children;

  if (typeof children === 'string') {
    return highlightContent(children, highlights);
  }

  if (Array.isArray(children)) {
    return children.map((child, i) => (
      <React.Fragment key={i}>{applyHighlightToChildren(child, highlights)}</React.Fragment>
    ));
  }

  return children;
}

// Helper function to apply regex highlighting to nodes
// applyHighlightRegex is unused in the simplified version; removing to keep surface area minimal

export function ConversationTrace({ messages, highlights, rawResponse }: { messages: Message[]; highlights?: string[]; rawResponse?: any }) {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
      {messages.map((m, i) => {
        const isStructuredContent = typeof m.content === 'object' && m.content !== null;
        const hasToolCalls = isStructuredContent && m.content.tool_calls;
        const content = getTextContent(m.content);

        // Determine background color based on role
        const getBackgroundColor = (role: string) => {
          if (role === "user") return "#f8fafc";
          if (role === "tool") return "#fef3c7";
          if (role === "info") return "#f0fdf4";
          return "#ffffff";
        };

        return (
          <Box key={i} sx={{
            p: 1.5,
            border: "1px solid #e5e7eb",
            borderRadius: 1,
            backgroundColor: getBackgroundColor(m.role),
          }}>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
              <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'capitalize' }}>
                {m.role}
              </Typography>
              {m.name && (
                <Chip label={m.name} size="small" variant="outlined" sx={{ height: '18px', fontSize: '0.65rem' }} />
              )}
            </Stack>

            {hasToolCalls && Array.isArray(m.content.tool_calls) && (
              <Box sx={{ mb: 1 }}>
                {m.content.tool_calls.map((tc: any, idx: number) => (
                  <Box key={idx} sx={{
                    mb: 0.5,
                    p: 1,
                    backgroundColor: '#f0f9ff',
                    border: '1px solid #bae6fd',
                    borderRadius: 1
                  }}>
                    <Typography variant="caption" sx={{ fontWeight: 600, color: '#0369a1' }}>
                      🔧 {tc.name || 'Tool Call'}
                    </Typography>
                    {tc.arguments && (
                      <Typography variant="body2" sx={{
                        fontFamily: 'monospace',
                        fontSize: '0.75rem',
                        whiteSpace: 'pre-wrap',
                        mt: 0.5,
                        color: '#475569'
                      }}>
                        {typeof tc.arguments === 'string' ? tc.arguments : JSON.stringify(tc.arguments, null, 2)}
                      </Typography>
                    )}
                  </Box>
                ))}
              </Box>
            )}

            {content && content.trim() && (() => {
            const hasMarkdown = /[#*`_\[\](){}]|^\s*[-+*]\s|^\s*\d+\.\s/m.test(content);
            const hasLaTeX = /\$\$[^$]*\$\$|\$[^$]+\$|\\[a-zA-Z]+\{|\\\\\(|\\\\\[/.test(content);

            // Render Markdown/LaTeX if detected
            if (hasMarkdown || hasLaTeX) {
              return (
                <Box
                  sx={{
                    '& p': { margin: '4px 0' },
                    '& code': { backgroundColor: '#f5f5f5', padding: '2px 4px', borderRadius: '4px', fontSize: '0.9em', fontFamily: 'monospace' },
                    '& pre': { backgroundColor: '#f5f5f5', padding: '8px', borderRadius: '4px', overflow: 'auto' },
                    '& pre code': { fontFamily: 'monospace' },
                    '& h1, & h2, & h3, & h4, & h5, & h6': { margin: '8px 0 4px 0', fontWeight: 600 },
                    '& ul, & ol': { margin: '4px 0', paddingLeft: '20px' },
                    '& blockquote': { borderLeft: '3px solid #ddd', paddingLeft: '12px', margin: '4px 0' },
                    '& .katex': { fontSize: '1em' },
                    '& .katex-display': { margin: '8px 0' },
                    fontSize: '0.875rem',
                    lineHeight: 1.5,
                  }}
                >
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                    components={{
                      p: ({ children }) => <p style={{ margin: '4px 0' }}>{applyHighlightToChildren(children, highlights)}</p>,
                      span: ({ children }) => <span>{applyHighlightToChildren(children, highlights)}</span>,
                      li: ({ children }) => <li>{applyHighlightToChildren(children, highlights)}</li>,
                      strong: ({ children }) => <strong>{applyHighlightToChildren(children, highlights)}</strong>,
                      em: ({ children }) => <em>{applyHighlightToChildren(children, highlights)}</em>,
                      h1: ({ children }) => <h1>{applyHighlightToChildren(children, highlights)}</h1>,
                      h2: ({ children }) => <h2>{applyHighlightToChildren(children, highlights)}</h2>,
                      h3: ({ children }) => <h3>{applyHighlightToChildren(children, highlights)}</h3>,
                      h4: ({ children }) => <h4>{applyHighlightToChildren(children, highlights)}</h4>,
                      h5: ({ children }) => <h5>{applyHighlightToChildren(children, highlights)}</h5>,
                      h6: ({ children }) => <h6>{applyHighlightToChildren(children, highlights)}</h6>,
                    }}
                  >
                    {content}
                  </ReactMarkdown>
                </Box>
              );
            }

            // Fallback plain text with highlighting
            return (
              <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                {highlights && highlights.length > 0 ? highlightContent(content, highlights) : content}
              </Typography>
            );
          })()}
          </Box>
        );
      })}

      {/* View Raw Response Accordion */}
      {rawResponse && (
        <Accordion sx={{ mt: 1 }}>
          <AccordionSummary
            expandIcon={<ExpandMoreIcon />}
            sx={{
              backgroundColor: '#f8fafc',
              '&:hover': { backgroundColor: '#f1f5f9' },
            }}
          >
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              View Raw Response
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box
              sx={{
                backgroundColor: '#f5f5f5',
                padding: 2,
                borderRadius: 1,
                overflow: 'auto',
                maxHeight: '400px',
              }}
            >
              <Typography
                variant="body2"
                component="pre"
                sx={{
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  margin: 0,
                }}
              >
                {typeof rawResponse === 'string' ? rawResponse : JSON.stringify(rawResponse, null, 2)}
              </Typography>
            </Box>
          </AccordionDetails>
        </Accordion>
      )}
    </Box>
  );
}

export default ConversationTrace;


