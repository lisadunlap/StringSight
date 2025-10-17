import React from 'react';
import { Box, Typography } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import DOMPurify from 'dompurify';
import 'katex/dist/katex.min.css';

export interface HighlightRange {
  start: number;
  end: number;
  color?: string;
}

interface ResponseContentProps {
  content: string;
  highlightedRanges?: HighlightRange[];
  className?: string;
}

/**
 * Base component for rendering model response content with optional highlighting.
 * - If highlightedRanges provided, render plain text with inline highlights (preserves index-based spans)
 * - Otherwise, detect and render Markdown/LaTeX/HTML similarly to legacy cards
 */
export default function ResponseContent({ content, highlightedRanges = [], className }: ResponseContentProps) {
  // If highlighting is requested, render the simple highlighter to preserve index-based ranges
  if (highlightedRanges && highlightedRanges.length > 0) {
    const sortedRanges = [...highlightedRanges].sort((a, b) => a.start - b.start);
    const segments: React.ReactNode[] = [];
    let currentPos = 0;

    sortedRanges.forEach((range, index) => {
      const { start, end, color = '#ffeb3b' } = range;
      const validStart = Math.max(0, Math.min(start, content.length));
      const validEnd = Math.max(validStart, Math.min(end, content.length));

      if (currentPos < validStart) {
        segments.push(<span key={`text-${index}`}>{content.slice(currentPos, validStart)}</span>);
      }
      if (validStart < validEnd) {
        segments.push(
          <Box
            key={`highlight-${index}`}
            component="span"
            sx={{
              backgroundColor: color,
              padding: '2px 4px',
              borderRadius: '3px',
              fontWeight: 600,
              border: '1px solid rgba(0,0,0,0.1)'
            }}
          >
            {content.slice(validStart, validEnd)}
          </Box>
        );
      }
      currentPos = validEnd;
    });

    if (currentPos < content.length) {
      segments.push(<span key="text-final">{content.slice(currentPos)}</span>);
    }

    return (
      <Typography
        variant="body2"
        className={className}
        sx={{
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          fontFamily: 'monospace',
          fontSize: '0.875rem',
          lineHeight: 1.5,
        }}
      >
        {segments}
      </Typography>
    );
  }

  // No highlighting: detect content type and render like legacy cards
  const hasMarkdown = /[#*`_\[\](){}]|^\s*[-+*]\s|^\s*\d+\.\s/m.test(content);
  const hasLaTeX = /\$\$[^$]*\$\$|\$[^$]+\$|\\[a-zA-Z]+\{/.test(content);
  const hasHTML = /<[^>]+>/.test(content);

  if (hasHTML) {
    const sanitizedHTML = DOMPurify.sanitize(content, {
      ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 'code', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'blockquote', 'a'],
      ALLOWED_ATTR: ['href', 'target', 'rel'],
    });
    return (
      <Box
        sx={{
          '& p': { margin: '4px 0' },
          '& code': { backgroundColor: '#f5f5f5', padding: '2px 4px', borderRadius: '4px' },
          '& pre': { backgroundColor: '#f5f5f5', padding: '8px', borderRadius: '4px', overflow: 'auto' },
          '& h1, & h2, & h3, & h4, & h5, & h6': { margin: '8px 0 4px 0', fontWeight: 600 },
          '& ul, & ol': { margin: '4px 0', paddingLeft: '20px' },
          '& blockquote': { borderLeft: '3px solid #ddd', paddingLeft: '12px', margin: '4px 0', fontStyle: 'italic' },
          fontFamily: 'monospace',
          fontSize: '0.875rem',
          lineHeight: 1.5,
        }}
        dangerouslySetInnerHTML={{ __html: sanitizedHTML }}
      />
    );
  }

  if (hasMarkdown || hasLaTeX) {
    return (
      <Box
        sx={{
          '& p': { margin: '4px 0' },
          '& code': { backgroundColor: '#f5f5f5', padding: '2px 4px', borderRadius: '4px', fontSize: '0.9em' },
          '& pre': { backgroundColor: '#f5f5f5', padding: '8px', borderRadius: '4px', overflow: 'auto' },
          '& h1, & h2, & h3, & h4, & h5, & h6': { margin: '8px 0 4px 0', fontWeight: 600 },
          '& ul, & ol': { margin: '4px 0', paddingLeft: '20px' },
          '& blockquote': { borderLeft: '3px solid #ddd', paddingLeft: '12px', margin: '4px 0' },
          '& .katex': { fontSize: '1em' },
          '& .katex-display': { margin: '8px 0' },
          fontFamily: 'monospace',
          fontSize: '0.875rem',
          lineHeight: 1.5,
        }}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex]}
          components={{
            a: ({ href, children, ...props }) => (
              <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
                {children}
              </a>
            ),
            p: ({ children }) => <span>{children}</span>,
          }}
        >
          {content}
        </ReactMarkdown>
      </Box>
    );
  }

  // Fallback plain text
  return (
    <Typography
      variant="body2"
      className={className}
      sx={{
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        fontFamily: 'monospace',
        fontSize: '0.875rem',
        lineHeight: 1.5,
      }}
    >
      {content}
    </Typography>
  );
}

/**
 * Utility function to convert evidence strings to highlight ranges
 */
export function evidenceToHighlightRanges(
  content: string,
  evidence: string[],
  _targetModel?: string
): HighlightRange[] {
  if (!evidence || evidence.length === 0) return [];
  const ranges: HighlightRange[] = [];
  evidence.forEach((evidenceText) => {
    if (!evidenceText || typeof evidenceText !== 'string') return;
    let searchStart = 0;
    while (searchStart < content.length) {
      const index = content.toLowerCase().indexOf(evidenceText.toLowerCase(), searchStart);
      if (index === -1) break;
      ranges.push({ start: index, end: index + evidenceText.length, color: '#ffeb3b' });
      searchStart = index + evidenceText.length;
    }
  });
  return ranges;
}
