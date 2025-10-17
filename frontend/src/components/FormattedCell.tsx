import React, { useState, useMemo } from 'react';
import { Box, Button } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import DOMPurify from 'dompurify';
import 'katex/dist/katex.min.css';

interface FormattedCellProps {
  text: string;
  maxLength?: number;
  isPrompt?: boolean;
}

const FormattedCell: React.FC<FormattedCellProps> = ({ 
  text, 
  maxLength = 200, 
  isPrompt = false 
}) => {
  const [expanded, setExpanded] = useState(false);

  // Detect content type and determine if we should render as formatted content
  const contentAnalysis = useMemo(() => {
    if (!isPrompt || !text) {
      return { hasFormatting: false, type: 'plain' };
    }

    const hasMarkdown = /[#*`_\[\](){}]|^\s*[-+*]\s|^\s*\d+\.\s/m.test(text);
    const hasLaTeX = /\$\$[\s\S]*?\$\$|\$[^$]+\$|\\[a-zA-Z]+\{/.test(text);
    const hasHTML = /<[^>]+>/.test(text);
    
    if (hasHTML) return { hasFormatting: true, type: 'html' };
    if (hasLaTeX || hasMarkdown) return { hasFormatting: true, type: 'markdown' };
    
    return { hasFormatting: false, type: 'plain' };
  }, [text, isPrompt]);

  // Render content based on type
  const renderContent = (content: string, truncated: boolean = false) => {
    if (!contentAnalysis.hasFormatting) {
      return <span>{content}</span>;
    }

    if (contentAnalysis.type === 'html') {
      const sanitizedHTML = DOMPurify.sanitize(content, {
        ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 'code', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'blockquote', 'a'],
        ALLOWED_ATTR: ['href', 'target', 'rel']
      });
      
      return (
        <Box 
          sx={{ 
            '& p': { margin: '4px 0' },
            '& code': { backgroundColor: '#f5f5f5', padding: '2px 4px', borderRadius: '4px' },
            '& pre': { backgroundColor: '#f5f5f5', padding: '8px', borderRadius: '4px', overflow: 'auto' },
            '& h1, & h2, & h3, & h4, & h5, & h6': { margin: '8px 0 4px 0', fontWeight: 600 },
            '& ul, & ol': { margin: '4px 0', paddingLeft: '20px' },
            '& blockquote': { borderLeft: '3px solid #ddd', paddingLeft: '12px', margin: '4px 0', fontStyle: 'italic' }
          }}
          dangerouslySetInnerHTML={{ __html: sanitizedHTML }}
        />
      );
    }

    if (contentAnalysis.type === 'markdown') {
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
            '& .katex-display': { margin: '8px 0' }
          }}
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={{
              // Custom link handling for security
              a: ({ href, children, ...props }) => (
                <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
                  {children}
                </a>
              ),
              // Prevent excessive nesting
              p: ({ children }) => <span>{children}</span>
            }}
          >
            {content}
          </ReactMarkdown>
        </Box>
      );
    }

    return <span>{content}</span>;
  };

  // Handle truncation logic
  const shouldTruncate = !expanded && text.length > maxLength;
  const displayText = shouldTruncate ? text.slice(0, maxLength) : text;
  const needsTruncation = text.length > maxLength;

  return (
    <Box>
      {renderContent(displayText, shouldTruncate)}
      {shouldTruncate && <span>…</span>}
      {needsTruncation && (
        <>
          {' '}
          <Button 
            size="small" 
            variant="text" 
            onClick={() => setExpanded(!expanded)}
            sx={{ minWidth: 'auto', padding: '2px 4px', fontSize: '0.75em' }}
          >
            {expanded ? 'Collapse' : 'Expand'}
          </Button>
        </>
      )}
    </Box>
  );
};

export default FormattedCell;
