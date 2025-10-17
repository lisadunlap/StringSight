/**
 * TagChips - Cluster metadata tags display.
 * 
 * Renders metadata fields as small chips with subtle styling.
 * Common fields: group, tag, category, etc.
 */

import React from 'react';
import {
  Chip,
  Stack
} from '@mui/material';

interface TagChipsProps {
  metadata: Record<string, any>;
  maxTags?: number;
}

export function TagChips({
  metadata,
  maxTags = 3
}: TagChipsProps) {
  
  // Extract meaningful tags from metadata
  const tagFields = ['group', 'tag', 'category', 'type', 'style'];
  const tags: string[] = [];
  
  for (const field of tagFields) {
    const value = metadata[field];
    if (value && typeof value === 'string' && value.trim()) {
      tags.push(value.trim());
    }
  }
  
  // Remove duplicates and limit count
  const uniqueTags = [...new Set(tags)].slice(0, maxTags);
  
  if (uniqueTags.length === 0) {
    return null;
  }

  const colorForTag = (label: string) => {
    const norm = label.trim().toLowerCase();
    if (norm === 'style') {
      return { text: '#6d28d9', bg: '#faf5ff', border: '#ede9fe' };
    }
    if (norm === 'positive') {
      return { text: '#28a745', bg: '#28a7451A', border: '#28a74533' };
    }
    if (norm === 'negative (non-critical)' || norm === 'negative non-critical' || norm === 'negative (non critical)') {
      return { text: '#ff7f0e', bg: '#ff7f0e1A', border: '#ff7f0e33' };
    }
    if (norm === 'negative (critical)' || norm === 'negative critical') {
      return { text: '#dc3545', bg: '#dc35451A', border: '#dc354533' };
    }
    // default
    return { text: '#4c6ef5', bg: '#4c6ef51A', border: '#4c6ef533' };
  };

  return (
    <Stack direction="row" spacing={0.5} sx={{ flexWrap: 'wrap', gap: 0.5 }}>
      {uniqueTags.map((tag, index) => {
        const c = colorForTag(tag);
        return (
          <Chip
            key={index}
            label={tag}
            size="small"
            variant="outlined"
            sx={{
              height: 18,
              fontSize: '0.65rem',
              color: c.text,
              borderColor: c.border,
              backgroundColor: c.bg,
              '& .MuiChip-label': {
                px: 0.75
              }
            }}
          />
        );
      })}
    </Stack>
  );
}

export default TagChips;
