import React from 'react';
import { Card, CardHeader, CardContent, Typography, Chip, Stack, Box, Button } from '@mui/material';
import ModelResponseCard from './ModelResponseCard';
import { evidenceToHighlightRanges } from './ResponseContent';

export interface ConversationLike {
  question_id: string;
  prompt?: string;
  model: string | [string, string];
  responses: string | [string, string];
  scores?: Record<string, any> | [Record<string, any>, Record<string, any>];
}

export interface PropertyLike {
  id: string;
  question_id: string;
  model: string;
  property_description?: string;
  category?: string;
  behavior_type?: string;
  reason?: string;
  evidence?: string | string[];
  contains_errors?: boolean;
  unexpected_behavior?: boolean;
}

interface PropertyCardProps {
  property: PropertyLike;
  conversation: ConversationLike | null;
  method: 'single_model' | 'side_by_side';
  onOpenConversation?: () => void;
  clusterLabel?: string; // optional label to show when used in clusters
}

export default function PropertyCard({ property, conversation, method, onOpenConversation, clusterLabel }: PropertyCardProps) {
  const description = property.property_description || '(No description)';
  const evidenceList = Array.isArray(property.evidence) ? property.evidence : property.evidence ? [property.evidence] : [];

  // Compute highlight ranges when conversation and evidence exist
  const highlightedRanges = React.useMemo(() => {
    if (!conversation || !evidenceList.length) return [];
    const resp = method === 'single_model' ? String(conversation.responses || '') : '';
    return evidenceToHighlightRanges(resp, evidenceList);
  }, [conversation, evidenceList, method]);

  return (
    <Card sx={{ border: '1px solid #E5E7EB', mb: 2 }}>
      <CardHeader
        title={
          <Box>
            {clusterLabel && (
              <Typography variant="overline" sx={{ display: 'block', color: '#64748B', mb: 0.5 }}>
                {clusterLabel}
              </Typography>
            )}
            <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 1 }}>
              {description}
            </Typography>
            <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', rowGap: 1 }}>
              {property.category && (
                <Chip label={property.category} size="small" variant="outlined" />
              )}
              {property.behavior_type && (
                <Chip 
                  label={property.behavior_type}
                  size="small"
                  color={property.behavior_type?.toLowerCase().includes('positive') ? 'success' : property.behavior_type?.toLowerCase().includes('negative') ? 'error' : 'default'}
                  variant="outlined"
                />
              )}
              <Chip label={property.model} size="small" variant="outlined" />
              {property.unexpected_behavior && (
                <Chip label="Unexpected behavior" size="small" color="warning" variant="outlined" />
              )}
              {property.contains_errors && (
                <Chip label="Contains errors" size="small" color="error" variant="outlined" />
              )}
            </Stack>
            {evidenceList.length > 0 && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.75, display: 'block' }}>
                Evidence: {evidenceList.join(' | ')}
              </Typography>
            )}
          </Box>
        }
        sx={{ pb: 0.5 }}
      />
      <CardContent>
        {/* Reason */}
        {property.reason && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 0.5, color: '#334155' }}>Reason</Typography>
            <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>{property.reason}</Typography>
          </Box>
        )}

        {/* Evidence and conversation */}
        {conversation && method === 'single_model' && (
          <Box>
            {evidenceList.length > 0 && (
              <Typography variant="subtitle2" sx={{ mb: 1, color: '#334155' }}>Evidence highlighting</Typography>
            )}
            <ModelResponseCard
              modelName={typeof conversation.model === 'string' ? conversation.model : String(conversation.model)}
              response={typeof conversation.responses === 'string' ? conversation.responses : String(conversation.responses)}
              highlightedRanges={highlightedRanges}
              metadata={
                typeof conversation.scores === 'object' && !Array.isArray(conversation.scores)
                  ? { score: conversation.scores }
                  : {}
              }
              variant="compact"
            />
          </Box>
        )}

        {!conversation && (
          <Box sx={{ p: 2, border: '1px dashed #E5E7EB', borderRadius: 1, background: '#FAFAFA' }}>
            <Typography variant="body2" color="text.secondary">Conversation not available for this property.</Typography>
          </Box>
        )}

        {/* Actions */}
        {onOpenConversation && (
          <Box sx={{ mt: 1 }}>
            <Button variant="outlined" size="small" onClick={onOpenConversation}>Open in sidebar</Button>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
