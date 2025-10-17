import React from 'react';
import { Card, CardHeader, CardContent, Typography, Box, Stack, Chip } from '@mui/material';
import PropertyCard, { type PropertyLike, type ConversationLike } from './PropertyCard';

interface ClusterLike {
  id: string | number;
  label: string;
  size?: number;
}

interface ClusterCardProps {
  cluster: ClusterLike;
  properties: PropertyLike[];
  conversations: ConversationLike[];
  method: 'single_model' | 'side_by_side';
}

export default function ClusterCard({ cluster, properties, conversations, method }: ClusterCardProps) {
  const findConversation = (p: PropertyLike): ConversationLike | null => {
    return (
      conversations.find(c => c.question_id === p.question_id && (
        (typeof c.model === 'string' && c.model === p.model) ||
        (Array.isArray(c.model) && (c.model[0] === p.model || c.model[1] === p.model))
      )) || null
    );
  };

  return (
    <Card sx={{ border: '1px solid #E5E7EB', mb: 3 }}>
      <CardHeader
        title={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="h6" sx={{ fontWeight: 700 }}>{cluster.label}</Typography>
            {typeof cluster.size === 'number' && (
              <Chip label={`Size: ${cluster.size}`} size="small" variant="outlined" />
            )}
          </Box>
        }
      />
      <CardContent>
        <Stack spacing={2}>
          {properties.map((prop) => (
            <PropertyCard
              key={prop.id}
              property={prop}
              conversation={findConversation(prop)}
              method={method}
              clusterLabel={cluster.label}
            />
          ))}
        </Stack>
      </CardContent>
    </Card>
  );
}
