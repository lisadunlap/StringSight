import React, { useState } from 'react';
import { Box, Typography, Button, Stack, Divider } from '@mui/material';
import ResponseContent, { evidenceToHighlightRanges } from './ResponseContent';
import ModelResponseCard, { SideBySideResponse } from './ModelResponseCard';
import PropertyCard from './PropertyCard';

/**
 * Test page for the new card components
 */
export default function CardTestPage() {
  const [showHighlights, setShowHighlights] = useState(true);
  
  // Sample data
  const sampleResponse = `Hello! I'd be happy to help you with that question. Here are a few key points to consider:\n\n1. First, make sure you understand the requirements clearly\n2. Break down the problem into smaller, manageable pieces\n3. Consider different approaches and their trade-offs\n\nThe most important thing is to take your time and think through each step carefully. This will help ensure you get the best possible result.\n\nIs there anything specific you'd like me to clarify or expand on?`;

  const sampleEvidence = ["help you with that question", "break down the problem", "take your time"];
  const highlightRanges = showHighlights ? evidenceToHighlightRanges(sampleResponse, sampleEvidence) : [];
  
  const sampleMetadata = {
    score: { helpfulness: 8.5, clarity: 9.2, accuracy: 8.8 },
    timestamp: "2024-01-15T10:30:00Z",
    tokens: 156
  };

  const sampleResponseB = `I can certainly assist with your inquiry. Let me provide a structured approach:\n\n• Start by identifying the core objective\n• Research available resources and constraints  \n• Develop a step-by-step action plan\n• Execute systematically while monitoring progress\n\nRemember that patience and methodical thinking are essential for success. Feel free to ask if you need further guidance on any aspect.`;

  // PropertyCard sample
  const sampleProperty = {
    id: 'prop_1',
    question_id: 'q1',
    model: 'gpt-4',
    property_description: 'The response gives clear, step-by-step guidance and emphasizes careful thought.',
    category: 'helpfulness',
    behavior_type: 'Positive',
    reason: 'The assistant provides structured steps and explains how they lead to better outcomes.',
    evidence: sampleEvidence,
    unexpected_behavior: false,
    contains_errors: false
  };

  const sampleConversation = {
    question_id: 'q1',
    prompt: 'How should I approach complex problems?',
    model: 'gpt-4',
    responses: sampleResponse,
    scores: sampleMetadata.score
  };

  const sampleCluster = {
    id: 'c1',
    label: 'Helpful, step-by-step guidance',
    size: 1
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h4" sx={{ mb: 3 }}>
        Card Components Test Page
      </Typography>
      
      <Button 
        variant="outlined" 
        onClick={() => setShowHighlights(!showHighlights)}
        sx={{ mb: 3 }}
      >
        {showHighlights ? 'Hide' : 'Show'} Evidence Highlighting
      </Button>

      <Stack spacing={4}>
        {/* ResponseContent Component */}
        <Box>
          <Typography variant="h5" sx={{ mb: 2 }}>
            1. ResponseContent Component
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Base component for text with highlighting
          </Typography>
          <Box sx={{ border: '1px solid #ddd', borderRadius: 1, p: 2, backgroundColor: '#f9f9f9' }}>
            <ResponseContent 
              content={sampleResponse}
              highlightedRanges={highlightRanges}
            />
          </Box>
        </Box>

        <Divider />

        {/* ModelResponseCard - Single */}
        <Box>
          <Typography variant="h5" sx={{ mb: 2 }}>
            2. ModelResponseCard Component - Single Model
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Complete model response with metadata and highlighting
          </Typography>
          
          <Stack spacing={2}>
            {/* Default variant */}
            <Box>
              <Typography variant="h6" sx={{ mb: 1 }}>Default Variant</Typography>
              <ModelResponseCard
                modelName="gpt-4"
                response={sampleResponse}
                highlightedRanges={highlightRanges}
                metadata={sampleMetadata}
                variant="default"
              />
            </Box>
            
            {/* Compact variant */}
            <Box>
              <Typography variant="h6" sx={{ mb: 1 }}>Compact Variant</Typography>
              <ModelResponseCard
                modelName="gpt-3.5-turbo"
                response={sampleResponse}
                highlightedRanges={highlightRanges}
                metadata={{ score: { overall: 7.8 } }}
                variant="compact"
              />
            </Box>
            
            {/* Expanded variant */}
            <Box>
              <Typography variant="h6" sx={{ mb: 1 }}>Expanded Variant</Typography>
              <ModelResponseCard
                modelName="claude-3-sonnet"
                response={sampleResponse}
                highlightedRanges={highlightRanges}
                metadata={sampleMetadata}
                variant="expanded"
              />
            </Box>
          </Stack>
        </Box>

        <Divider />

        {/* Side-by-Side Comparison */}
        <Box>
          <Typography variant="h5" sx={{ mb: 2 }}>
            3. SideBySideResponse Component
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Side-by-side model comparison
          </Typography>
          
          <SideBySideResponse
            modelA="gpt-4"
            modelB="claude-3-sonnet"
            responseA={sampleResponse}
            responseB={sampleResponseB}
            highlightedRangesA={highlightRanges}
            highlightedRangesB={showHighlights ? evidenceToHighlightRanges(sampleResponseB, ["structured approach", "step-by-step action plan"]) : []}
            metadataA={{ score: { helpfulness: 8.5, clarity: 9.2 } }}
            metadataB={{ score: { helpfulness: 8.8, clarity: 8.9 } }}
            variant="compact"
          />
        </Box>

        <Divider />

        {/* Property Card */}
        <Box>
          <Typography variant="h5" sx={{ mb: 2 }}>
            4. PropertyCard Component
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Property description styling + evidence highlighting over the conversation
          </Typography>
          <PropertyCard
            property={sampleProperty}
            conversation={sampleConversation}
            method="single_model"
            onOpenConversation={() => alert('Open in sidebar')}
          />
        </Box>

        <Divider />

        {/* Cluster Card (via PropertyCard) */}
        <Box>
          <Typography variant="h5" sx={{ mb: 2 }}>
            5. Cluster (PropertyCard with clusterLabel)
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Same card with a cluster label displayed above the description
          </Typography>
          <PropertyCard
            property={sampleProperty}
            conversation={sampleConversation}
            method="single_model"
            clusterLabel={sampleCluster.label}
          />
        </Box>
      </Stack>
    </Box>
  );
}
