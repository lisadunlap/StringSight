import React from "react";
import type { Message } from "../lib/traces";
import { Box, Typography } from "@mui/material";
import ConversationTrace from "./ConversationTrace";

export function SideBySideTrace({
  messagesA,
  messagesB,
  modelA,
  modelB,
  highlights,
  targetModel,
  rawResponseA,
  rawResponseB,
}: {
  messagesA: Message[];
  messagesB: Message[];
  modelA: string;
  modelB: string;
  highlights?: string[];
  targetModel?: string;
  rawResponseA?: any;
  rawResponseB?: any;
}) {
  return (
    <Box sx={{
      display: "grid",
      gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
      gap: 2,
    }}>
      <Box>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>{modelA}</Typography>
        <ConversationTrace messages={messagesA} highlights={targetModel && targetModel !== modelA ? [] : highlights} rawResponse={rawResponseA} />
      </Box>
      <Box>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>{modelB}</Typography>
        <ConversationTrace messages={messagesB} highlights={targetModel && targetModel !== modelB ? [] : highlights} rawResponse={rawResponseB} />
      </Box>
    </Box>
  );
}

export default SideBySideTrace;


