import React from "react";
import { Dialog, DialogTitle, DialogContent, DialogActions, Button, List, ListItemButton, ListItemText, Stack, TextField, Typography, Divider } from "@mui/material";
import { listPath } from "../lib/api";

export type RemoteBrowserMode = "directory" | "file";

export interface RemoteBrowserDialogProps {
  open: boolean;
  mode: RemoteBrowserMode;
  initialPath?: string; // "." to start at server base
  exts?: string[]; // only used in file mode for filtering files
  title?: string;
  onClose: () => void;
  onSelect: (absolutePath: string) => void;
}

type Entry = { name: string; path: string; type: "file" | "dir" };

function parentOf(p: string): string {
  if (!p || p === ".") return ".";
  const norm = p.replace(/\\/g, "/");
  const idx = norm.lastIndexOf("/");
  if (idx <= 0) return ".";
  return norm.slice(0, idx);
}

export default function RemoteBrowserDialog(props: RemoteBrowserDialogProps) {
  const { open, mode, initialPath = ".", exts, title, onClose, onSelect } = props;
  const [currentPath, setCurrentPath] = React.useState<string>(initialPath);
  const [entries, setEntries] = React.useState<Entry[]>([]);
  const [error, setError] = React.useState<string | null>(null);

  const refresh = React.useCallback(async (p: string) => {
    try {
      setError(null);
      const resp = await listPath(p, exts);
      const items = (resp?.entries || []) as Entry[];
      // Always show directories first
      items.sort((a, b) => (a.type === b.type ? a.name.localeCompare(b.name) : a.type === "dir" ? -1 : 1));
      setEntries(items);
    } catch (e: any) {
      setError(String(e?.message || e));
      setEntries([]);
    }
  }, [exts]);

  React.useEffect(() => {
    if (open) {
      setCurrentPath(initialPath);
      void refresh(initialPath);
    }
  }, [open, initialPath, refresh]);

  const goUp = async () => {
    const parent = parentOf(currentPath);
    setCurrentPath(parent);
    await refresh(parent);
  };

  const enter = async (entry: Entry) => {
    if (entry.type !== "dir") return;
    setCurrentPath(entry.path);
    await refresh(entry.path);
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="md">
      <DialogTitle>{title || (mode === "directory" ? "Browse Server (Directories)" : "Browse Server (Files)")}</DialogTitle>
      <DialogContent dividers>
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
          <Button variant="outlined" size="small" onClick={goUp} disabled={currentPath === "."}>Up</Button>
          <TextField size="small" fullWidth value={currentPath} onChange={(e) => setCurrentPath(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') { void refresh(currentPath); } }} />
          <Button variant="outlined" size="small" onClick={() => refresh(currentPath)}>Go</Button>
        </Stack>
        {error && (
          <Typography color="error" variant="body2" sx={{ mb: 1 }}>{error}</Typography>
        )}
        <Divider sx={{ mb: 1 }} />
        <List dense>
          {entries.map((e) => (
            <ListItemButton key={e.path} onClick={() => (e.type === "dir" ? enter(e) : onSelect(e.path))}>
              <ListItemText primary={`${e.type === "dir" ? "📁" : "📄"} ${e.name}`} secondary={e.path} />
            </ListItemButton>
          ))}
          {entries.length === 0 && (
            <Typography variant="body2" color="text.secondary">No entries</Typography>
          )}
        </List>
      </DialogContent>
      <DialogActions>
        {mode === "directory" && (
          <Button onClick={() => onSelect(currentPath)} variant="contained">Select This Folder</Button>
        )}
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}





