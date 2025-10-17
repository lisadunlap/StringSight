import React, { useState, useEffect, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Breadcrumbs,
  Link,
  Typography,
  Box,
  CircularProgress
} from '@mui/material';
import FolderIcon from '@mui/icons-material/Folder';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import { listPath } from '../lib/api';

interface ServerFolderBrowserProps {
  open: boolean;
  onClose: () => void;
  onSelectPath: (path: string) => void;
  title?: string;
  acceptedExtensions?: string[];
}

interface PathEntry {
  name: string;
  type: 'file' | 'dir';
  path: string;
  size?: number;
  modified?: string;
}

export default function ServerFolderBrowser({
  open,
  onClose,
  onSelectPath,
  title = "Select Folder",
  acceptedExtensions = ['.json', '.jsonl']
}: ServerFolderBrowserProps) {
  const [currentPath, setCurrentPath] = useState<string>('.');
  const [pathStack, setPathStack] = useState<string[]>(['']);
  const [entries, setEntries] = useState<PathEntry[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Load directory contents
  const loadDirectory = useCallback(async (path: string) => {
    console.log('[ServerFolderBrowser] Loading directory:', path);
    setLoading(true);
    setError(null);
    try {
      const response = await listPath(path, acceptedExtensions);
      console.log('[ServerFolderBrowser] Loaded entries:', response.entries);
      
      // Sort entries: directories first, then files, each sorted by date (newest first)
      const sorted = (response.entries || []).slice().sort((a: PathEntry, b: PathEntry) => {
        // Directories before files
        if (a.type === 'dir' && b.type === 'file') return -1;
        if (a.type === 'file' && b.type === 'dir') return 1;
        
        // Within same type, sort by modified date (newest first)
        if (a.modified && b.modified) {
          return new Date(b.modified).getTime() - new Date(a.modified).getTime();
        }
        // Handle missing dates (put at end)
        if (a.modified && !b.modified) return -1;
        if (!a.modified && b.modified) return 1;
        
        // If both missing dates, sort alphabetically
        return a.name.localeCompare(b.name);
      });
      
      setEntries(sorted);
    } catch (err) {
      console.error('[ServerFolderBrowser] Error loading directory:', err);
      setError(err instanceof Error ? err.message : 'Failed to load directory');
      setEntries([]);
    } finally {
      setLoading(false);
    }
  }, [acceptedExtensions]);

  // Navigate to a subdirectory
  const navigateToPath = (newPath: string) => {
    console.log('[ServerFolderBrowser] Navigating to path:', newPath);
    setCurrentPath(newPath);
    const newStack = newPath === '.' ? [''] : newPath.split('/').filter(Boolean);
    setPathStack(['', ...newStack]);
  };

  // Navigate up one level
  const navigateUp = () => {
    if (pathStack.length > 1) {
      const newStack = pathStack.slice(0, -1);
      setPathStack(newStack);
      const newPath = newStack.length === 1 ? '.' : newStack.slice(1).join('/');
      setCurrentPath(newPath);
    }
  };

  // Navigate to specific breadcrumb
  const navigateToBreadcrumb = (index: number) => {
    const newStack = pathStack.slice(0, index + 1);
    setPathStack(newStack);
    const newPath = newStack.length === 1 ? '.' : newStack.slice(1).join('/');
    setCurrentPath(newPath);
  };

  // Handle entry click
  const handleEntryClick = (entry: PathEntry) => {
    console.log('[ServerFolderBrowser] Entry clicked:', entry);
    if (entry.type === 'dir') {
      const newPath = currentPath === '.' ? entry.name : `${currentPath}/${entry.name}`;
      console.log('[ServerFolderBrowser] New path will be:', newPath);
      navigateToPath(newPath);
    }
  };

  // Check if current path contains required result files
  const hasResultFiles = () => {
    // Prefer fast JSONL artifacts first, but keep support for legacy JSON
    const files = new Set(entries.filter(e => e.type === 'file').map(e => e.name));
    const jsonlSignals = [
      'clustered_results_lightweight.jsonl',
      'parsed_properties.jsonl',
      'model_cluster_scores_df.jsonl',
      'cluster_scores_df.jsonl',
      'model_scores_df.jsonl',
    ];
    const legacyJson = [
      'full_dataset.json',
      'model_cluster_scores.json',
      'cluster_scores.json',
      'model_scores.json',
    ];
    return jsonlSignals.some(name => files.has(name)) || legacyJson.some(name => files.has(name));
  };

  // Reset to root when dialog opens
  useEffect(() => {
    if (open) {
      setCurrentPath('.');
      setPathStack(['']);
      setError(null);
    }
  }, [open]);

  // Load directory when path changes
  useEffect(() => {
    if (open) {
      loadDirectory(currentPath);
    }
  }, [currentPath, open, loadDirectory]);

  const handleSelectCurrent = () => {
    onSelectPath(currentPath);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        {title}
        {currentPath !== '.' && (
          <Typography variant="caption" display="block" color="text.secondary">
            Current path: {currentPath}
          </Typography>
        )}
      </DialogTitle>
      <DialogContent>
        <Box sx={{ mb: 2 }}>
          <Breadcrumbs>
            {pathStack.map((segment, index) => (
              <Link
                key={index}
                component="button"
                variant="body2"
                onClick={() => navigateToBreadcrumb(index)}
                sx={{ textDecoration: 'none' }}
              >
                {index === 0 ? 'Root' : segment}
              </Link>
            ))}
          </Breadcrumbs>
        </Box>

        {error && (
          <Typography color="error" sx={{ mb: 2 }}>
            {error}
          </Typography>
        )}

        {loading ? (
          <Box display="flex" justifyContent="center" p={2}>
            <CircularProgress />
          </Box>
        ) : (
          <List sx={{ maxHeight: 400, overflow: 'auto' }}>
            {pathStack.length > 1 && (
              <ListItem disablePadding>
                <ListItemButton onClick={navigateUp}>
                  <ListItemIcon>
                    <FolderIcon />
                  </ListItemIcon>
                  <ListItemText primary=".." />
                </ListItemButton>
              </ListItem>
            )}
            
            {entries.map((entry, index) => (
              <ListItem key={index} disablePadding>
                <ListItemButton onClick={() => handleEntryClick(entry)}>
                  <ListItemIcon>
                    {entry.type === 'dir' ? <FolderOpenIcon /> : <InsertDriveFileIcon />}
                  </ListItemIcon>
                  <ListItemText 
                    primary={entry.name}
                    secondary={entry.type === 'file' && entry.size ? `${entry.size} bytes` : undefined}
                  />
                </ListItemButton>
              </ListItem>
            ))}
            
            {entries.length === 0 && !loading && (
              <ListItem>
                <ListItemText primary="No items found" />
              </ListItem>
            )}
          </List>
        )}

        {hasResultFiles() && (
          <Box sx={{ mt: 2, p: 2, bgcolor: 'success.light', borderRadius: 1 }}>
            <Typography variant="body2" color="success.contrastText">
              ✓ This folder contains result files and can be selected
            </Typography>
          </Box>
        )}
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button 
          onClick={handleSelectCurrent} 
          variant="contained"
          disabled={!hasResultFiles()}
        >
          Select This Folder
        </Button>
      </DialogActions>
    </Dialog>
  );
}

