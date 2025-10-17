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
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { listPath } from '../lib/api';

interface ServerFileBrowserProps {
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

export default function ServerFileBrowser({
  open,
  onClose,
  onSelectPath,
  title = "Select File",
  acceptedExtensions = ['.json', '.jsonl', '.csv']
}: ServerFileBrowserProps) {
  const [currentPath, setCurrentPath] = useState<string>('.');
  const [pathStack, setPathStack] = useState<string[]>(['']);
  const [entries, setEntries] = useState<PathEntry[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  // Load directory contents
  const loadDirectory = useCallback(async (path: string) => {
    console.log('[ServerFileBrowser] Loading directory:', path);
    setLoading(true);
    setError(null);
    try {
      const response = await listPath(path, acceptedExtensions);
      console.log('[ServerFileBrowser] Loaded entries:', response.entries);
      
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
      console.error('[ServerFileBrowser] Error loading directory:', err);
      setError(err instanceof Error ? err.message : 'Failed to load directory');
      setEntries([]);
    } finally {
      setLoading(false);
    }
  }, [acceptedExtensions]);

  // Navigate to a subdirectory
  const navigateToPath = (newPath: string) => {
    console.log('[ServerFileBrowser] Navigating to path:', newPath);
    setCurrentPath(newPath);
    const newStack = newPath === '.' ? [''] : newPath.split('/').filter(Boolean);
    setPathStack(['', ...newStack]);
    setSelectedFile(null);
  };

  // Navigate up one level
  const navigateUp = () => {
    if (pathStack.length > 1) {
      const newStack = pathStack.slice(0, -1);
      setPathStack(newStack);
      const newPath = newStack.length === 1 ? '.' : newStack.slice(1).join('/');
      setCurrentPath(newPath);
      setSelectedFile(null);
    }
  };

  // Navigate to specific breadcrumb
  const navigateToBreadcrumb = (index: number) => {
    const newStack = pathStack.slice(0, index + 1);
    setPathStack(newStack);
    const newPath = newStack.length === 1 ? '.' : newStack.slice(1).join('/');
    setCurrentPath(newPath);
    setSelectedFile(null);
  };

  // Handle entry click
  const handleEntryClick = (entry: PathEntry) => {
    console.log('[ServerFileBrowser] Entry clicked:', entry);
    if (entry.type === 'dir') {
      const newPath = currentPath === '.' ? entry.name : `${currentPath}/${entry.name}`;
      console.log('[ServerFileBrowser] New path will be:', newPath);
      navigateToPath(newPath);
    } else if (entry.type === 'file') {
      // Select the file
      setSelectedFile(entry.path);
    }
  };

  // Reset to root when dialog opens
  useEffect(() => {
    if (open) {
      setCurrentPath('.');
      setPathStack(['']);
      setError(null);
      setSelectedFile(null);
    }
  }, [open]);

  // Load directory when path changes
  useEffect(() => {
    if (open) {
      loadDirectory(currentPath);
    }
  }, [currentPath, open, loadDirectory]);

  const handleSelectFile = () => {
    if (selectedFile) {
      onSelectPath(selectedFile);
      onClose();
    }
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
              <ListItem 
                key={index} 
                disablePadding
                sx={{
                  bgcolor: selectedFile === entry.path ? 'action.selected' : 'transparent'
                }}
              >
                <ListItemButton onClick={() => handleEntryClick(entry)}>
                  <ListItemIcon>
                    {entry.type === 'dir' ? (
                      <FolderOpenIcon />
                    ) : selectedFile === entry.path ? (
                      <CheckCircleIcon color="primary" />
                    ) : (
                      <InsertDriveFileIcon />
                    )}
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

        {selectedFile && (
          <Box sx={{ mt: 2, p: 2, bgcolor: 'success.light', borderRadius: 1 }}>
            <Typography variant="body2" color="success.contrastText">
              ✓ Selected: {selectedFile.split('/').pop()}
            </Typography>
          </Box>
        )}
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button 
          onClick={handleSelectFile} 
          variant="contained"
          disabled={!selectedFile}
        >
          Load File
        </Button>
      </DialogActions>
    </Dialog>
  );
}

