'use strict';

const { contextBridge, ipcRenderer } = require('electron');

// Get synchronous API token on preload initialization
let initialApiToken = '';
let initialApiBaseUrl = '';
try {
  initialApiToken = ipcRenderer.sendSync('get-api-token-sync') || '';
  initialApiBaseUrl = ipcRenderer.sendSync('get-api-base-url-sync') || '';
} catch (err) {
  console.warn('[Preload] Could not retrieve sync API token:', err);
}

// Expose safe, minimal API to renderer (React)
const electronAPI = {
  // Synchronous API token for immediate access
  apiToken: initialApiToken,
  apiBaseUrl: initialApiBaseUrl,

  // Asynchronous auth token getter
  getApiToken: () => ipcRenderer.invoke('get-api-token'),
  getBackendStatus: () => ipcRenderer.invoke('get-backend-status'),
  onBackendStatusChange: (cb) => {
    if (typeof cb !== 'function') return () => {};
    const handler = (_, state) => cb(state);
    ipcRenderer.on('backend-state-change', handler);
    return () => ipcRenderer.removeListener('backend-state-change', handler);
  },

  // Window controls
  minimize:          () => ipcRenderer.send('win:minimize'),
  maximize:          () => ipcRenderer.send('win:maximize'),
  close:             () => ipcRenderer.send('win:close'),
  isMaximized:       () => ipcRenderer.invoke('win:isMaximized'),
  toggleAlwaysOnTop: () => ipcRenderer.invoke('win:toggleAlwaysOnTop'),
  isAlwaysOnTop:     () => ipcRenderer.invoke('win:isAlwaysOnTop'),

  // Window state updates from main → renderer
  onWindowStateChange: (cb) => {
    if (typeof cb !== 'function') return () => {};
    const handler = (_, state) => cb(state);
    ipcRenderer.on('window-state-change', handler);
    return () => ipcRenderer.removeListener('window-state-change', handler);
  },

  // Native file open dialog
  openFileDialog: async (options) => {
    const res = await ipcRenderer.invoke('dialog:openFile', options);
    if (!res) return null;
    return Array.isArray(res) ? res : [res];
  },
  openFile: async (options) => {
    const res = await ipcRenderer.invoke('dialog:openFile', options);
    if (!res) return null;
    return Array.isArray(res) ? res : [res];
  },

  // Native directory open dialog for workspace agent
  openDirectoryDialog: (options) => ipcRenderer.invoke('dialog:openDirectory', options),

  // Native desktop screen capture
  captureScreen: (options) => ipcRenderer.invoke('screen:capture', options),

  // Open external links safely in default browser
  openExternal: (url) => ipcRenderer.send('open-external', url),

  // Observability & UI Action Telemetry
  logUIAction: (action, details) => ipcRenderer.send('log:ui-action', action, details),
};

// Expose on both window.electron and window.electronAPI for maximum compatibility
contextBridge.exposeInMainWorld('electron', electronAPI);
contextBridge.exposeInMainWorld('electronAPI', electronAPI);
