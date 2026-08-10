'use strict';

const { contextBridge, ipcRenderer } = require('electron');

// Expose a safe, minimal API to the renderer (React)
contextBridge.exposeInMainWorld('electron', {
  // Auth token for API calls
  getApiToken: () => ipcRenderer.invoke('get-api-token'),

  // Window controls
  minimize:          () => ipcRenderer.send('win:minimize'),
  maximize:          () => ipcRenderer.send('win:maximize'),
  close:             () => ipcRenderer.send('win:close'),
  isMaximized:       () => ipcRenderer.invoke('win:isMaximized'),
  toggleAlwaysOnTop: () => ipcRenderer.invoke('win:toggleAlwaysOnTop'),
  isAlwaysOnTop:     () => ipcRenderer.invoke('win:isAlwaysOnTop'),

  // Window state updates from main → renderer
  onWindowStateChange: (cb) => {
    const handler = (_, state) => cb(state);
    ipcRenderer.on('window-state-change', handler);
    return () => ipcRenderer.removeListener('window-state-change', handler);
  },

  // Native file open dialog
  openFileDialog: () => ipcRenderer.invoke('dialog:openFile'),

  // Open external links in default browser
  openExternal: (url) => ipcRenderer.send('open-external', url),
});
