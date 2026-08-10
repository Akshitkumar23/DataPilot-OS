'use strict';

const { app, BrowserWindow, ipcMain, shell, dialog, screen } = require('electron');
const path = require('path');
const http = require('http');
const crypto = require('crypto');
const fs = require('fs');
const { spawn, execSync } = require('child_process');

const isDev = process.env.NODE_ENV === 'development';
const BACKEND_PORT = 8000;
const DEV_UI_URL = `http://localhost:5173`;
const MAX_RESTARTS = 5;

function getOrCreateApiToken() {
  const tokenPath = path.join(__dirname, '..', '.env_token');
  if (process.env.DATAPILOT_API_TOKEN) {
    return process.env.DATAPILOT_API_TOKEN;
  }
  if (fs.existsSync(tokenPath)) {
    try {
      const existing = fs.readFileSync(tokenPath, 'utf8').trim();
      if (existing) return existing;
    } catch (e) {}
  }
  const token = crypto.randomBytes(32).toString('hex');
  try {
    fs.writeFileSync(tokenPath, token, 'utf8');
  } catch (e) {}
  return token;
}

const API_TOKEN = getOrCreateApiToken();
process.env.DATAPILOT_API_TOKEN = API_TOKEN;

let mainWindow = null;
let pythonProcess = null;
let restartCount = 0;
let isQuitting = false;

// ──────────────────────────────────────────────
// Python backend lifecycle
// ──────────────────────────────────────────────
function getPythonExecutable() {
  const isWin = process.platform === 'win32';
  const fs = require('fs');

  // Production: bundled server binary next to app resources
  const bundledBinaryInSubdir = path.join(process.resourcesPath, 'python', isWin ? 'server.exe' : 'server');
  const bundledBinaryDirect = path.join(process.resourcesPath, isWin ? 'server.exe' : 'server');

  if (fs.existsSync(bundledBinaryInSubdir)) return bundledBinaryInSubdir;
  if (fs.existsSync(bundledBinaryDirect)) return bundledBinaryDirect;

  // Resolve ASAR vs unpacked directory path
  const baseDir = __dirname.includes('app.asar')
    ? __dirname.replace('app.asar', 'app.asar.unpacked')
    : __dirname;

  const venvPython = isWin
    ? path.join(baseDir, '..', '.venv', 'Scripts', 'python.exe')
    : path.join(baseDir, '..', '.venv', 'bin', 'python');

  if (fs.existsSync(venvPython)) {
    try {
      execSync(`"${venvPython}" -c "import requests, fastapi, multipart"`, { stdio: 'ignore' });
      return venvPython;
    } catch (e) {
      console.warn('[Python] .venv incomplete, falling back to system python');
    }
  }

  // System fallback
  return isWin ? 'python' : 'python3';
}

function getPythonScriptPath() {
  const fs = require('fs');
  const baseDir = __dirname.includes('app.asar')
    ? __dirname.replace('app.asar', 'app.asar.unpacked')
    : __dirname;

  const devScript = path.join(baseDir, '..', 'server.py');
  if (fs.existsSync(devScript)) return devScript;

  const prodScript = path.join(process.resourcesPath, 'server.py');
  if (fs.existsSync(prodScript)) return prodScript;

  return devScript;
}

function killPort8000Sync() {
  if (process.platform === 'win32') {
    try {
      execSync('cmd /c "for /f \\"tokens=5\\" %a in (\'netstat -aon ^| findstr :8000\') do taskkill /f /pid %a"', { stdio: 'ignore' });
    } catch (e) {}
  } else {
    try {
      execSync('lsof -t -i:8000 | xargs kill -9 2>/dev/null || true', { stdio: 'ignore' });
    } catch (e) {}
  }
}

function checkPort8000Healthy() {
  return new Promise((resolve) => {
    const req = http.get(`http://localhost:${BACKEND_PORT}/health`, (res) => {
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(800, () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function startPythonBackend() {
  const isHealthy = await checkPort8000Healthy();
  if (isHealthy) {
    console.log('[Python] Backend engine is already running on port 8000. Connected cleanly.');
    return;
  }

  killPort8000Sync();
  const pythonExe = getPythonExecutable();
  const isStandaloneBinary = pythonExe.endsWith('server.exe') || pythonExe.endsWith('server');
  const scriptPath = getPythonScriptPath();
  const spawnArgs = isStandaloneBinary ? [] : [scriptPath];
  const spawnCwd = path.dirname(scriptPath);

  try {
    pythonProcess = spawn(pythonExe, spawnArgs, {
      cwd: spawnCwd,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, DATAPILOT_API_TOKEN: API_TOKEN },
    });

    setTimeout(() => {
      if (pythonProcess) {
        restartCount = 0;
      }
    }, 3000);

    pythonProcess.on('error', (err) => {
      console.error('[Python spawn error]:', err);
    });

    pythonProcess.stdout.on('data', (data) => {
      console.log('[Python]', data.toString().trim());
    });
    pythonProcess.stderr.on('data', (data) => {
      console.log('[Python]', data.toString().trim());
    });
    pythonProcess.on('exit', (code) => {
      console.log(`[Python] process exited with code ${code}.`);
      pythonProcess = null;

      if (isQuitting || code === 3) return;

      if (restartCount < MAX_RESTARTS) {
        restartCount++;
        console.log(`[Python] Auto-restarting Python backend (attempt ${restartCount}/${MAX_RESTARTS}) in 1s...`);
        setTimeout(() => {
          if (!pythonProcess && !isQuitting) {
            startPythonBackend();
          }
        }, 1000);
      } else {
        console.error(`[Python] Maximum auto-restart retries (${MAX_RESTARTS}) reached. Halting restart loop.`);
      }
    });
  } catch (err) {
    console.error('[Electron] Failed to start Python backend:', err);
  }
}

function stopPythonBackend() {
  if (pythonProcess) {
    if (process.platform === 'win32') {
      try {
        spawn('taskkill', ['/pid', pythonProcess.pid, '/f', '/t']);
      } catch (e) {}
    } else {
      pythonProcess.kill('SIGTERM');
    }
    pythonProcess = null;
  }
  killPort8000Sync();
}

function waitForBackend(url, maxAttempts = 150, intervalMs = 100) {
  return new Promise((resolve) => {
    let attempts = 0;
    const check = () => {
      attempts++;
      const req = http.get(url, (res) => {
        restartCount = 0;
        resolve(true);
      });
      req.on('error', () => {
        if (attempts >= maxAttempts) {
          console.warn(`[Electron] Backend health check timed out after ${maxAttempts} attempts.`);
          resolve(false);
        } else {
          setTimeout(check, intervalMs);
        }
      });
      req.end();
    };
    check();
  });
}

// ──────────────────────────────────────────────
// Window creation — Gemini Floating Side Panel
// ──────────────────────────────────────────────
function createWindow() {
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;
  const panelWidth = 440;
  const panelHeight = Math.min(920, screenHeight - 20);
  const xPos = Math.max(0, screenWidth - panelWidth - 10);
  const yPos = 10;

  mainWindow = new BrowserWindow({
    width: panelWidth,
    height: panelHeight,
    x: xPos,
    y: yPos,
    minWidth: 380,
    minHeight: 500,
    alwaysOnTop: true,
    frame: false,
    titleBarStyle: 'hidden',
    backgroundColor: '#0a0d14',
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
    },
  });

  const distHtml = path.join(__dirname, '..', 'dist', 'index.html');

  if (isDev) {
    mainWindow.loadURL(DEV_UI_URL);
  } else if (require('fs').existsSync(distHtml)) {
    mainWindow.loadFile(distHtml);
  } else {
    mainWindow.loadURL(`http://127.0.0.1:${BACKEND_PORT}`);
  }

  mainWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
    console.log(`[Browser Console L${level}]: ${message} (source: ${sourceId}:${line})`);
  });

  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.log(`[Browser Fail Load]: Code ${errorCode} - ${errorDescription}`);
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    mainWindow.focus();
  });

  mainWindow.on('maximize', () => {
    mainWindow.webContents.send('window-state-change', { maximized: true });
  });
  mainWindow.on('unmaximize', () => {
    mainWindow.webContents.send('window-state-change', { maximized: false });
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ──────────────────────────────────────────────
// App lifecycle
// ──────────────────────────────────────────────
app.whenReady().then(async () => {
  startPythonBackend();

  const backendUrl = `http://127.0.0.1:${BACKEND_PORT}`;
  await waitForBackend(backendUrl);

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  stopPythonBackend();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  isQuitting = true;
  stopPythonBackend();
});

// ──────────────────────────────────────────────
// IPC handlers — window controls & Pin state
// ──────────────────────────────────────────────
ipcMain.handle('get-api-token', () => API_TOKEN);
ipcMain.on('win:minimize', () => mainWindow?.minimize());
ipcMain.on('win:maximize', () => {
  if (mainWindow?.isMaximized()) mainWindow.unmaximize();
  else mainWindow?.maximize();
});
ipcMain.on('win:close', () => mainWindow?.close());
ipcMain.handle('win:isMaximized', () => mainWindow?.isMaximized() ?? false);

ipcMain.handle('win:toggleAlwaysOnTop', () => {
  if (!mainWindow) return false;
  const current = mainWindow.isAlwaysOnTop();
  mainWindow.setAlwaysOnTop(!current);
  return !current;
});
ipcMain.handle('win:isAlwaysOnTop', () => mainWindow?.isAlwaysOnTop() ?? true);

// File open dialog
ipcMain.handle('dialog:openFile', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Data Files', extensions: ['csv', 'xlsx', 'xls', 'parquet', 'json'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });
  return result.canceled ? null : result.filePaths[0];
});

// Open external URL in default browser
ipcMain.on('open-external', (_, rawUrl) => {
  try {
    const parsed = new URL(rawUrl);
    if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
      shell.openExternal(rawUrl);
    } else {
      console.warn(`[Electron] Blocked opening invalid/untrusted URL protocol: ${parsed.protocol}`);
    }
  } catch (err) {
    console.error(`[Electron] Invalid URL for open-external: ${rawUrl}`, err);
  }
});

