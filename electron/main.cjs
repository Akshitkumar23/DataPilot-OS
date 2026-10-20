'use strict';

const { app, BrowserWindow, ipcMain, shell, dialog, screen, desktopCapturer, Menu } = require('electron');
const path = require('path');
const http = require('http');
const crypto = require('crypto');
const fs = require('fs');
const { spawn, execSync } = require('child_process');

// `electron .` does not automatically set NODE_ENV. app.isPackaged is the
// authoritative distinction, otherwise the local developer launcher was
// incorrectly treated like the detached production service.
const isDev = !app.isPackaged || process.env.NODE_ENV === 'development';
const configuredPort = Number.parseInt(process.env.DATAPILOT_BACKEND_PORT || '8011', 10);
const BACKEND_PORT = Number.isInteger(configuredPort) && configuredPort > 0 && configuredPort < 65536
  ? configuredPort
  : 8011;
const BACKEND_BASE_URL = `http://127.0.0.1:${BACKEND_PORT}`;
const DEV_UI_URL = 'http://localhost:5173';
const MAX_RESTARTS = 5;
// The desktop shell owns exactly one backend child for its own lifetime.
// Keeping a detached process alive seemed faster, but it made a later window
// attach to an unknown/stale server on the shared port.  A short predictable
// boot is safer than incorrect state, duplicate startup, or a false "ready"
// indicator.  Do not make this configurable from user environment variables.
const KEEP_BACKEND_WARM = false;

// Enforce single application instance lock
const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  console.log('[Electron] Another instance of DataPilot OS is already running. Exiting.');
  app.quit();
}

function getOrCreateApiToken() {
  // app.asar is read-only. Persist the per-user token outside the package so
  // a warm backend and a subsequently opened Electron window share one token.
  let tokenDir;
  try {
    tokenDir = app.getPath('userData');
    fs.mkdirSync(tokenDir, { recursive: true });
  } catch (e) {
    tokenDir = process.env.APPDATA || process.cwd();
  }
  const tokenPath = path.join(tokenDir, '.env_token');
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

function getBackendLogPath() {
  let logsDir;
  try {
    logsDir = path.join(app.getPath('userData'), 'logs');
  } catch (e) {
    logsDir = path.join(process.env.APPDATA || process.cwd(), 'DataPilot OS', 'logs');
  }
  fs.mkdirSync(logsDir, { recursive: true });
  return path.join(logsDir, 'backend.log');
}

const BACKEND_LOG_PATH = getBackendLogPath();
const DESKTOP_LOG_PATH = path.join(path.dirname(BACKEND_LOG_PATH), 'desktop.log');
const SESSION_LOG_PATH = path.join(path.dirname(BACKEND_LOG_PATH), 'session.log');
process.env.DATAPILOT_SESSION_LOG = SESSION_LOG_PATH;

function writeSessionLog(line) {
  try { fs.appendFileSync(SESSION_LOG_PATH, line.trimEnd() + '\n', 'utf8'); } catch (e) {}
}

function writeDesktopLog(level, message) {
  const ts = new Date().toISOString().replace('T', ' ').slice(0, 19);
  const line = `${new Date().toISOString()} - Electron - ${level} - ${message}\n`;
  try { fs.appendFileSync(DESKTOP_LOG_PATH, line, 'utf8'); } catch (e) {}
  writeSessionLog(`[${ts}] [SYSTEM] Electron ${level}: ${message}`);
  if (level === 'ERROR' || level === 'WARN') console.warn(`[Electron] ${message}`);
  else console.log(`[Electron] ${message}`);
}

// Disable GPU disk cache & http cache locking errors on Windows
app.commandLine.appendSwitch('disable-gpu-shader-disk-cache');
app.commandLine.appendSwitch('disable-http-cache');

let mainWindow = null;
let pythonProcess = null;
let restartCount = 0;
let isQuitting = false;
let backendStopRequested = false;
let backendStartupPromise = null;
let restartScheduled = false;
let backendState = 'starting';

function publishBackendState(state) {
  backendState = state;
  writeDesktopLog('INFO', `backend_state=${state}`);
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend-state-change', state);
  }
}

function resetManagedBackendLog() {
  try {
    // The launcher tails this exact file.  Reusing a historical log caused
    // previous failed plans to look like failures from the active query.
    fs.writeFileSync(BACKEND_LOG_PATH, '', 'utf8');
  } catch (e) {
    writeDesktopLog('WARN', `could not reset backend session log: ${e.message}`);
  }
}

// ──────────────────────────────────────────────
// Python backend lifecycle
// ──────────────────────────────────────────────
function getPythonExecutable() {
  const isWin = process.platform === 'win32';

  // Production: bundled server binary or project virtual environment.
  const bundledBinaryInSubdir = path.join(process.resourcesPath, 'python', isWin ? 'server.exe' : 'server');
  const bundledBinaryDirect = path.join(process.resourcesPath, isWin ? 'server.exe' : 'server');
  const bundledVenvPython = isWin
    ? path.join(process.resourcesPath, 'python', 'Scripts', 'python.exe')
    : path.join(process.resourcesPath, 'python', 'bin', 'python');

  if (fs.existsSync(bundledBinaryInSubdir)) return bundledBinaryInSubdir;
  if (fs.existsSync(bundledBinaryDirect)) return bundledBinaryDirect;
  if (fs.existsSync(bundledVenvPython)) return bundledVenvPython;

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

function getBackendReleaseId() {
  // A packaged backend is an immutable PyInstaller binary. Its content hash
  // is an opaque release identity, preventing a new desktop shell from
  // silently attaching to a warm backend left behind by an older package.
  // Development mode deliberately has one stable identity because its Python
  // process is sourced from this checkout rather than a bundled binary.
  if (isDev) return 'development';
  const pythonExe = getPythonExecutable();
  try {
    if (fs.existsSync(pythonExe)) {
      return crypto.createHash('sha256').update(fs.readFileSync(pythonExe)).digest('hex').slice(0, 20);
    }
  } catch (e) {
    writeDesktopLog('WARN', `could not fingerprint bundled backend: ${e.message}`);
  }
  return null;
}

const BACKEND_RELEASE_ID = getBackendReleaseId();

function readBackendLiveness() {
  return new Promise((resolve) => {
    const req = http.get(`${BACKEND_BASE_URL}/livez`, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => {
        if (res.statusCode < 200 || res.statusCode >= 400) return resolve(null);
        try {
          const status = JSON.parse(body);
          resolve(status);
        } catch (e) {
          resolve(null);
        }
      });
    });
    req.on('error', () => resolve(null));
    req.setTimeout(500, () => {
      req.destroy();
      resolve(null);
    });
  });
}

function isCompatibleBackend(status) {
  if (status?.backend !== 'DataPilot OS' || status?.status !== 'live') return false;
  // A backend with no release_id is a legacy package and is intentionally not
  // reusable. Never accept it just because it happens to own the public port.
  return Boolean(BACKEND_RELEASE_ID) && status.release_id === BACKEND_RELEASE_ID;
}

async function checkBackendHealthy() {
  return isCompatibleBackend(await readBackendLiveness());
}

function getListeningPidForBackendPort() {
  if (process.platform !== 'win32') return null;
  try {
    const lines = execSync('netstat -aon -p tcp', {
      encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'],
    }).split(/\r?\n/);
    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts.length < 5 || parts[3] !== 'LISTENING') continue;
      const localAddress = parts[1];
      if (!localAddress.endsWith(`:${BACKEND_PORT}`)) continue;
      const pid = Number.parseInt(parts[4], 10);
      return Number.isInteger(pid) && pid > 0 ? pid : null;
    }
  } catch (e) {}
  return null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForBackendPortRelease(previousPid, maxAttempts = 40, intervalMs = 125) {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const currentPid = getListeningPidForBackendPort();
    if (!currentPid || currentPid !== previousPid) return !currentPid;
    await sleep(intervalMs);
  }
  return false;
}

async function replaceVerifiedIncompatibleBackend(livenessStatus) {
  // /livez is intentionally lightweight and unauthenticated. It is the only
  // identity probe used during an upgrade: /health may wait on Ollama and was
  // the source of an upgrade race in earlier releases. We terminate a process
  // only after this liveness payload proves it is DataPilot OS.
  if (livenessStatus?.backend !== 'DataPilot OS' || livenessStatus?.status !== 'live') return false;
  const ownerPid = getListeningPidForBackendPort();
  if (!ownerPid || ownerPid === process.pid) return false;
  try {
    execSync(`taskkill /pid ${ownerPid} /f /t`, { stdio: 'ignore' });
    const released = await waitForBackendPortRelease(ownerPid);
    if (!released) {
      writeDesktopLog('WARN', `verified incompatible backend pid=${ownerPid} did not release port ${BACKEND_PORT}`);
      return false;
    }
    writeDesktopLog('INFO', `replaced verified incompatible DataPilot backend pid=${ownerPid} during upgrade`);
    return true;
  } catch (e) {
    writeDesktopLog('WARN', `could not replace verified incompatible backend on port ${BACKEND_PORT}: ${e.message}`);
    return false;
  }
}

function ensureOllamaRunning() {
  if (process.platform === 'win32') {
    try {
      const tasklist = execSync('tasklist /FI "IMAGENAME eq ollama.exe"', { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
      if (!tasklist || !tasklist.includes('ollama.exe')) {
        console.log('[Electron] Ollama daemon is not running. Launching ollama serve in background...');
        spawn('ollama', ['serve'], { detached: true, stdio: 'ignore' }).unref();
      }
    } catch (e) {
      try {
        spawn('ollama', ['serve'], { detached: true, stdio: 'ignore' }).unref();
      } catch (err) {}
    }
  }
}

async function startPythonBackend() {
  // All callers (initial boot and crash recovery) share one atomic startup
  // attempt.  Without this guard, two async callers could both observe a free
  // port and race to spawn bundled server.exe.
  if (backendStartupPromise) return backendStartupPromise;
  backendStartupPromise = startPythonBackendOnce();
  try {
    return await backendStartupPromise;
  } finally {
    backendStartupPromise = null;
  }
}

async function startPythonBackendOnce() {
  ensureOllamaRunning();
  const liveness = await readBackendLiveness();
  const isManagedChild = isCompatibleBackend(liveness) && pythonProcess && !pythonProcess.killed;
  if (isManagedChild) {
    writeDesktopLog('INFO', `managed compatible backend already live on port ${BACKEND_PORT}`);
    publishBackendState('ready');
    return;
  }

  if (liveness?.backend === 'DataPilot OS') {
    writeDesktopLog(
      'INFO',
      `recycling unmanaged DataPilot backend on port ${BACKEND_PORT} for release ${BACKEND_RELEASE_ID || 'unknown'}`,
    );
  }

  if (liveness?.backend === 'DataPilot OS') {
    const replaced = await replaceVerifiedIncompatibleBackend(liveness);
    if (!replaced) {
      writeDesktopLog('ERROR', `refusing to start over an incompatible DataPilot backend on port ${BACKEND_PORT}`);
      publishBackendState('failed');
      return;
    }
  } else if (getListeningPidForBackendPort()) {
    // A non-DataPilot process may legitimately use the configured port. Do
    // not kill or attach to it; surface a deterministic lifecycle failure.
    writeDesktopLog('ERROR', `port ${BACKEND_PORT} is occupied by an unverified process; backend was not started`);
    publishBackendState('failed');
    return;
  }

  backendStopRequested = false;
  resetManagedBackendLog();
  publishBackendState('starting');

  const pythonExe = getPythonExecutable();
  const isStandaloneBinary = pythonExe.endsWith('server.exe') || pythonExe.endsWith('server');
  const spawnArgs = isStandaloneBinary ? [] : ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', String(BACKEND_PORT), '--no-access-log'];
  // In a packaged Electron build ``__dirname`` resides inside app.asar.
  // app.asar is a file, not a real Windows working directory, so using it as
  // cwd makes spawn fail before the bundled backend can bind its API port.
  // The PyInstaller executable is self-contained; its own on-disk directory
  // is the correct production cwd. Development Python still uses the project
  // directory as before.
  const spawnCwd = isStandaloneBinary ? path.dirname(pythonExe) : path.join(__dirname, '..');

  try {
    const spawnedProcess = spawn(pythonExe, spawnArgs, {
      cwd: spawnCwd,
      detached: false,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        DATAPILOT_API_TOKEN: API_TOKEN,
        DATAPILOT_RELEASE_ID: BACKEND_RELEASE_ID || 'unfingerprinted',
        DATAPILOT_DATA_DIR: path.join(app.getPath('userData'), 'backend-data'),
        DATAPILOT_LOG_FILE: BACKEND_LOG_PATH,
        DATAPILOT_SESSION_LOG: SESSION_LOG_PATH,
      },
    });
    pythonProcess = spawnedProcess;

    setTimeout(() => {
      if (pythonProcess === spawnedProcess) {
        restartCount = 0;
      }
    }, 3000);

    spawnedProcess.on('error', (err) => {
      writeDesktopLog('ERROR', `backend spawn error: ${err.message}`);
    });

    spawnedProcess.stdout.on('data', (data) => {
      console.log('[Python]', data.toString().trim());
    });
    spawnedProcess.stderr.on('data', (data) => {
      console.log('[Python]', data.toString().trim());
    });
    spawnedProcess.on('exit', (code) => {
      writeDesktopLog('WARN', `backend process exited with code ${code}`);
      // A previous child may finish after a replacement started.  It must
      // never mark the replacement's healthy UI as failed.
      if (pythonProcess !== spawnedProcess) return;
      pythonProcess = null;

      if (isQuitting || backendStopRequested) return;
      if (code === 3) {
        // Exit code 3 is a bind conflict from Uvicorn/PyInstaller. A ready
        // state is never published from an arbitrary HTTP response; retries
        // would only obscure the owning process, so fail explicitly.
        writeDesktopLog('ERROR', `bundled backend could not bind port ${BACKEND_PORT}`);
        publishBackendState('failed');
        return;
      }

      if (!restartScheduled && restartCount < MAX_RESTARTS) {
        restartScheduled = true;
        restartCount++;
        writeDesktopLog('WARN', `backend auto-restart ${restartCount}/${MAX_RESTARTS} scheduled`);
        setTimeout(() => {
          restartScheduled = false;
          if (!pythonProcess && !isQuitting && !backendStopRequested) {
            startPythonBackend();
          }
        }, 1000);
      } else {
        writeDesktopLog('ERROR', `backend auto-restart limit ${MAX_RESTARTS} reached`);
        publishBackendState('failed');
      }
    });

    // PyInstaller self-extraction and ML package imports on Windows can
    // take 20-35s on cold boot. Allow up to 240 attempts (60s), but fast-fail
    // immediately if the spawned process crashes or exits.
    const ready = await waitForCompatibleBackend(240, 250);
    if (pythonProcess === spawnedProcess) {
      publishBackendState(ready ? 'ready' : 'failed');
    }
  } catch (err) {
    writeDesktopLog('ERROR', `failed to start backend: ${err.message}`);
    publishBackendState('failed');
  }
}

function stopPythonBackend() {
  backendStopRequested = true;
  restartScheduled = false;
  if (pythonProcess && pythonProcess.pid) {
    if (process.platform === 'win32') {
      try {
        execSync(`taskkill /pid ${pythonProcess.pid} /f /t`, { stdio: 'ignore' });
      } catch (e) {}
    } else {
      try {
        pythonProcess.kill('SIGTERM');
      } catch (e) {}
    }
    pythonProcess = null;
  }
}

async function waitForCompatibleBackend(maxAttempts = 240, intervalMs = 250) {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    if (isCompatibleBackend(await readBackendLiveness())) {
      restartCount = 0;
      return true;
    }
    // If the child process died while waiting, fail fast without stalling
    if (pythonProcess === null) {
      writeDesktopLog('WARN', `compatible backend probe aborted: backend process exited early at attempt ${attempt}`);
      return false;
    }
    await sleep(intervalMs);
  }
  writeDesktopLog('WARN', `compatible backend startup probe timed out after ${maxAttempts} attempts (${(maxAttempts * intervalMs) / 1000}s)`);
  return false;
}

// ──────────────────────────────────────────────
// Window creation — Gemini Floating Side Panel
// ──────────────────────────────────────────────
function createWindow() {
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;
  const winWidth = Math.min(1440, Math.max(1024, Math.round(screenWidth * 0.88)));
  const winHeight = Math.min(900, Math.max(680, Math.round(screenHeight * 0.88)));
  const xPos = Math.max(0, Math.round((screenWidth - winWidth) / 2));
  const yPos = Math.max(0, Math.round((screenHeight - winHeight) / 2));

  mainWindow = new BrowserWindow({
    width: winWidth,
    height: winHeight,
    x: xPos,
    y: yPos,
    minWidth: 800,
    minHeight: 550,
    alwaysOnTop: false,
    frame: false,
    titleBarStyle: 'hidden',
    backgroundColor: '#0a0d14',
    show: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
    },
  });

  // Remove default application menu
  mainWindow.setMenu(null);

  const distHtml = path.join(__dirname, '..', 'dist', 'index.html');

  if (isDev) {
    mainWindow.loadURL(DEV_UI_URL);
  } else if (fs.existsSync(distHtml)) {
    mainWindow.loadFile(distHtml);
  } else {
    mainWindow.loadURL(`http://127.0.0.1:${BACKEND_PORT}`);
  }

  mainWindow.once('ready-to-show', () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.show();
      mainWindow.focus();
    }
  });

  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.log(`[Browser Fail Load]: Code ${errorCode} - ${errorDescription}`);
    if (mainWindow && !mainWindow.isDestroyed() && !mainWindow.isVisible()) {
      mainWindow.show();
    }
  });

  mainWindow.webContents.on('render-process-gone', (event, details) => {
    console.error(`[Electron] Renderer process gone: ${details.reason} (exitCode: ${details.exitCode})`);
  });

  mainWindow.webContents.on('console-message', ({ level, message, lineNumber, sourceId }) => {
    console.log(`[Renderer Console] [${level}] ${message} (${sourceId}:${lineNumber})`);
  });

  // IPC / Window state update events
  mainWindow.on('maximize', () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('window-state-change', { maximized: true });
    }
  });

  mainWindow.on('unmaximize', () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('window-state-change', { maximized: false });
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Security: Intercept external navigation and window creation
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    try {
      const parsed = new URL(url);
      if (parsed.protocol === 'http:' || parsed.protocol === 'https:' || parsed.protocol === 'mailto:') {
        shell.openExternal(url);
      }
    } catch (e) {
      console.warn('[Electron] Blocked opening malformed window URL:', url);
    }
    return { action: 'deny' };
  });

  mainWindow.webContents.on('will-navigate', (event, url) => {
    const isLocalDev = url.startsWith('http://localhost:') || url.startsWith('http://127.0.0.1:');
    const isLocalFile = url.startsWith('file://');
    if (!isLocalDev && !isLocalFile) {
      event.preventDefault();
      try {
        shell.openExternal(url);
      } catch (e) {}
    }
  });
}

// ──────────────────────────────────────────────
// App lifecycle
// ──────────────────────────────────────────────
if (gotTheLock) {
  app.on('second-instance', () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
    }
  });

  app.whenReady().then(async () => {
    // Disable application menu globally
    Menu.setApplicationMenu(null);

    // 1. Create window immediately so screen appears instantly without delay
    createWindow();

    // 2. Start Python backend in parallel
    startPythonBackend();

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

  app.on('will-quit', () => {
    stopPythonBackend();
  });

  process.on('SIGINT', () => {
    stopPythonBackend();
    process.exit(0);
  });

  process.on('SIGTERM', () => {
    stopPythonBackend();
    process.exit(0);
  });
}

// ──────────────────────────────────────────────
// IPC handlers — window controls & pin state
// ──────────────────────────────────────────────
// Synchronous token handler for immediate access on startup
ipcMain.on('get-api-token-sync', (event) => {
  event.returnValue = API_TOKEN;
});

ipcMain.on('get-api-base-url-sync', (event) => {
  event.returnValue = BACKEND_BASE_URL;
});

// Asynchronous token handler
ipcMain.handle('get-api-token', () => API_TOKEN);
ipcMain.handle('get-backend-status', () => backendState);

// Structured UI Telemetry & Observability
ipcMain.on('log:ui-action', (event, action, details) => {
  const ts = new Date().toISOString().replace('T', ' ').slice(0, 19);
  let detailStr = '';
  if (details) {
    if (typeof details === 'string') {
      detailStr = details;
    } else {
      try {
        detailStr = Object.entries(details)
          .filter(([_, v]) => v !== undefined && v !== null && v !== '')
          .map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`)
          .join(' | ');
      } catch (e) {
        detailStr = String(details);
      }
    }
  }
  writeSessionLog(`[${ts}] [UI_ACTION] ${action}${detailStr ? ' -> ' + detailStr : ''}`);
});

function getTargetWindow(event) {
  if (event && event.sender) {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (win && !win.isDestroyed()) return win;
  }
  return (mainWindow && !mainWindow.isDestroyed()) ? mainWindow : null;
}

ipcMain.on('win:minimize', (event) => {
  const win = getTargetWindow(event);
  if (win) win.minimize();
});

ipcMain.on('win:maximize', (event) => {
  const win = getTargetWindow(event);
  if (win) {
    if (win.isMaximized()) win.unmaximize();
    else win.maximize();
  }
});

ipcMain.on('win:close', (event) => {
  const win = getTargetWindow(event);
  if (win) win.close();
});

ipcMain.handle('win:isMaximized', (event) => {
  const win = getTargetWindow(event);
  return win ? win.isMaximized() : false;
});

ipcMain.handle('win:toggleAlwaysOnTop', (event) => {
  const win = getTargetWindow(event);
  if (!win) return false;
  const current = win.isAlwaysOnTop();
  win.setAlwaysOnTop(!current);
  return !current;
});

ipcMain.handle('win:isAlwaysOnTop', (event) => {
  const win = getTargetWindow(event);
  return win ? win.isAlwaysOnTop() : true;
});

// Native HD Desktop Screen Capture handler
ipcMain.handle('screen:capture', async (event, options = {}) => {
  const win = getTargetWindow(event);
  let wasVisible = false;

  try {
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width, height } = primaryDisplay.size;
    const scaleFactor = primaryDisplay.scaleFactor || 1;

    const thumbnailSize = options.thumbnailSize || {
      width: Math.round(width * scaleFactor),
      height: Math.round(height * scaleFactor),
    };

    if (win && !win.isDestroyed() && win.isVisible()) {
      wasVisible = true;
      win.setOpacity(0);
    }

    await new Promise(r => setTimeout(r, 40));

    const sources = await desktopCapturer.getSources({
      types: options.types || ['screen'],
      thumbnailSize: thumbnailSize,
    });

    if (!sources || sources.length === 0) {
      throw new Error('No desktop screen sources found');
    }

    const primarySource = sources.find((s) => s.display_id === String(primaryDisplay.id)) || sources[0];
    const dataUrl = primarySource.thumbnail.toDataURL();

    return {
      success: true,
      dataUrl,
      resolution: `${thumbnailSize.width}x${thumbnailSize.height}`,
    };
  } catch (err) {
    console.error('[Electron] Desktop capture failed:', err);
    return { success: false, error: err.message };
  } finally {
    if (win && !win.isDestroyed() && wasVisible) {
      win.setOpacity(1);
    }
  }
});

// Native File Open Dialog
ipcMain.handle('dialog:openFile', async (event, options = {}) => {
  try {
    const win = getTargetWindow(event);
    const defaultFilters = [
      {
        name: 'Data Files',
        extensions: ['csv', 'xlsx', 'xls', 'parquet', 'json', 'sqlite', 'db', 'tsv', 'feather', 'arrow', 'txt'],
      },
      { name: 'All Files', extensions: ['*'] },
    ];

    const dialogOptions = {
      properties: ['openFile'],
      filters: defaultFilters,
      ...options,
    };

    const result = await dialog.showOpenDialog(win, dialogOptions);

    if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
      return null;
    }

    const normalizedPaths = result.filePaths.map(p => path.normalize(p).replace(/\\/g, '/'));
    return normalizedPaths;
  } catch (err) {
    console.error('[Electron] Error executing dialog:openFile:', err);
    return null;
  }
});

// Native Directory Open Dialog for Workspace Agent
ipcMain.handle('dialog:openDirectory', async (event, options = {}) => {
  try {
    const win = getTargetWindow(event);
    const dialogOptions = {
      properties: ['openDirectory', 'createDirectory'],
      title: options.title || 'Select Workspace Directory for DataPilot Agent',
      defaultPath: options.defaultPath,
      ...options,
    };

    const result = await dialog.showOpenDialog(win, dialogOptions);
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
      return null;
    }

    const normalizedPath = path.normalize(result.filePaths[0]).replace(/\\/g, '/');
    return normalizedPath;
  } catch (err) {
    console.error('[Electron] Error executing dialog:openDirectory:', err);
    return null;
  }
});

// Open external URL in default browser safely
const handleOpenExternal = (_, rawUrl) => {
  try {
    if (!rawUrl || typeof rawUrl !== 'string') return;
    const parsed = new URL(rawUrl);
    if (parsed.protocol === 'http:' || parsed.protocol === 'https:' || parsed.protocol === 'mailto:') {
      shell.openExternal(rawUrl);
    } else {
      console.warn(`[Electron] Blocked opening untrusted URL protocol: ${parsed.protocol}`);
    }
  } catch (err) {
    console.error(`[Electron] Invalid URL for open-external: ${rawUrl}`, err);
  }
};

ipcMain.on('open-external', handleOpenExternal);
ipcMain.on('shell:openExternal', handleOpenExternal);
ipcMain.handle('open-external', (event, url) => {
  handleOpenExternal(event, url);
  return true;
});
