'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const projectRoot = path.resolve(__dirname, '..');
const python = process.platform === 'win32'
  ? path.join(projectRoot, '.venv', 'Scripts', 'python.exe')
  : path.join(projectRoot, '.venv', 'bin', 'python');

if (!fs.existsSync(python)) {
  console.error('A project virtual environment is required at .venv to package the backend.');
  process.exit(1);
}

const result = spawnSync(python, [
  '-m', 'PyInstaller', '--noconfirm', '--onedir', '--name', 'server',
  '--distpath', 'build/backend-dist', '--workpath', 'build/pyinstaller-work',
  '--specpath', 'build/pyinstaller-spec', '--paths', projectRoot,
  // NLTK is not used by DataPilot. Excluding it avoids an upstream Python 3.14 hook issue.
  '--exclude-module', 'nltk',
  '--exclude-module', 'pytest',
  '--collect-all', 'sklearn',
  '--collect-all', 'statsmodels',
  '--collect-all', 'duckdb',
  path.join('backend', 'production_entry.py')
], { cwd: projectRoot, stdio: 'inherit' });

process.exit(result.status ?? 1);
