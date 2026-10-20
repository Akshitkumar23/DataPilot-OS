/**
 * Frontend UI Action Telemetry Helper for DataPilot OS.
 * Dispatches structured user action events directly to the Electron
 * main process, writing to the unified session.log in real-time.
 */

export const logUserAction = (action, details = {}) => {
  try {
    if (typeof window !== 'undefined' && window.electron?.logUIAction) {
      window.electron.logUIAction(action, details);
    } else if (typeof window !== 'undefined' && window.electronAPI?.logUIAction) {
      window.electronAPI.logUIAction(action, details);
    } else {
      console.log(`[UI_ACTION] ${action}`, details);
    }
  } catch (err) {
    // Non-blocking telemetry
  }
};
