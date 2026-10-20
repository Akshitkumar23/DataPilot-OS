export const initialWorkflowState = {
  workflowId: null,
  userQuery: '',
  intent: null,
  clarification: null,
  graphState: null,
  codeVersion: null,
  validationResult: null,
  executionRecord: null,
  artifacts: [],
  validationReport: null,
  insight: null,
  productionState: null,
  phase: 'idle',
  error: null,
  isOllamaAvailable: false,
  ollamaStatus: null,
  messages: [],
};

export function workflowReducer(state, action) {
  switch (action.type) {
    case 'SET_WORKFLOW_ID': return { ...state, workflowId: action.payload };
    case 'SET_QUERY': return { ...state, userQuery: action.payload };
    case 'SET_INTENT': return { ...state, intent: action.payload };
    case 'SET_CLARIFICATION': return { ...state, clarification: action.payload };
    case 'SET_GRAPH': return { ...state, graphState: action.payload };
    case 'SET_CODE': return { ...state, codeVersion: action.payload };
    case 'SET_EXECUTION': return { ...state, executionRecord: action.payload };
    case 'SET_ARTIFACTS': return { ...state, artifacts: action.payload };
    case 'SET_VALIDATION': return { ...state, validationReport: action.payload };
    case 'SET_INSIGHT': return { ...state, insight: action.payload };
    case 'SET_PRODUCTION_STATE': return { ...state, productionState: action.payload };
    case 'SET_PHASE': return { ...state, phase: action.payload };
    case 'SET_ERROR': return { ...state, error: action.payload };
    case 'SET_OLLAMA_STATUS': return { ...state, isOllamaAvailable: action.payload.available, ollamaStatus: action.payload.status };
    case 'ADD_MESSAGE': return { ...state, messages: [...(state.messages || []), action.payload] };
    case 'PATCH_MESSAGE': {
      const msgs = [...(state.messages || [])];
      const { id, index, patch, updater } = action.payload || {};
      if (typeof index === 'number' && msgs[index]) {
        msgs[index] = typeof updater === 'function' ? updater(msgs[index]) : { ...msgs[index], ...(patch || action.payload) };
      } else if (id) {
        const idx = msgs.findIndex(m => m.id === id);
        if (idx !== -1) {
          msgs[idx] = typeof updater === 'function' ? updater(msgs[idx]) : { ...msgs[idx], ...(patch || action.payload) };
        }
      }
      return { ...state, messages: msgs };
    }
    case 'UPDATE_MESSAGE': {
      const msgs = [...(state.messages || [])];
      if (typeof action.payload?.index === 'number' && msgs[action.payload.index]) {
        msgs[action.payload.index] = typeof action.payload.updater === 'function' ? action.payload.updater(msgs[action.payload.index]) : { ...msgs[action.payload.index], ...action.payload };
      }
      return { ...state, messages: msgs };
    }
    case 'SET_MESSAGES': return { ...state, messages: action.payload };
    case 'CLEAR_MESSAGES': return { ...state, messages: [] };
    case 'RESET': return { ...initialWorkflowState };
    default: return state;
  }
}
