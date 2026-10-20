import React, { createContext, useContext, useReducer } from 'react';
import { initialWorkflowState, workflowReducer } from './workflowStore';
import { initialDatasetState, datasetReducer } from './datasetStore';

export const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [workflowState, workflowDispatch] = useReducer(workflowReducer, initialWorkflowState);
  const [datasetState, datasetDispatch] = useReducer(datasetReducer, initialDatasetState);

  return (
    <AppContext.Provider value={{
      workflowState, workflowDispatch,
      datasetState, datasetDispatch
    }}>
      {children}
    </AppContext.Provider>
  );
}

export function useWorkflow() {
  const context = useContext(AppContext);
  if (!context) throw new Error('useWorkflow must be used within AppProvider');
  return { state: context.workflowState, dispatch: context.workflowDispatch };
}

export function useDatasets() {
  const context = useContext(AppContext);
  if (!context) throw new Error('useDatasets must be used within AppProvider');
  return { state: context.datasetState, dispatch: context.datasetDispatch };
}
