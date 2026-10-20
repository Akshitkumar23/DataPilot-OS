export const initialDatasetState = {
  datasets: [],
  activeDatasetId: null,
  relationships: [],
  isLoading: false,
  error: null,
};

export function normalizeDataset(ds) {
  if (!ds) return null;
  const id = ds.id || ds.dataset_id || `ds_${Date.now()}`;
  const name = ds.name || 'Dataset';
  const rowCount = ds.rowCount ?? ds.row_count ?? ds.rows ?? ds.total_rows ?? 0;
  const columns = ds.columns || ds.schema || [];
  const colCount = ds.colCount ?? ds.column_count ?? (Array.isArray(columns) ? columns.length : Object.keys(columns).length);
  const domain = ds.domain || ds.estimated_domain || null;
  const format = ds.format || ds.source_type || 'csv';
  const loadedAt = ds.loadedAt || ds.loaded_at || Date.now();

  return {
    ...ds,
    id,
    dataset_id: id,
    name,
    rowCount,
    row_count: rowCount,
    colCount,
    column_count: colCount,
    columns,
    schema: columns,
    domain,
    format,
    loadedAt,
  };
}

export function datasetReducer(state, action) {
  switch (action.type) {
    case 'SET_DATASETS': {
      const normalized = (action.payload || []).map(normalizeDataset).filter(Boolean);
      const currentActiveStillExists = normalized.some(d => d.id === state.activeDatasetId);
      const activeDatasetId = currentActiveStillExists
        ? state.activeDatasetId
        : (normalized.find(d => d.is_active)?.id || (normalized.length > 0 ? normalized[0].id : null));
      return {
        ...state,
        datasets: normalized,
        activeDatasetId,
        error: null,
      };
    }
    case 'ADD_DATASET': {
      const normalized = normalizeDataset(action.payload);
      if (!normalized) return state;
      const existsIndex = state.datasets.findIndex(d => d.id === normalized.id);
      let updatedDatasets;
      if (existsIndex >= 0) {
        updatedDatasets = state.datasets.map((d, i) => i === existsIndex ? { ...d, ...normalized } : d);
      } else {
        updatedDatasets = [...state.datasets, normalized];
      }
      return {
        ...state,
        datasets: updatedDatasets,
        activeDatasetId: normalized.id,
        error: null,
      };
    }
    case 'UPDATE_DATASET': {
      const id = action.payload?.id || action.payload?.dataset_id;
      if (!id) return state;
      const updatedDatasets = state.datasets.map(d =>
        d.id === id ? normalizeDataset({ ...d, ...action.payload }) : d
      );
      return {
        ...state,
        datasets: updatedDatasets,
      };
    }
    case 'REMOVE_DATASET': {
      const targetId = action.payload;
      const remaining = state.datasets.filter(d => d.id !== targetId && d.dataset_id !== targetId);
      const nextActiveId = state.activeDatasetId === targetId
        ? (remaining.length > 0 ? (remaining[0].id || remaining[0].dataset_id) : null)
        : state.activeDatasetId;
      return {
        ...state,
        datasets: remaining,
        activeDatasetId: nextActiveId,
      };
    }
    case 'SET_ACTIVE': {
      return {
        ...state,
        activeDatasetId: action.payload,
      };
    }
    case 'SET_RELATIONSHIPS':
      return { ...state, relationships: action.payload || [] };
    case 'SET_LOADING':
      return { ...state, isLoading: !!action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false };
    case 'RESET':
      return { ...initialDatasetState };
    default:
      return state;
  }
}
