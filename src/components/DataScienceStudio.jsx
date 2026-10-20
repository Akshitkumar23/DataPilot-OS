import { useState, useEffect, useMemo } from 'react';
import { 
  BarChart2, 
  Cpu, 
  Wrench, 
  Layers, 
  Sparkles, 
  Play, 
  CheckCircle2, 
  Database,
  TrendingUp,
  Table as TableIcon,
  Sliders,
  AlertCircle,
  HelpCircle,
  Activity,
  FileSpreadsheet,
  PieChart,
  Grid,
  Copy,
  Check
} from 'lucide-react';

import { API, getAuthHeaders } from '../apiConfig';
import MLStudio from './MLStudio';

export default function DataScienceStudio({
  schema,
  rowCount,
  activeDataset,
  datasetId,
  pipelineSteps,
  onRunCode,
  onRefresh,
  defaultTab = 'eda'
}) {
  const [activeTab, setActiveTab] = useState(defaultTab);
  const [edaSubTab, setEdaSubTab] = useState('descriptive'); // descriptive | missingness | distribution | correlation | skewness
  const [selectedTarget, setSelectedTarget] = useState('');
  const [selectedTask, setSelectedTask] = useState('auto'); // auto | classification | regression
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('auto'); // auto | random_forest | decision_tree | logistic_regression | linear_regression
  const [selectedValidationStrategy, setSelectedValidationStrategy] = useState('split'); // split | cv
  const [testSplitRatio, setTestSplitRatio] = useState(0.2);
  const [cvFolds, setCvFolds] = useState(5);
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [trainingResult, setTrainingResult] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [mlError, setMlError] = useState(null);
  const [lastMlAuditEvent, setLastMlAuditEvent] = useState(null);
  const [interactivePredictionInput, setInteractivePredictionInput] = useState({});
  const [interactivePredictionResult, setInteractivePredictionResult] = useState(null);
  const [predictionError, setPredictionError] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);

  // Phase 5: Unsupervised Studio State
  const [unsupSubTab, setUnsupSubTab] = useState('clustering'); // clustering | pca
  const [clusterFeatures, setClusterFeatures] = useState([]);
  const [clusterK, setClusterK] = useState(3);
  const [clusterScaling, setClusterScaling] = useState('standard');
  const [isClustering, setIsClustering] = useState(false);
  const [clusterResult, setClusterResult] = useState(null);
  const [clusterError, setClusterError] = useState(null);
  const [lastClusterAuditEvent, setLastClusterAuditEvent] = useState(null);
  const [elbowCurve, setElbowCurve] = useState(null);
  const [isComputingElbow, setIsComputingElbow] = useState(false);
  const [elbowError, setElbowError] = useState(null);

  // PCA State
  const [pcaFeatures, setPcaFeatures] = useState([]);
  const [pcaComponents, setPcaComponents] = useState(2);
  const [pcaScaling, setPcaScaling] = useState('standard');
  const [isPca, setIsPca] = useState(false);
  const [pcaResult, setPcaResult] = useState(null);
  const [pcaError, setPcaError] = useState(null);
  const [lastPcaAuditEvent, setLastPcaAuditEvent] = useState(null);
  const [copiedCodeType, setCopiedCodeType] = useState(null);

  // Phase 6: Time-Series & Forecasting Studio State
  const [tsSubTab, setTsSubTab] = useState('forecast'); // forecast | diagnostics
  const [tsTimeCol, setTsTimeCol] = useState('');
  const [tsTargetCol, setTsTargetCol] = useState('');
  const [tsHorizon, setTsHorizon] = useState(7);
  const [tsModel, setTsModel] = useState('arima'); // arima | sarimax | autoreg | naive
  const [tsParams, setTsParams] = useState({ p: 1, d: 1, q: 1, P: 1, D: 0, Q: 1, s: 7, lags: 2 });
  const [tsValidation, setTsValidation] = useState(null);
  const [isValidatingTs, setIsValidatingTs] = useState(false);
  const [tsValidateError, setTsValidateError] = useState(null);

  // Forecasting Execution State
  const [isForecasting, setIsForecasting] = useState(false);
  const [forecastResult, setForecastResult] = useState(null);
  const [forecastError, setForecastError] = useState(null);
  const [lastForecastAuditEvent, setLastForecastAuditEvent] = useState(null);

  // Diagnostics Execution State
  const [isTestingStationarity, setIsTestingStationarity] = useState(false);
  const [stationarityResult, setStationarityResult] = useState(null);
  const [stationarityError, setStationarityError] = useState(null);
  const [isTestingAcf, setIsTestingAcf] = useState(false);
  const [acfResult, setAcfResult] = useState(null);
  const [acfError, setAcfError] = useState(null);
  const [maxLags, setMaxLags] = useState(15);
  const [isDecomposing, setIsDecomposing] = useState(false);
  const [decompResult, setDecompResult] = useState(null);
  const [decompPeriod, setDecompPeriod] = useState(7);
  const [decompModel, setDecompModel] = useState('additive');
  const [decompError, setDecompError] = useState(null);

  // EDA Studio State
  const [edaLoading, setEdaLoading] = useState(false);
  const [edaError, setEdaError] = useState(null);
  const [edaResults, setEdaResults] = useState({
    descriptive: null,
    missingness: null,
    distribution: null,
    correlation: null,
    skewness: null,
  });
  const [lastAuditEvent, setLastAuditEvent] = useState(null);
  const [selectedDistCol, setSelectedDistCol] = useState('');
  const [distBins, setDistBins] = useState(10);
  const [corrMethod, setCorrMethod] = useState('pearson');

  // Track active dataset ID to invalidate stale results on dataset switch
  const currentDsId = datasetId || activeDataset?.id || activeDataset?.dataset_id || '';
  const [lastActiveDatasetId, setLastActiveDatasetId] = useState(currentDsId);

  useEffect(() => {
    if (defaultTab) {
      setActiveTab(defaultTab);
    }
  }, [defaultTab]);

  // Invalidate stale EDA & ML results whenever active dataset switches
  useEffect(() => {
    if (currentDsId && currentDsId !== lastActiveDatasetId) {
      setLastActiveDatasetId(currentDsId);
      setEdaResults({
        descriptive: null,
        missingness: null,
        distribution: null,
        correlation: null,
        skewness: null,
      });
      setLastAuditEvent(null);
      setEdaError(null);
      setSelectedDistCol('');

      // Invalidate ML state
      setSelectedTarget('');
      setSelectedFeatures([]);
      setTrainingResult(null);
      setMlError(null);
      setLastMlAuditEvent(null);
      setInteractivePredictionInput({});
      setInteractivePredictionResult(null);
      setPredictionError(null);

      // Invalidate Phase 5 Unsupervised state
      setClusterFeatures([]);
      setClusterResult(null);
      setClusterError(null);
      setLastClusterAuditEvent(null);
      setElbowCurve(null);
      setElbowError(null);
      setPcaFeatures([]);
      setPcaResult(null);
      setPcaError(null);
      setLastPcaAuditEvent(null);

      // Invalidate Phase 6 Time-Series state
      setTsTimeCol('');
      setTsTargetCol('');
      setTsValidation(null);
      setTsValidateError(null);
      setForecastResult(null);
      setForecastError(null);
      setLastForecastAuditEvent(null);
      setStationarityResult(null);
      setStationarityError(null);
      setAcfResult(null);
      setAcfError(null);
      setDecompResult(null);
      setDecompError(null);
      setGovernedOpFeedback(null);
      setGovernedOpLoading(null);
    }
  }, [currentDsId, lastActiveDatasetId]);

  // Derived column lists
  const { allColumns, numericColumns } = useMemo(() => {
    if (!schema) return { allColumns: [], numericColumns: [] };
    let cols = [];
    if (Array.isArray(schema)) {
      cols = schema.map(s => {
        if (typeof s === 'string') return { name: s, type: 'UNKNOWN' };
        if (typeof s === 'object' && s !== null) {
          const name = s.name || s.column_name || s.column || JSON.stringify(s);
          const type = (s.type || s.dtype || s.physical_type || '').toUpperCase();
          const isNum = type.includes('INT') || type.includes('FLOAT') || type.includes('NUM') || s.is_possible_measure;
          return { name, type: isNum ? 'NUM' : type || 'STR', isNumeric: isNum };
        }
        return { name: String(s), type: 'STR' };
      });
    } else if (typeof schema === 'object') {
      cols = Object.entries(schema).map(([name, type]) => {
        const typeStr = String(type).toUpperCase();
        const isNum = typeStr.includes('INT') || typeStr.includes('FLOAT') || typeStr.includes('NUM');
        return { name, type: isNum ? 'NUM' : typeStr, isNumeric: isNum };
      });
    }

    const all = cols.map(c => c.name);
    const nums = cols.filter(c => c.isNumeric || c.type === 'NUM').map(c => c.name);
    return { allColumns: all, numericColumns: nums.length > 0 ? nums : all };
  }, [schema]);

  const categoricalColumns = useMemo(() => {
    return allColumns.filter(c => !numericColumns.includes(c));
  }, [allColumns, numericColumns]);

  // Governed Clean & Transform State (replacing quarantined terminal runner)
  const [governedOpLoading, setGovernedOpLoading] = useState(null);
  const [governedOpFeedback, setGovernedOpFeedback] = useState(null);

  const handleApplyGovernedOp = async (operation, targetColumns, fillValue = null) => {
    if (!currentDsId) {
      setGovernedOpFeedback({ type: 'error', message: 'No active dataset connected.' });
      return;
    }
    if (!targetColumns || targetColumns.length === 0) {
      setGovernedOpFeedback({ type: 'error', message: `No compatible columns found in schema for ${operation}.` });
      return;
    }
    setGovernedOpLoading(operation);
    setGovernedOpFeedback(null);
    try {
      const res = await fetch(`${API}/api/operations/apply`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          dataset_id: currentDsId,
          operation: operation,
          columns: targetColumns,
          fill_value: fillValue
        })
      });
      const data = await res.json();
      if (!res.ok) {
        const errDetail = data?.detail?.message || data?.detail || data?.message || 'Operation failed';
        setGovernedOpFeedback({ type: 'error', message: `[ERROR] ${typeof errDetail === 'object' ? JSON.stringify(errDetail) : errDetail}` });
      } else {
        const rowDiff = data?.rows_affected ?? data?.rows_after ?? '';
        setGovernedOpFeedback({
          type: 'success',
          message: `[SUCCESS] Applied ${operation} on [${targetColumns.join(', ')}]. ${rowDiff ? `Rows: ${rowDiff}` : ''}`
        });
        if (onRefresh) {
          onRefresh(currentDsId);
        }
      }
    } catch (err) {
      setGovernedOpFeedback({ type: 'error', message: `[ERROR] Network error: ${err.message}` });
    } finally {
      setGovernedOpLoading(null);
    }
  };

  // Default distribution target column to first numeric column if unset
  useEffect(() => {
    if (!selectedDistCol && numericColumns.length > 0) {
      setSelectedDistCol(numericColumns[0]);
    }
  }, [numericColumns, selectedDistCol]);

  // Governed EDA Execution Handler
  const runEdaOperation = async (operation, extraPayload = {}) => {
    setEdaLoading(true);
    setEdaError(null);

    const payload = {
      dataset_id: currentDsId,
      operation,
      ...extraPayload
    };

    try {
      const resp = await fetch(`${API}/api/eda/execute`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(payload)
      });
      const data = await resp.json();

      if (resp.ok && data.status === 'SUCCEEDED') {
        const resultPayload = data.result;
        setEdaResults(prev => ({ ...prev, [operation]: resultPayload }));
        if (data.provenance_event) {
          setLastAuditEvent(data.provenance_event);
        }
      } else {
        const errDetail = data.detail || data.error || {};
        const errCode = errDetail.error_code || 'OPERATION_FAILED';
        const errMsg = errDetail.message || data.message || 'EDA operation could not be completed.';
        setEdaError({ code: errCode, message: errMsg });
      }
    } catch (err) {
      setEdaError({ code: 'NETWORK_ERROR', message: err.message });
    } finally {
      setEdaLoading(false);
    }
  };

  // Trigger handlers
  const handleRunDescriptive = () => {
    setEdaSubTab('descriptive');
    runEdaOperation('descriptive');
  };

  const handleRunMissingness = () => {
    setEdaSubTab('missingness');
    runEdaOperation('missingness');
  };

  const handleRunDistribution = () => {
    if (!selectedDistCol) {
      setEdaError({ code: 'MISSING_COLUMN', message: 'Please select a column for distribution analysis.' });
      return;
    }
    setEdaSubTab('distribution');
    runEdaOperation('distribution', { column: selectedDistCol, n_bins: distBins });
  };

  const handleRunCorrelation = () => {
    setEdaSubTab('correlation');
    runEdaOperation('correlation', { method: corrMethod });
  };

  const handleRunSkewness = () => {
    setEdaSubTab('skewness');
    runEdaOperation('skewness');
  };

  // Available features: allColumns minus selectedTarget
  const availableFeatures = useMemo(() => {
    return allColumns.filter(c => c !== selectedTarget);
  }, [allColumns, selectedTarget]);

  // When target changes, remove it from selectedFeatures
  useEffect(() => {
    if (selectedTarget) {
      setSelectedFeatures(prev => prev.filter(f => f !== selectedTarget));
    }
  }, [selectedTarget]);

  const toggleFeature = (colName) => {
    setSelectedFeatures(prev => {
      if (prev.includes(colName)) {
        return prev.filter(f => f !== colName);
      } else {
        return [...prev, colName];
      }
    });
  };

  const handleSelectAllFeatures = () => {
    setSelectedFeatures([...availableFeatures]);
  };

  const handleSelectNumericFeatures = () => {
    setSelectedFeatures(numericColumns.filter(c => c !== selectedTarget));
  };

  const handleClearFeatures = () => {
    setSelectedFeatures([]);
  };

  // Governed ML Training Handler
  const handleTrainMLModel = async () => {
    if (!selectedTarget) {
      setMlError({ code: 'MISSING_TARGET', message: 'Please select a target column to train the Machine Learning model.' });
      return;
    }
    setIsTraining(true);
    setMlError(null);
    setTrainingResult(null);
    setInteractivePredictionResult(null);

    const payload = {
      dataset_id: currentDsId,
      target: selectedTarget,
      features: selectedFeatures.length > 0 ? selectedFeatures : undefined,
      task: selectedTask,
      algorithm: selectedAlgorithm,
      validation_strategy: selectedValidationStrategy,
      test_size: testSplitRatio,
      cv_folds: cvFolds,
    };

    try {
      const resp = await fetch(`${API}/api/ml/train`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(payload)
      });
      const data = await resp.json();

      if (resp.ok && data.status === 'SUCCEEDED') {
        setTrainingResult(data.result);
        if (data.provenance_event) {
          setLastMlAuditEvent(data.provenance_event);
        }
        // Initialize interactive prediction inputs
        const initialInputs = {};
        (data.result.features || []).forEach(f => {
          initialInputs[f] = '10.0';
        });
        setInteractivePredictionInput(initialInputs);
        if (onRefresh) onRefresh();
      } else {
        const errDetail = data.detail || data.error || {};
        const errCode = errDetail.error_code || 'TRAINING_FAILED';
        const errMsg = errDetail.message || data.message || 'Machine learning model training could not be completed.';
        setMlError({ code: errCode, message: errMsg });
      }
    } catch (err) {
      setMlError({ code: 'NETWORK_ERROR', message: err.message });
    } finally {
      setIsTraining(false);
    }
  };

  // Interactive Live Prediction Handler
  const handleRunInteractivePrediction = async () => {
    if (!trainingResult || !trainingResult.model_id) return;
    setIsPredicting(true);
    setPredictionError(null);
    try {
      const resp = await fetch(`${API}/api/ml/models/${trainingResult.model_id}/predict`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          rows: [interactivePredictionInput]
        })
      });
      const data = await resp.json();
      if (resp.ok && data.status === 'SUCCEEDED') {
        setInteractivePredictionResult(data.result);
      } else {
        const errDetail = data.detail || data.error || {};
        setPredictionError(`[${errDetail.error_code || 'ERROR'}]: ${errDetail.message || 'Inference failed'}`);
      }
    } catch (err) {
      setPredictionError(`Prediction error: ${err.message}`);
    } finally {
      setIsPredicting(false);
    }
  };

  // Phase 5: Unsupervised Execution Handlers
  const handleRunKMeans = async () => {
    if (!currentDsId) {
      setClusterError('No active dataset connected.');
      return;
    }
    if (clusterFeatures.length < 2) {
      setClusterError('Feature validation error: At least 2 numerical features are required for clustering.');
      return;
    }
    setIsClustering(true);
    setClusterError(null);

    try {
      const res = await fetch(`${API}/api/unsupervised/kmeans`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          dataset_id: currentDsId,
          features: clusterFeatures,
          n_clusters: parseInt(clusterK, 10),
          scaling: clusterScaling
        })
      });

      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail || data;
        throw new Error(detail.error_message || detail.message || `Clustering failed (${res.status})`);
      }

      setClusterResult(data);
      if (data.provenance_event_id) {
        setLastClusterAuditEvent({
          event_id: data.provenance_event_id,
          execution_time_ms: data.execution_time_ms
        });
      }
    } catch (err) {
      console.error('K-Means execution failed:', err);
      setClusterError(err.message || 'Clustering execution failed');
    } finally {
      setIsClustering(false);
    }
  };

  const handleComputeElbow = async () => {
    if (!currentDsId || clusterFeatures.length === 0) return;
    setIsComputingElbow(true);
    setElbowError(null);

    try {
      const res = await fetch(`${API}/api/unsupervised/elbow`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          dataset_id: currentDsId,
          features: clusterFeatures,
          max_k: 8,
          scaling: clusterScaling
        })
      });

      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail || data;
        throw new Error(detail.error_message || detail.message || `Elbow evaluation failed (${res.status})`);
      }

      setElbowCurve(data.curve || []);
    } catch (err) {
      console.error('Elbow curve evaluation failed:', err);
      setElbowError(err.message || 'Elbow curve evaluation failed');
    } finally {
      setIsComputingElbow(false);
    }
  };

  const handleRunPCA = async () => {
    if (!currentDsId) {
      setPcaError('No active dataset connected.');
      return;
    }
    if (pcaFeatures.length < 2) {
      setPcaError('Feature validation error: PCA requires at least 2 numerical features.');
      return;
    }
    setIsPca(true);
    setPcaError(null);

    try {
      const res = await fetch(`${API}/api/unsupervised/pca`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          dataset_id: currentDsId,
          features: pcaFeatures,
          n_components: parseInt(pcaComponents, 10),
          scaling: pcaScaling
        })
      });

      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail || data;
        throw new Error(detail.error_message || detail.message || `PCA failed (${res.status})`);
      }

      setPcaResult(data);
      if (data.provenance_event_id) {
        setLastPcaAuditEvent({
          event_id: data.provenance_event_id,
          execution_time_ms: data.execution_time_ms
        });
      }
    } catch (err) {
      console.error('PCA execution failed:', err);
      setPcaError(err.message || 'PCA execution failed');
    } finally {
      setIsPca(false);
    }
  };

  // Phase 6: Time-Series & Forecasting Execution Handlers
  const handleValidateTimeSeries = async () => {
    if (!currentDsId) {
      setTsValidateError('No active dataset connected.');
      return;
    }
    if (!tsTimeCol || !tsTargetCol) {
      setTsValidateError('Please select both a temporal column and a numerical target column.');
      return;
    }
    setIsValidatingTs(true);
    setTsValidateError(null);
    try {
      const res = await fetch(`${API}/api/timeseries/validate`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          dataset_id: currentDsId,
          time_column: tsTimeCol,
          target_column: tsTargetCol
        })
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail || data;
        throw new Error(detail.error_message || detail.message || `Validation failed (${res.status})`);
      }
      setTsValidation(data);
    } catch (err) {
      console.error('Time series validation failed:', err);
      setTsValidateError(err.message || 'Validation failed');
    } finally {
      setIsValidatingTs(false);
    }
  };

  const handleRunForecast = async () => {
    if (!currentDsId) {
      setForecastError('No active dataset connected.');
      return;
    }
    if (!tsTimeCol || !tsTargetCol) {
      setForecastError('Feature validation error: Please select both a temporal column and a target column.');
      return;
    }
    setIsForecasting(true);
    setForecastError(null);
    try {
      const res = await fetch(`${API}/api/timeseries/forecast`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          dataset_id: currentDsId,
          time_column: tsTimeCol,
          target_column: tsTargetCol,
          model_type: tsModel,
          horizon: parseInt(tsHorizon, 10),
          parameters: tsParams
        })
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail || data;
        throw new Error(detail.error_message || detail.message || `Forecasting failed (${res.status})`);
      }
      setForecastResult(data);
      if (data.provenance_event_id) {
        setLastForecastAuditEvent({
          event_id: data.provenance_event_id,
          execution_time_ms: data.execution_time_ms
        });
      }
    } catch (err) {
      console.error('Forecast execution failed:', err);
      setForecastError(err.message || 'Forecasting failed');
    } finally {
      setIsForecasting(false);
    }
  };

  const handleRunStationarity = async () => {
    if (!currentDsId) {
      setStationarityError('No active dataset connected.');
      return;
    }
    if (!tsTimeCol || !tsTargetCol) {
      setStationarityError('Validation error: Please select both a temporal column and a target column.');
      return;
    }
    setIsTestingStationarity(true);
    setStationarityError(null);
    try {
      const res = await fetch(`${API}/api/timeseries/stationarity`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          dataset_id: currentDsId,
          time_column: tsTimeCol,
          target_column: tsTargetCol
        })
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail || data;
        throw new Error(detail.error_message || detail.message || `Stationarity test failed (${res.status})`);
      }
      setStationarityResult(data);
    } catch (err) {
      console.error('Stationarity test failed:', err);
      setStationarityError(err.message || 'Stationarity test failed');
    } finally {
      setIsTestingStationarity(false);
    }
  };

  const handleRunAcfPacf = async () => {
    if (!currentDsId) {
      setAcfError('No active dataset connected.');
      return;
    }
    if (!tsTimeCol || !tsTargetCol) {
      setAcfError('Validation error: Please select both a temporal column and a target column.');
      return;
    }
    setIsTestingAcf(true);
    setAcfError(null);
    try {
      const res = await fetch(`${API}/api/timeseries/acf-pacf`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          dataset_id: currentDsId,
          time_column: tsTimeCol,
          target_column: tsTargetCol,
          max_lags: parseInt(maxLags, 10)
        })
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail || data;
        throw new Error(detail.error_message || detail.message || `ACF/PACF computation failed (${res.status})`);
      }
      setAcfResult(data);
    } catch (err) {
      console.error('ACF/PACF computation failed:', err);
      setAcfError(err.message || 'ACF/PACF computation failed');
    } finally {
      setIsTestingAcf(false);
    }
  };

  const handleRunDecomposition = async () => {
    if (!currentDsId) {
      setDecompError('No active dataset connected.');
      return;
    }
    if (!tsTimeCol || !tsTargetCol) {
      setDecompError('Validation error: Please select both a temporal column and a target column.');
      return;
    }
    setIsDecomposing(true);
    setDecompError(null);
    try {
      const res = await fetch(`${API}/api/timeseries/decompose`, {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          dataset_id: currentDsId,
          time_column: tsTimeCol,
          target_column: tsTargetCol,
          period: parseInt(decompPeriod, 10),
          model: decompModel
        })
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail || data;
        throw new Error(detail.error_message || detail.message || `Decomposition failed (${res.status})`);
      }
      setDecompResult(data);
    } catch (err) {
      console.error('Decomposition failed:', err);
      setDecompError(err.message || 'Decomposition failed');
    } finally {
      setIsDecomposing(false);
    }
  };

  const handleCopyCode = (code, type) => {
    if (!code) return;
    navigator.clipboard.writeText(code);
    setCopiedCodeType(type);
    setTimeout(() => setCopiedCodeType(null), 2000);
  };

  // Helper for heatmap cell color interpolation: -1 (blue) -> 0 (neutral dark) -> +1 (crimson)
  const getCorrelationColor = (val) => {
    if (val === null || val === undefined || isNaN(val)) return 'rgba(30, 41, 59, 0.6)';
    if (val > 0) {
      const intensity = Math.min(val, 1.0);
      return `rgba(239, 68, 68, ${0.15 + intensity * 0.75})`;
    } else if (val < 0) {
      const intensity = Math.min(Math.abs(val), 1.0);
      return `rgba(59, 130, 246, ${0.15 + intensity * 0.75})`;
    }
    return 'rgba(30, 41, 59, 0.4)';
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: '#090d16',
      color: '#f8fafc',
      fontFamily: 'var(--font-sans, system-ui, sans-serif)',
      overflow: 'hidden'
    }}>
      {/* Studio Header */}
      <div style={{
        padding: '12px 20px',
        background: '#0f172a',
        borderBottom: '1px solid #1e293b',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            background: 'linear-gradient(135deg, #10b981, #0ea5e9)',
            padding: 8,
            borderRadius: 8,
            display: 'flex'
          }}>
            <Cpu size={20} color="#ffffff" />
          </div>
          <div>
            <h2 style={{ margin: 0, fontSize: 15, fontWeight: 800, color: '#f8fafc', letterSpacing: '-0.01em' }}>
              Master Data Science & AutoML Studio
            </h2>
            <p style={{ margin: 0, fontSize: 11, color: '#94a3b8' }}>
              Statistical Diagnostics, Real Distributions, Correlation & Supervised Modeling
            </p>
          </div>
        </div>

        {/* Dataset Meta & Audit Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {lastAuditEvent && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: '#064e3b',
              color: '#34d399',
              padding: '4px 10px',
              borderRadius: 14,
              fontSize: 11,
              fontWeight: 700,
              border: '1px solid #059669'
            }}>
              <Activity size={12} />
              <span>Audit: {lastAuditEvent.event_id}</span>
              <span style={{ color: '#a7f3d0' }}>({(lastAuditEvent.execution_time_ms || 0).toFixed(1)}ms)</span>
            </div>
          )}

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            background: '#1e293b',
            padding: '5px 12px',
            borderRadius: 20,
            border: '1px solid #334155',
            fontSize: 12,
            fontWeight: 600,
            color: '#38bdf8'
          }}>
            <Database size={13} color="#38bdf8" />
            <span>{(rowCount || 0).toLocaleString()} Rows</span>
            <span style={{ color: '#475569' }}>|</span>
            <span>{allColumns.length} Columns</span>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div style={{
        display: 'flex',
        background: '#0c121e',
        borderBottom: '1px solid #1e293b',
        padding: '0 16px',
        gap: 4,
        flexShrink: 0
      }}>
        {[
          { id: 'eda', label: 'Exploratory Analysis (EDA)', icon: BarChart2 },
          { id: 'prep', label: 'Governed Preparation', icon: Wrench },
          { id: 'feature', label: 'Feature Engineering', icon: Sliders },
          { id: 'ml', label: 'Machine Learning Studio', icon: Cpu },
          { id: 'unsupervised', label: 'Unsupervised Studio', icon: Layers },
          { id: 'timeseries', label: 'Time-Series & Forecasting', icon: TrendingUp },
          { id: 'tree', label: 'Provenance History', icon: Database }
        ].map(tab => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '10px 16px',
                background: isActive ? '#0f172a' : 'transparent',
                border: 'none',
                borderBottom: isActive ? '2px solid #0ea5e9' : '2px solid transparent',
                color: isActive ? '#38bdf8' : '#94a3b8',
                fontWeight: isActive ? 700 : 500,
                fontSize: 12,
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
            >
              <Icon size={15} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Studio Content Area */}
      <div style={{ flex: 1, padding: 18, overflowY: 'auto', background: '#090d16' }}>
        
        {/* ======================================================== */}
        {/* TAB 1: PRODUCTION-GRADE GOVERNED EDA & STATISTICS        */}
        {/* ======================================================== */}
        {activeTab === 'eda' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16, height: '100%' }}>
            
            {/* EDA Control Bar */}
            <div style={{
              background: '#0f172a',
              border: '1px solid #1e293b',
              borderRadius: 10,
              padding: 14,
              display: 'flex',
              flexWrap: 'wrap',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 12
            }}>
              {/* Primary Action Buttons */}
              <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8 }}>
                <button
                  onClick={handleRunDescriptive}
                  disabled={edaLoading}
                  style={{
                    background: edaSubTab === 'descriptive' ? '#0ea5e9' : '#1e293b',
                    color: '#ffffff',
                    border: '1px solid #334155',
                    borderRadius: 6,
                    padding: '8px 14px',
                    fontWeight: 700,
                    fontSize: 12,
                    cursor: edaLoading ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    transition: 'all 0.15s ease'
                  }}
                >
                  <TableIcon size={14} />
                  <span>Descriptive Stats</span>
                </button>

                <button
                  onClick={handleRunMissingness}
                  disabled={edaLoading}
                  style={{
                    background: edaSubTab === 'missingness' ? '#0ea5e9' : '#1e293b',
                    color: '#ffffff',
                    border: '1px solid #334155',
                    borderRadius: 6,
                    padding: '8px 14px',
                    fontWeight: 700,
                    fontSize: 12,
                    cursor: edaLoading ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    transition: 'all 0.15s ease'
                  }}
                >
                  <PieChart size={14} />
                  <span>Missingness Analysis</span>
                </button>

                <button
                  onClick={handleRunCorrelation}
                  disabled={edaLoading}
                  style={{
                    background: edaSubTab === 'correlation' ? '#0ea5e9' : '#1e293b',
                    color: '#ffffff',
                    border: '1px solid #334155',
                    borderRadius: 6,
                    padding: '8px 14px',
                    fontWeight: 700,
                    fontSize: 12,
                    cursor: edaLoading ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    transition: 'all 0.15s ease'
                  }}
                >
                  <TrendingUp size={14} />
                  <span>Correlation Matrix</span>
                </button>

                <button
                  onClick={handleRunSkewness}
                  disabled={edaLoading}
                  style={{
                    background: edaSubTab === 'skewness' ? '#0ea5e9' : '#1e293b',
                    color: '#ffffff',
                    border: '1px solid #334155',
                    borderRadius: 6,
                    padding: '8px 14px',
                    fontWeight: 700,
                    fontSize: 12,
                    cursor: edaLoading ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    transition: 'all 0.15s ease'
                  }}
                >
                  <Activity size={14} />
                  <span>Skewness & Normality</span>
                </button>
              </div>

              {/* Distribution Target Picker & Run Button */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, background: '#1e293b', padding: '4px 10px', borderRadius: 6, border: '1px solid #334155' }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8' }}>Distribution:</span>
                <select
                  value={selectedDistCol}
                  onChange={(e) => setSelectedDistCol(e.target.value)}
                  style={{
                    background: '#0f172a',
                    color: '#f8fafc',
                    border: '1px solid #334155',
                    borderRadius: 4,
                    padding: '4px 8px',
                    fontSize: 11,
                    fontWeight: 600,
                    maxWidth: 160
                  }}
                >
                  {numericColumns.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>

                <button
                  onClick={handleRunDistribution}
                  disabled={edaLoading || !selectedDistCol}
                  style={{
                    background: edaSubTab === 'distribution' ? '#0ea5e9' : '#10b981',
                    color: '#ffffff',
                    border: 'none',
                    borderRadius: 4,
                    padding: '4px 10px',
                    fontSize: 11,
                    fontWeight: 700,
                    cursor: edaLoading || !selectedDistCol ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4
                  }}
                >
                  <BarChart2 size={13} />
                  <span>Plot</span>
                </button>
              </div>
            </div>

            {/* Error Alert Banner */}
            {edaError && (
              <div style={{
                background: 'rgba(239, 68, 68, 0.12)',
                border: '1px solid #ef4444',
                borderRadius: 8,
                padding: '10px 14px',
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                color: '#fca5a5',
                fontSize: 12
              }}>
                <AlertCircle size={16} color="#ef4444" style={{ flexShrink: 0 }} />
                <div>
                  <strong style={{ color: '#f87171' }}>[{edaError.code}]</strong> {edaError.message}
                </div>
              </div>
            )}

            {/* Loading Indicator */}
            {edaLoading && (
              <div style={{
                padding: 30,
                textAlign: 'center',
                background: '#0f172a',
                borderRadius: 10,
                border: '1px solid #1e293b',
                color: '#38bdf8',
                fontSize: 13,
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 10
              }}>
                <div style={{
                  width: 16,
                  height: 16,
                  border: '2px solid #38bdf8',
                  borderTopColor: 'transparent',
                  borderRadius: '50%',
                  animation: 'spin 0.8s linear infinite'
                }} />
                <span>Computing Factual Statistical Metrics on Active DataFrame...</span>
              </div>
            )}

            {/* Sub-Tab Content View */}
            {!edaLoading && (
              <div style={{ flex: 1 }}>
                
                {/* SUB-VIEW 1: DESCRIPTIVE STATISTICS TABLE */}
                {edaSubTab === 'descriptive' && (
                  <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <TableIcon size={16} color="#38bdf8" />
                        <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700 }}>Numerical Descriptive Statistics</h3>
                      </div>
                      {edaResults.descriptive && (
                        <span style={{ fontSize: 11, color: '#94a3b8' }}>
                          {edaResults.descriptive.columns?.length || 0} numeric variables analyzed
                        </span>
                      )}
                    </div>

                    {edaResults.descriptive && edaResults.descriptive.statistics ? (
                      <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'left' }}>
                          <thead>
                            <tr style={{ background: '#1e293b', color: '#94a3b8', borderBottom: '1px solid #334155' }}>
                              <th style={{ padding: '8px 10px', fontWeight: 700 }}>Column</th>
                              <th style={{ padding: '8px 10px' }}>Count</th>
                              <th style={{ padding: '8px 10px' }}>Mean</th>
                              <th style={{ padding: '8px 10px' }}>Std Dev</th>
                              <th style={{ padding: '8px 10px' }}>Min</th>
                              <th style={{ padding: '8px 10px' }}>25%</th>
                              <th style={{ padding: '8px 10px' }}>Median (50%)</th>
                              <th style={{ padding: '8px 10px' }}>75%</th>
                              <th style={{ padding: '8px 10px' }}>Max</th>
                              <th style={{ padding: '8px 10px' }}>IQR</th>
                              <th style={{ padding: '8px 10px' }}>Variance</th>
                            </tr>
                          </thead>
                          <tbody>
                            {edaResults.descriptive.statistics.map((row, idx) => (
                              <tr key={row.column} style={{
                                borderBottom: '1px solid #1e293b',
                                background: idx % 2 === 0 ? 'transparent' : 'rgba(30, 41, 59, 0.25)'
                              }}>
                                <td style={{ padding: '8px 10px', fontWeight: 700, color: '#38bdf8' }}>{row.column}</td>
                                <td style={{ padding: '8px 10px', color: '#f8fafc' }}>{row.count?.toLocaleString()}</td>
                                <td style={{ padding: '8px 10px', color: '#f8fafc' }}>{row.mean !== null ? row.mean.toFixed(2) : '—'}</td>
                                <td style={{ padding: '8px 10px', color: '#f8fafc' }}>{row.std !== null ? row.std.toFixed(2) : '—'}</td>
                                <td style={{ padding: '8px 10px', color: '#f8fafc' }}>{row.min !== null ? row.min.toFixed(2) : '—'}</td>
                                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{row.q25 !== null ? row.q25.toFixed(2) : '—'}</td>
                                <td style={{ padding: '8px 10px', fontWeight: 700, color: '#10b981' }}>{row.median !== null ? row.median.toFixed(2) : '—'}</td>
                                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{row.q75 !== null ? row.q75.toFixed(2) : '—'}</td>
                                <td style={{ padding: '8px 10px', color: '#f8fafc' }}>{row.max !== null ? row.max.toFixed(2) : '—'}</td>
                                <td style={{ padding: '8px 10px', color: '#f8fafc' }}>{row.iqr !== null ? row.iqr.toFixed(2) : '—'}</td>
                                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{row.variance !== null ? row.variance.toFixed(2) : '—'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div style={{ textAlign: 'center', padding: '30px 10px', color: '#64748b' }}>
                        <FileSpreadsheet size={36} style={{ margin: '0 auto 10px', opacity: 0.5 }} />
                        <p style={{ margin: 0, fontSize: 13 }}>Click <strong>"Descriptive Stats"</strong> above to compute real statistical summaries for all numeric columns.</p>
                      </div>
                    )}
                  </div>
                )}

                {/* SUB-VIEW 2: MISSINGNESS ANALYSIS */}
                {edaSubTab === 'missingness' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    {edaResults.missingness && edaResults.missingness.summary && (
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 10 }}>
                        <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 12, textAlign: 'center' }}>
                          <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>Total Dataset Cells</span>
                          <strong style={{ fontSize: 16, color: '#f8fafc' }}>{edaResults.missingness.summary.total_cells?.toLocaleString()}</strong>
                        </div>
                        <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 12, textAlign: 'center' }}>
                          <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>Total Missing Cells</span>
                          <strong style={{ fontSize: 16, color: edaResults.missingness.summary.total_missing_cells > 0 ? '#f59e0b' : '#10b981' }}>
                            {edaResults.missingness.summary.total_missing_cells?.toLocaleString()}
                          </strong>
                        </div>
                        <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 12, textAlign: 'center' }}>
                          <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>Overall Missingness</span>
                          <strong style={{ fontSize: 16, color: edaResults.missingness.summary.overall_missing_percentage > 5 ? '#f87171' : '#38bdf8' }}>
                            {edaResults.missingness.summary.overall_missing_percentage}%
                          </strong>
                        </div>
                        <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 12, textAlign: 'center' }}>
                          <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>Complete Rows</span>
                          <strong style={{ fontSize: 16, color: '#10b981' }}>{edaResults.missingness.summary.complete_rows?.toLocaleString()}</strong>
                        </div>
                      </div>
                    )}

                    <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                        <PieChart size={16} color="#38bdf8" />
                        <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700 }}>Column-by-Column Missingness Ratios</h3>
                      </div>

                      {edaResults.missingness && edaResults.missingness.columns ? (
                        <div style={{ overflowX: 'auto' }}>
                          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'left' }}>
                            <thead>
                              <tr style={{ background: '#1e293b', color: '#94a3b8', borderBottom: '1px solid #334155' }}>
                                <th style={{ padding: '8px 10px', fontWeight: 700 }}>Column</th>
                                <th style={{ padding: '8px 10px' }}>Type</th>
                                <th style={{ padding: '8px 10px' }}>Missing Count</th>
                                <th style={{ padding: '8px 10px' }}>Missingness Ratio</th>
                                <th style={{ padding: '8px 10px' }}>Non-Missing Count</th>
                                <th style={{ padding: '8px 10px' }}>Data Quality Status</th>
                              </tr>
                            </thead>
                            <tbody>
                              {edaResults.missingness.columns.map((col, idx) => {
                                const statusColor = col.status === 'Complete' ? '#10b981' : col.status === 'Low Missingness' ? '#38bdf8' : col.status === 'Moderate Missingness' ? '#f59e0b' : '#ef4444';
                                return (
                                  <tr key={col.column} style={{
                                    borderBottom: '1px solid #1e293b',
                                    background: idx % 2 === 0 ? 'transparent' : 'rgba(30, 41, 59, 0.25)'
                                  }}>
                                    <td style={{ padding: '8px 10px', fontWeight: 700, color: '#f8fafc' }}>{col.column}</td>
                                    <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{col.dtype}</td>
                                    <td style={{ padding: '8px 10px', color: col.missing_count > 0 ? '#f59e0b' : '#94a3b8' }}>
                                      {col.missing_count}
                                    </td>
                                    <td style={{ padding: '8px 10px', width: 220 }}>
                                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                        <div style={{ flex: 1, background: '#1e293b', height: 6, borderRadius: 3, overflow: 'hidden' }}>
                                          <div style={{
                                            width: `${Math.min(col.missing_percentage, 100)}%`,
                                            background: col.missing_percentage > 20 ? '#ef4444' : col.missing_percentage > 0 ? '#f59e0b' : '#10b981',
                                            height: '100%'
                                          }} />
                                        </div>
                                        <span style={{ fontSize: 11, width: 44, textAlign: 'right', color: '#94a3b8' }}>
                                          {col.missing_percentage}%
                                        </span>
                                      </div>
                                    </td>
                                    <td style={{ padding: '8px 10px', color: '#f8fafc' }}>{col.non_missing_count}</td>
                                    <td style={{ padding: '8px 10px' }}>
                                      <span style={{
                                        fontSize: 10,
                                        fontWeight: 700,
                                        color: statusColor,
                                        background: `${statusColor}22`,
                                        padding: '2px 8px',
                                        borderRadius: 10,
                                        border: `1px solid ${statusColor}44`
                                      }}>
                                        {col.status}
                                      </span>
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <div style={{ textAlign: 'center', padding: '30px 10px', color: '#64748b' }}>
                          <PieChart size={36} style={{ margin: '0 auto 10px', opacity: 0.5 }} />
                          <p style={{ margin: 0, fontSize: 13 }}>Click <strong>"Missingness Analysis"</strong> above to audit missing cell rates and completeness.</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* SUB-VIEW 3: DISTRIBUTION HISTOGRAM */}
                {edaSubTab === 'distribution' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    {edaResults.distribution && (
                      <>
                        {/* Distribution Metrics Strip */}
                        {edaResults.distribution.summary && (
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 10 }}>
                            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 10, textAlign: 'center' }}>
                              <span style={{ fontSize: 11, color: '#94a3b8' }}>Sample Count</span>
                              <strong style={{ fontSize: 15, color: '#f8fafc', display: 'block' }}>{edaResults.distribution.sample_size}</strong>
                            </div>
                            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 10, textAlign: 'center' }}>
                              <span style={{ fontSize: 11, color: '#94a3b8' }}>Mean</span>
                              <strong style={{ fontSize: 15, color: '#38bdf8', display: 'block' }}>
                                {edaResults.distribution.summary.mean !== null ? edaResults.distribution.summary.mean.toFixed(2) : '—'}
                              </strong>
                            </div>
                            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 10, textAlign: 'center' }}>
                              <span style={{ fontSize: 11, color: '#94a3b8' }}>Median (Q2)</span>
                              <strong style={{ fontSize: 15, color: '#10b981', display: 'block' }}>
                                {edaResults.distribution.summary.median !== null ? edaResults.distribution.summary.median.toFixed(2) : '—'}
                              </strong>
                            </div>
                            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 10, textAlign: 'center' }}>
                              <span style={{ fontSize: 11, color: '#94a3b8' }}>Std Dev</span>
                              <strong style={{ fontSize: 15, color: '#f8fafc', display: 'block' }}>
                                {edaResults.distribution.summary.std !== null ? edaResults.distribution.summary.std.toFixed(2) : '—'}
                              </strong>
                            </div>
                            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 10, textAlign: 'center' }}>
                              <span style={{ fontSize: 11, color: '#94a3b8' }}>IQR Spread</span>
                              <strong style={{ fontSize: 15, color: '#f8fafc', display: 'block' }}>
                                {edaResults.distribution.summary.iqr !== null ? edaResults.distribution.summary.iqr.toFixed(2) : '—'}
                              </strong>
                            </div>
                            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 10, textAlign: 'center' }}>
                              <span style={{ fontSize: 11, color: '#94a3b8' }}>Outliers (1.5x IQR)</span>
                              <strong style={{ fontSize: 15, color: edaResults.distribution.summary.outlier_count > 0 ? '#f59e0b' : '#10b981', display: 'block' }}>
                                {edaResults.distribution.summary.outlier_count}
                              </strong>
                            </div>
                          </div>
                        )}

                        {/* Interactive Data-Driven SVG Histogram Chart */}
                        <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <BarChart2 size={16} color="#38bdf8" />
                              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700 }}>
                                Frequency Distribution: {edaResults.distribution.column}
                              </h3>
                            </div>
                            <span style={{ fontSize: 11, color: '#94a3b8' }}>
                              {edaResults.distribution.bins?.length} Bins | Total: {edaResults.distribution.sample_size} observations
                            </span>
                          </div>

                          {edaResults.distribution.bins && edaResults.distribution.bins.length > 0 && (() => {
                            const bins = edaResults.distribution.bins;
                            const maxCount = Math.max(...bins.map(b => b.count), 1);
                            const chartHeight = 180;
                            const chartWidth = 600;
                            const barWidth = (chartWidth / bins.length) - 6;

                            return (
                              <div style={{ width: '100%', overflowX: 'auto' }}>
                                <svg
                                  viewBox={`0 0 ${chartWidth} ${chartHeight + 40}`}
                                  style={{ width: '100%', maxHeight: 240, overflow: 'visible' }}
                                >
                                  {/* Grid Lines */}
                                  <line x1="0" y1={chartHeight} x2={chartWidth} y2={chartHeight} stroke="#334155" strokeWidth="1" />
                                  <line x1="0" y1={chartHeight / 2} x2={chartWidth} y2={chartHeight / 2} stroke="#1e293b" strokeDasharray="3 3" />

                                  {/* Bins */}
                                  {bins.map((b, i) => {
                                    const h = (b.count / maxCount) * (chartHeight - 20);
                                    const x = i * (chartWidth / bins.length) + 3;
                                    const y = chartHeight - h;

                                    return (
                                      <g key={b.bin_index} className="histogram-bar-group">
                                        <rect
                                          x={x}
                                          y={y}
                                          width={barWidth}
                                          height={h}
                                          fill="#0ea5e9"
                                          rx="3"
                                          opacity="0.85"
                                          style={{ transition: 'opacity 0.2s', cursor: 'pointer' }}
                                        >
                                          <title>{`Range: ${b.label}\nCount: ${b.count} (${b.percentage}%)`}</title>
                                        </rect>
                                        {/* Count Label on top of bar */}
                                        {b.count > 0 && (
                                          <text
                                            x={x + barWidth / 2}
                                            y={Math.max(y - 4, 12)}
                                            textAnchor="middle"
                                            fill="#94a3b8"
                                            fontSize="9"
                                            fontWeight="600"
                                          >
                                            {b.count}
                                          </text>
                                        )}
                                        {/* Bin X-axis Label */}
                                        <text
                                          x={x + barWidth / 2}
                                          y={chartHeight + 16}
                                          textAnchor="middle"
                                          fill="#64748b"
                                          fontSize="8"
                                          transform={`rotate(25, ${x + barWidth / 2}, ${chartHeight + 16})`}
                                        >
                                          {b.bin_start !== null ? b.bin_start : ''}
                                        </text>
                                      </g>
                                    );
                                  })}
                                </svg>
                              </div>
                            );
                          })()}
                        </div>
                      </>
                    )}

                    {!edaResults.distribution && (
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 30, textAlign: 'center', color: '#64748b' }}>
                        <BarChart2 size={36} style={{ margin: '0 auto 10px', opacity: 0.5 }} />
                        <p style={{ margin: 0, fontSize: 13 }}>
                          Select a target numeric column in the toolbar above and click <strong>"Plot"</strong> to generate a real histogram distribution.
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* SUB-VIEW 4: CORRELATION MATRIX & HEATMAP */}
                {edaSubTab === 'correlation' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <TrendingUp size={16} color="#38bdf8" />
                          <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700 }}>
                            Feature Correlation Matrix ({corrMethod.toUpperCase()})
                          </h3>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span style={{ fontSize: 11, color: '#94a3b8' }}>Method:</span>
                          <select
                            value={corrMethod}
                            onChange={(e) => setCorrMethod(e.target.value)}
                            style={{
                              background: '#1e293b',
                              color: '#f8fafc',
                              border: '1px solid #334155',
                              borderRadius: 4,
                              padding: '3px 8px',
                              fontSize: 11
                            }}
                          >
                            <option value="pearson">Pearson (Linear)</option>
                            <option value="spearman">Spearman (Rank)</option>
                          </select>
                          <button
                            onClick={handleRunCorrelation}
                            disabled={edaLoading}
                            style={{
                              background: '#0ea5e9',
                              color: '#ffffff',
                              border: 'none',
                              borderRadius: 4,
                              padding: '4px 10px',
                              fontSize: 11,
                              fontWeight: 700,
                              cursor: 'pointer'
                            }}
                          >
                            Recompute
                          </button>
                        </div>
                      </div>

                      {/* Real Correlation Heatmap Grid */}
                      {edaResults.correlation && edaResults.correlation.matrix ? (
                        <div>
                          <div style={{ overflowX: 'auto', marginBottom: 18 }}>
                            <table style={{ borderCollapse: 'collapse', fontSize: 11, margin: '0 auto' }}>
                              <thead>
                                <tr>
                                  <th style={{ padding: '6px 8px', color: '#64748b' }}></th>
                                  {edaResults.correlation.columns.map(col => (
                                    <th key={col} style={{
                                      padding: '6px 8px',
                                      color: '#94a3b8',
                                      maxWidth: 90,
                                      overflow: 'hidden',
                                      textOverflow: 'ellipsis',
                                      whiteSpace: 'nowrap'
                                    }}>
                                      {col}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {edaResults.correlation.matrix.map((row, rIdx) => {
                                  const rName = edaResults.correlation.columns[rIdx];
                                  return (
                                    <tr key={rName}>
                                      <td style={{
                                        padding: '6px 8px',
                                        fontWeight: 700,
                                        color: '#94a3b8',
                                        textAlign: 'right',
                                        maxWidth: 90,
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        whiteSpace: 'nowrap'
                                      }}>
                                        {rName}
                                      </td>
                                      {row.map((val, cIdx) => (
                                        <td
                                          key={cIdx}
                                          style={{
                                            padding: '8px 10px',
                                            textAlign: 'center',
                                            fontWeight: 700,
                                            background: getCorrelationColor(val),
                                            color: '#ffffff',
                                            border: '1px solid #1e293b',
                                            minWidth: 55
                                          }}
                                          title={`${rName} × ${edaResults.correlation.columns[cIdx]}: ${val !== null ? val : 'N/A'}`}
                                        >
                                          {val !== null ? val.toFixed(2) : '—'}
                                        </td>
                                      ))}
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          </div>

                          {/* Top Ranked Associated Pairs */}
                          {edaResults.correlation.pairs && edaResults.correlation.pairs.length > 0 && (
                            <div>
                              <h4 style={{ margin: '0 0 10px 0', fontSize: 12, fontWeight: 700, color: '#f8fafc' }}>
                                Ranked Correlation Pairs (Ordered by Absolute Strength)
                              </h4>
                              <div style={{ overflowX: 'auto' }}>
                                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, textAlign: 'left' }}>
                                  <thead>
                                    <tr style={{ background: '#1e293b', color: '#94a3b8' }}>
                                      <th style={{ padding: '6px 8px' }}>Feature A</th>
                                      <th style={{ padding: '6px 8px' }}>Feature B</th>
                                      <th style={{ padding: '6px 8px' }}>Correlation (r)</th>
                                      <th style={{ padding: '6px 8px' }}>Direction</th>
                                      <th style={{ padding: '6px 8px' }}>Association Strength</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {edaResults.correlation.pairs.slice(0, 10).map((pair, idx) => (
                                      <tr key={idx} style={{
                                        borderBottom: '1px solid #1e293b',
                                        background: idx % 2 === 0 ? 'transparent' : 'rgba(30, 41, 59, 0.25)'
                                      }}>
                                        <td style={{ padding: '6px 8px', fontWeight: 700, color: '#38bdf8' }}>{pair.feature_a}</td>
                                        <td style={{ padding: '6px 8px', fontWeight: 700, color: '#38bdf8' }}>{pair.feature_b}</td>
                                        <td style={{ padding: '6px 8px', fontWeight: 700, color: pair.correlation > 0 ? '#10b981' : '#f87171' }}>
                                          {pair.correlation !== null ? pair.correlation.toFixed(4) : '—'}
                                        </td>
                                        <td style={{ padding: '6px 8px', color: '#f8fafc' }}>{pair.direction}</td>
                                        <td style={{ padding: '6px 8px', color: '#94a3b8' }}>{pair.strength}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div style={{ textAlign: 'center', padding: '30px 10px', color: '#64748b' }}>
                          <TrendingUp size={36} style={{ margin: '0 auto 10px', opacity: 0.5 }} />
                          <p style={{ margin: 0, fontSize: 13 }}>Click <strong>"Correlation Matrix"</strong> above to compute linear or rank correlation coefficients across numeric features.</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* SUB-VIEW 5: SKEWNESS & NORMALITY CHECK */}
                {edaSubTab === 'skewness' && (
                  <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                      <Activity size={16} color="#38bdf8" />
                      <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700 }}>
                        Skewness, Kurtosis & Distribution Normality Assessment
                      </h3>
                    </div>

                    {edaResults.skewness && edaResults.skewness.results ? (
                      <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'left' }}>
                          <thead>
                            <tr style={{ background: '#1e293b', color: '#94a3b8', borderBottom: '1px solid #334155' }}>
                              <th style={{ padding: '8px 10px', fontWeight: 700 }}>Column</th>
                              <th style={{ padding: '8px 10px' }}>Type</th>
                              <th style={{ padding: '8px 10px' }}>Skewness Score</th>
                              <th style={{ padding: '8px 10px' }}>Kurtosis Score</th>
                              <th style={{ padding: '8px 10px' }}>Normality Shape</th>
                              <th style={{ padding: '8px 10px' }}>Tail Direction</th>
                              <th style={{ padding: '8px 10px' }}>Preprocessing Recommendation</th>
                            </tr>
                          </thead>
                          <tbody>
                            {edaResults.skewness.results.map((item, idx) => {
                              const badgeColor = item.classification === 'Symmetric / Normal' ? '#10b981' : item.classification === 'Moderately Skewed' ? '#f59e0b' : '#ef4444';
                              return (
                                <tr key={item.column} style={{
                                  borderBottom: '1px solid #1e293b',
                                  background: idx % 2 === 0 ? 'transparent' : 'rgba(30, 41, 59, 0.25)'
                                }}>
                                  <td style={{ padding: '8px 10px', fontWeight: 700, color: '#38bdf8' }}>{item.column}</td>
                                  <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{item.dtype}</td>
                                  <td style={{ padding: '8px 10px', fontWeight: 700, color: '#f8fafc' }}>
                                    {item.skewness !== null ? item.skewness.toFixed(3) : '—'}
                                  </td>
                                  <td style={{ padding: '8px 10px', color: '#f8fafc' }}>
                                    {item.kurtosis !== null ? item.kurtosis.toFixed(3) : '—'}
                                  </td>
                                  <td style={{ padding: '8px 10px' }}>
                                    <span style={{
                                      fontSize: 10,
                                      fontWeight: 700,
                                      color: badgeColor,
                                      background: `${badgeColor}22`,
                                      padding: '2px 8px',
                                      borderRadius: 10,
                                      border: `1px solid ${badgeColor}44`
                                    }}>
                                      {item.classification}
                                    </span>
                                  </td>
                                  <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{item.direction}</td>
                                  <td style={{ padding: '8px 10px', color: '#38bdf8', fontSize: 11 }}>{item.recommendation}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div style={{ textAlign: 'center', padding: '30px 10px', color: '#64748b' }}>
                        <Activity size={36} style={{ margin: '0 auto 10px', opacity: 0.5 }} />
                        <p style={{ margin: 0, fontSize: 13 }}>Click <strong>"Skewness & Normality"</strong> above to assess Gaussian bell curves vs high skewness.</p>
                      </div>
                    )}
                  </div>
                )}

              </div>
            )}
          </div>
        )}

        {/* ======================================================== */}
        {/* ======================================================== */}
        {/* TAB 2: Cleaning & Preprocessing (Governed)               */}
        {/* ======================================================== */}
        {activeTab === 'prep' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
            {governedOpFeedback && (
              <div style={{
                gridColumn: '1 / -1',
                padding: '10px 14px',
                borderRadius: 8,
                background: governedOpFeedback.type === 'error' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                border: `1px solid ${governedOpFeedback.type === 'error' ? '#ef4444' : '#10b981'}`,
                color: governedOpFeedback.type === 'error' ? '#fca5a5' : '#6ee7b7',
                fontSize: 12,
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}>
                <span>{governedOpFeedback.message}</span>
                <button onClick={() => setGovernedOpFeedback(null)} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', fontWeight: 700 }}>✕</button>
              </div>
            )}

            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 8 }}>
                IQR Outlier Capping (1.5x Boundary)
              </h3>
              <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 16 }}>
                Automatically detect and clip extreme values beyond upper and lower interquartile bounds [Q1 - 1.5*IQR, Q3 + 1.5*IQR] across numeric columns through the Governed Operations API.
              </p>
              <button
                onClick={() => handleApplyGovernedOp('clip_iqr', numericColumns)}
                disabled={governedOpLoading !== null || numericColumns.length === 0}
                style={{
                  width: '100%',
                  background: governedOpLoading === 'clip_iqr' ? '#047857' : '#10b981',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 6,
                  padding: '10px 14px',
                  fontWeight: 700,
                  fontSize: 12,
                  cursor: (governedOpLoading !== null || numericColumns.length === 0) ? 'not-allowed' : 'pointer',
                  opacity: (governedOpLoading !== null || numericColumns.length === 0) ? 0.6 : 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8
                }}
              >
                {governedOpLoading === 'clip_iqr' ? 'Applying Governed IQR Capping...' : 'Execute Governed IQR Outlier Capping'}
              </button>
            </div>

            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 8 }}>
                Automated Null Value Imputation
              </h3>
              <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 16 }}>
                Impute missing numeric values using statistical column mean through the Governed Operations API, updating the active dataset registry with SHA-256 provenance tracking.
              </p>
              <button
                onClick={() => handleApplyGovernedOp('impute_mean', numericColumns)}
                disabled={governedOpLoading !== null || numericColumns.length === 0}
                style={{
                  width: '100%',
                  background: governedOpLoading === 'impute_mean' ? '#047857' : '#10b981',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 6,
                  padding: '10px 14px',
                  fontWeight: 700,
                  fontSize: 12,
                  cursor: (governedOpLoading !== null || numericColumns.length === 0) ? 'not-allowed' : 'pointer',
                  opacity: (governedOpLoading !== null || numericColumns.length === 0) ? 0.6 : 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8
                }}
              >
                {governedOpLoading === 'impute_mean' ? 'Applying Governed Imputation...' : 'Execute Governed Mean Null Imputation'}
              </button>
            </div>
          </div>
        )}

        {/* ======================================================== */}
        {/* TAB 3: Feature Engineering (Governed)                     */}
        {/* ======================================================== */}
        {activeTab === 'feature' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
            {governedOpFeedback && (
              <div style={{
                gridColumn: '1 / -1',
                padding: '10px 14px',
                borderRadius: 8,
                background: governedOpFeedback.type === 'error' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                border: `1px solid ${governedOpFeedback.type === 'error' ? '#ef4444' : '#10b981'}`,
                color: governedOpFeedback.type === 'error' ? '#fca5a5' : '#6ee7b7',
                fontSize: 12,
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}>
                <span>{governedOpFeedback.message}</span>
                <button onClick={() => setGovernedOpFeedback(null)} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', fontWeight: 700 }}>✕</button>
              </div>
            )}

            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 8 }}>
                Yeo-Johnson Skewness Transformation
              </h3>
              <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 16 }}>
                Transform heavily non-normal numeric features into Gaussian bell curves to satisfy linear and distance model assumptions via the Governed Operations API.
              </p>
              <button
                onClick={() => handleApplyGovernedOp('yeo_johnson', numericColumns)}
                disabled={governedOpLoading !== null || numericColumns.length === 0}
                style={{
                  width: '100%',
                  background: governedOpLoading === 'yeo_johnson' ? '#0369a1' : '#0ea5e9',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 6,
                  padding: '10px 14px',
                  fontWeight: 700,
                  fontSize: 12,
                  cursor: (governedOpLoading !== null || numericColumns.length === 0) ? 'not-allowed' : 'pointer',
                  opacity: (governedOpLoading !== null || numericColumns.length === 0) ? 0.6 : 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8
                }}
              >
                {governedOpLoading === 'yeo_johnson' ? 'Applying Power Transform...' : 'Apply Governed Power Transformation'}
              </button>
            </div>

            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 8 }}>
                One-Hot Categorical Dummy Encoding
              </h3>
              <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 16 }}>
                Convert categorical text columns into numerical binary dummy columns for Machine Learning models via the Governed Operations API.
              </p>
              <button
                onClick={() => handleApplyGovernedOp('one_hot_encode', categoricalColumns)}
                disabled={governedOpLoading !== null || categoricalColumns.length === 0}
                style={{
                  width: '100%',
                  background: governedOpLoading === 'one_hot_encode' ? '#047857' : '#10b981',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 6,
                  padding: '10px 14px',
                  fontWeight: 700,
                  fontSize: 12,
                  cursor: (governedOpLoading !== null || categoricalColumns.length === 0) ? 'not-allowed' : 'pointer',
                  opacity: (governedOpLoading !== null || categoricalColumns.length === 0) ? 0.6 : 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8
                }}
              >
                {governedOpLoading === 'one_hot_encode' ? 'Applying One-Hot Encoding...' : (categoricalColumns.length > 0 ? 'Execute Governed One-Hot Dummy Encoding' : 'No Categorical Columns Found')}
              </button>
            </div>
          </div>
        )}

        {/* ======================================================== */}
        {/* TAB 4: PRODUCTION-GRADE GOVERNED ML & AUTOML STUDIO      */}
        {/* ======================================================== */}
        {activeTab === 'ml' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* ML Control Bar / Config Card */}
            <div style={{
              background: '#0f172a',
              border: '1px solid #1e293b',
              borderRadius: 10,
              padding: 20,
              display: 'flex',
              flexDirection: 'column',
              gap: 16
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid #1e293b', paddingBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Sparkles size={18} color="#0ea5e9" />
                  <div>
                    <h3 style={{ margin: 0, fontSize: 15, fontWeight: 800, color: '#f8fafc' }}>
                      Supervised Machine Learning & AutoML Studio
                    </h3>
                    <p style={{ margin: 0, fontSize: 11, color: '#94a3b8' }}>
                      Deterministic Preprocessing Pipelines, Held-Out Cross-Validation & Live Model Artifacts
                    </p>
                  </div>
                </div>

                {lastMlAuditEvent && (
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    background: '#064e3b',
                    color: '#34d399',
                    padding: '4px 10px',
                    borderRadius: 14,
                    fontSize: 11,
                    fontWeight: 700,
                    border: '1px solid #059669'
                  }}>
                    <Activity size={12} />
                    <span>Audit: {lastMlAuditEvent.event_id}</span>
                    <span style={{ color: '#a7f3d0' }}>({(lastMlAuditEvent.execution_time_ms || 0).toFixed(1)}ms)</span>
                  </div>
                )}
              </div>

              {/* Grid of Inputs: Target, Task, Algorithm, Validation */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14 }}>
                {/* Target Column */}
                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#94a3b8', marginBottom: 6 }}>
                    Target Column (y) *
                  </label>
                  <select
                    value={selectedTarget}
                    onChange={(e) => setSelectedTarget(e.target.value)}
                    style={{
                      width: '100%',
                      background: '#1e293b',
                      color: '#f8fafc',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 10px',
                      fontSize: 12,
                      fontWeight: 600
                    }}
                  >
                    <option value="">-- Choose Target --</option>
                    {allColumns.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>

                {/* Task Type */}
                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#94a3b8', marginBottom: 6 }}>
                    Learning Task
                  </label>
                  <select
                    value={selectedTask}
                    onChange={(e) => setSelectedTask(e.target.value)}
                    style={{
                      width: '100%',
                      background: '#1e293b',
                      color: '#f8fafc',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 10px',
                      fontSize: 12,
                      fontWeight: 600
                    }}
                  >
                    <option value="auto">Auto-Detect (Schema-Driven)</option>
                    <option value="classification">Classification (Discrete / Class)</option>
                    <option value="regression">Regression (Continuous Numeric)</option>
                  </select>
                </div>

                {/* Algorithm */}
                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#94a3b8', marginBottom: 6 }}>
                    Model / Algorithm
                  </label>
                  <select
                    value={selectedAlgorithm}
                    onChange={(e) => setSelectedAlgorithm(e.target.value)}
                    style={{
                      width: '100%',
                      background: '#1e293b',
                      color: '#f8fafc',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 10px',
                      fontSize: 12,
                      fontWeight: 600
                    }}
                  >
                    <option value="auto">🏆 AutoML Tournament (Benchmark & Rank All)</option>
                    <option value="random_forest">🌲 Random Forest</option>
                    <option value="decision_tree">🌳 Decision Tree</option>
                    <option value="logistic_regression">📈 Logistic / Linear Regression</option>
                  </select>
                </div>

                {/* Validation Strategy */}
                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#94a3b8', marginBottom: 6 }}>
                    Validation Strategy
                  </label>
                  <select
                    value={selectedValidationStrategy}
                    onChange={(e) => setSelectedValidationStrategy(e.target.value)}
                    style={{
                      width: '100%',
                      background: '#1e293b',
                      color: '#f8fafc',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 10px',
                      fontSize: 12,
                      fontWeight: 600
                    }}
                  >
                    <option value="split">Held-Out Test Split (80% Train / 20% Test)</option>
                    <option value="cv">5-Fold Cross-Validation</option>
                  </select>
                </div>
              </div>

              {/* Feature Selection Filter Strip */}
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                  <label style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8' }}>
                    Feature Columns (X) — {selectedFeatures.length > 0 ? `${selectedFeatures.length} selected` : 'All available features (default)'}
                  </label>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      type="button"
                      onClick={handleSelectAllFeatures}
                      style={{ background: '#1e293b', color: '#38bdf8', border: '1px solid #334155', borderRadius: 4, padding: '3px 8px', fontSize: 10, cursor: 'pointer' }}
                    >
                      Select All
                    </button>
                    <button
                      type="button"
                      onClick={handleSelectNumericFeatures}
                      style={{ background: '#1e293b', color: '#10b981', border: '1px solid #334155', borderRadius: 4, padding: '3px 8px', fontSize: 10, cursor: 'pointer' }}
                    >
                      Numeric Only
                    </button>
                    <button
                      type="button"
                      onClick={handleClearFeatures}
                      style={{ background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', borderRadius: 4, padding: '3px 8px', fontSize: 10, cursor: 'pointer' }}
                    >
                      Clear
                    </button>
                  </div>
                </div>

                <div style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: 6,
                  maxHeight: 110,
                  overflowY: 'auto',
                  background: '#090d16',
                  padding: 8,
                  borderRadius: 6,
                  border: '1px solid #1e293b'
                }}>
                  {availableFeatures.map(col => {
                    const isSelected = selectedFeatures.includes(col);
                    return (
                      <button
                        key={col}
                        type="button"
                        onClick={() => toggleFeature(col)}
                        style={{
                          background: isSelected ? 'rgba(14, 165, 233, 0.2)' : '#1e293b',
                          color: isSelected ? '#38bdf8' : '#94a3b8',
                          border: isSelected ? '1px solid #0ea5e9' : '1px solid #334155',
                          borderRadius: 4,
                          padding: '4px 8px',
                          fontSize: 11,
                          fontWeight: 600,
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 4
                        }}
                      >
                        <span style={{ fontSize: 9 }}>{isSelected ? '✓' : '+'}</span>
                        <span>{col}</span>
                      </button>
                    );
                  })}
                  {availableFeatures.length === 0 && (
                    <span style={{ fontSize: 11, color: '#64748b' }}>No features available (dataset has only 1 column).</span>
                  )}
                </div>
              </div>

              {/* Train Button */}
              <button
                onClick={handleTrainMLModel}
                disabled={isTraining || !selectedTarget}
                style={{
                  width: '100%',
                  background: isTraining ? '#334155' : 'linear-gradient(135deg, #10b981, #0ea5e9)',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: 8,
                  padding: '12px 18px',
                  fontWeight: 800,
                  fontSize: 13,
                  cursor: isTraining || !selectedTarget ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8,
                  boxShadow: '0 4px 14px rgba(14,165,233,0.3)',
                  transition: 'all 0.15s ease'
                }}
              >
                {isTraining ? (
                  <>
                    <div style={{
                      width: 14,
                      height: 14,
                      border: '2px solid #ffffff',
                      borderTopColor: 'transparent',
                      borderRadius: '50%',
                      animation: 'spin 0.8s linear infinite'
                    }} />
                    <span>Training Pipeline & Evaluating Out-of-Sample Metrics...</span>
                  </>
                ) : (
                  <>
                    <Play size={15} />
                    <span>
                      {selectedAlgorithm === 'auto'
                        ? 'Execute AutoML Tournament (Benchmark & Rank All Models)'
                        : `Train ${selectedAlgorithm.replace('_', ' ').toUpperCase()} Model & Evaluate`}
                    </span>
                  </>
                )}
              </button>
            </div>

            {/* Error Alert Banner */}
            {mlError && (
              <div style={{
                background: 'rgba(239, 68, 68, 0.12)',
                border: '1px solid #ef4444',
                borderRadius: 8,
                padding: '10px 14px',
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                color: '#fca5a5',
                fontSize: 12
              }}>
                <AlertCircle size={16} color="#ef4444" style={{ flexShrink: 0 }} />
                <div>
                  <strong style={{ color: '#f87171' }}>[{mlError.code}]</strong> {mlError.message}
                </div>
              </div>
            )}

            {/* Results Presentation View */}
            {trainingResult && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                
                {/* Result Header Strip */}
                <div style={{
                  background: '#0f172a',
                  border: '1px solid #10b981',
                  borderRadius: 10,
                  padding: 16,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: 12
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{ background: '#064e3b', padding: 8, borderRadius: 8, color: '#34d399' }}>
                      <CheckCircle2 size={20} />
                    </div>
                    <div>
                      <h4 style={{ margin: 0, fontSize: 15, fontWeight: 800, color: '#f8fafc' }}>
                        Trained Model: {trainingResult.algorithm_name}
                      </h4>
                      <p style={{ margin: 0, fontSize: 11, color: '#94a3b8' }}>
                        Target: <strong style={{ color: '#38bdf8' }}>{trainingResult.target}</strong> | Task: <strong style={{ color: '#10b981', textTransform: 'capitalize' }}>{trainingResult.task}</strong> | Model ID: <span style={{ color: '#a7f3d0' }}>{trainingResult.model_id}</span>
                      </p>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: 12, fontSize: 11, color: '#94a3b8' }}>
                    <span>Train: <strong>{trainingResult.sample_counts?.train}</strong></span>
                    <span>•</span>
                    <span>Test: <strong>{trainingResult.sample_counts?.test}</strong></span>
                    <span>•</span>
                    <span>Fit: <strong>{(trainingResult.timing?.fit_time_ms || 0).toFixed(1)}ms</strong></span>
                  </div>
                </div>

                {/* Primary Metric KPI Cards */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                  {trainingResult.task === 'classification' ? (
                    <>
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14, textAlign: 'center' }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>Accuracy Score</span>
                        <strong style={{ fontSize: 20, color: '#38bdf8' }}>
                          {(trainingResult.metrics?.accuracy * 100).toFixed(2)}%
                        </strong>
                      </div>
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14, textAlign: 'center' }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>Weighted F1</span>
                        <strong style={{ fontSize: 20, color: '#10b981' }}>
                          {(trainingResult.metrics?.f1_weighted * 100).toFixed(2)}%
                        </strong>
                      </div>
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14, textAlign: 'center' }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>Macro Precision</span>
                        <strong style={{ fontSize: 20, color: '#f8fafc' }}>
                          {(trainingResult.metrics?.precision_macro * 100).toFixed(2)}%
                        </strong>
                      </div>
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14, textAlign: 'center' }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>Macro Recall</span>
                        <strong style={{ fontSize: 20, color: '#f8fafc' }}>
                          {(trainingResult.metrics?.recall_macro * 100).toFixed(2)}%
                        </strong>
                      </div>
                    </>
                  ) : (
                    <>
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14, textAlign: 'center' }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>R² Score (Goodness of Fit)</span>
                        <strong style={{ fontSize: 20, color: '#38bdf8' }}>
                          {trainingResult.metrics?.r2 !== null ? `${(trainingResult.metrics.r2 * 100).toFixed(2)}%` : '—'}
                        </strong>
                      </div>
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14, textAlign: 'center' }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>RMSE (Root Mean Sq Error)</span>
                        <strong style={{ fontSize: 20, color: '#10b981' }}>
                          {trainingResult.metrics?.rmse !== null ? trainingResult.metrics.rmse.toFixed(4) : '—'}
                        </strong>
                      </div>
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14, textAlign: 'center' }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>MAE (Mean Absolute Error)</span>
                        <strong style={{ fontSize: 20, color: '#f8fafc' }}>
                          {trainingResult.metrics?.mae !== null ? trainingResult.metrics.mae.toFixed(4) : '—'}
                        </strong>
                      </div>
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14, textAlign: 'center' }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', display: 'block' }}>MSE (Mean Squared Error)</span>
                        <strong style={{ fontSize: 20, color: '#94a3b8' }}>
                          {trainingResult.metrics?.mse !== null ? trainingResult.metrics.mse.toFixed(4) : '—'}
                        </strong>
                      </div>
                    </>
                  )}
                </div>

                {/* Classification: Confusion Matrix Grid */}
                {trainingResult.task === 'classification' && trainingResult.metrics?.confusion_matrix && (
                  <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                      <h4 style={{ margin: 0, fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                        Confusion Matrix (Rows = Actual Class, Columns = Predicted Class)
                      </h4>
                      <span style={{ fontSize: 11, color: '#94a3b8' }}>
                        {trainingResult.metrics.confusion_matrix.classes?.length} Distinct Classes
                      </span>
                    </div>

                    <div style={{ overflowX: 'auto', marginBottom: 14 }}>
                      <table style={{ borderCollapse: 'collapse', fontSize: 11, margin: '0 auto' }}>
                        <thead>
                          <tr>
                            <th style={{ padding: '6px 10px', color: '#64748b' }}>Actual \ Pred</th>
                            {trainingResult.metrics.confusion_matrix.classes.map(cls => (
                              <th key={cls} style={{ padding: '6px 10px', color: '#38bdf8', fontWeight: 700 }}>
                                {cls}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {trainingResult.metrics.confusion_matrix.matrix.map((row, rIdx) => {
                            const actualLabel = trainingResult.metrics.confusion_matrix.classes[rIdx];
                            return (
                              <tr key={actualLabel}>
                                <td style={{ padding: '6px 10px', fontWeight: 700, color: '#38bdf8', textAlign: 'right' }}>
                                  {actualLabel}
                                </td>
                                {row.map((count, cIdx) => {
                                  const isDiagonal = rIdx === cIdx;
                                  return (
                                    <td
                                      key={cIdx}
                                      style={{
                                        padding: '8px 12px',
                                        textAlign: 'center',
                                        fontWeight: 700,
                                        background: isDiagonal && count > 0 ? 'rgba(16, 185, 129, 0.25)' : count > 0 ? 'rgba(239, 68, 68, 0.2)' : 'rgba(30, 41, 59, 0.3)',
                                        color: isDiagonal && count > 0 ? '#34d399' : count > 0 ? '#fca5a5' : '#64748b',
                                        border: '1px solid #1e293b',
                                        minWidth: 45
                                      }}
                                    >
                                      {count}
                                    </td>
                                  );
                                })}
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>

                    {/* Class Distribution Tags */}
                    {trainingResult.metrics.class_distribution && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', fontSize: 11 }}>
                        <span style={{ color: '#94a3b8' }}>Test Class Balance:</span>
                        {Object.entries(trainingResult.metrics.class_distribution).map(([label, count]) => (
                          <span
                            key={label}
                            style={{
                              background: '#1e293b',
                              color: '#f8fafc',
                              padding: '2px 8px',
                              borderRadius: 4,
                              border: '1px solid #334155'
                            }}
                          >
                            {label}: <strong>{count}</strong>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* FW-02: Asymmetric Decision Policy & Threshold Optimization */}
                {trainingResult.task === 'classification' && (
                  <MLStudio
                    datasetId={currentDsId}
                    trainingResult={trainingResult}
                    onRefresh={onRefresh}
                  />
                )}

                {/* Regression: Residuals Summary */}
                {trainingResult.task === 'regression' && trainingResult.metrics?.residuals && (
                  <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 16 }}>
                    <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                      Residual Error Distribution (Actual - Predicted)
                    </h4>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, fontSize: 12 }}>
                      <div style={{ background: '#1e293b', padding: 8, borderRadius: 6, textAlign: 'center' }}>
                        <span style={{ fontSize: 10, color: '#94a3b8', display: 'block' }}>Min Residual</span>
                        <strong style={{ color: '#f8fafc' }}>{trainingResult.metrics.residuals.min.toFixed(4)}</strong>
                      </div>
                      <div style={{ background: '#1e293b', padding: 8, borderRadius: 6, textAlign: 'center' }}>
                        <span style={{ fontSize: 10, color: '#94a3b8', display: 'block' }}>Mean Residual</span>
                        <strong style={{ color: '#38bdf8' }}>{trainingResult.metrics.residuals.mean.toFixed(4)}</strong>
                      </div>
                      <div style={{ background: '#1e293b', padding: 8, borderRadius: 6, textAlign: 'center' }}>
                        <span style={{ fontSize: 10, color: '#f8fafc', display: 'block' }}>Max Residual</span>
                        <strong style={{ color: '#f8fafc' }}>{trainingResult.metrics.residuals.max.toFixed(4)}</strong>
                      </div>
                      <div style={{ background: '#1e293b', padding: 8, borderRadius: 6, textAlign: 'center' }}>
                        <span style={{ fontSize: 10, color: '#94a3b8', display: 'block' }}>Std Dev Residual</span>
                        <strong style={{ color: '#10b981' }}>{trainingResult.metrics.residuals.std.toFixed(4)}</strong>
                      </div>
                    </div>
                  </div>
                )}

                {/* Cross-Validation Strip (if CV performed) */}
                {trainingResult.cross_validation?.performed && (
                  <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 16 }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                      <h4 style={{ margin: 0, fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                        K-Fold Cross-Validation Scores
                      </h4>
                      <span style={{ fontSize: 12, fontWeight: 700, color: '#38bdf8' }}>
                        Mean: {(trainingResult.cross_validation.mean * 100).toFixed(2)}% (± {(trainingResult.cross_validation.std * 100).toFixed(2)}%)
                      </span>
                    </div>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {trainingResult.cross_validation.scores.map((score, sIdx) => (
                        <div key={sIdx} style={{ background: '#1e293b', padding: '4px 10px', borderRadius: 4, fontSize: 11 }}>
                          <span style={{ color: '#94a3b8' }}>Fold {sIdx + 1}: </span>
                          <strong style={{ color: '#f8fafc' }}>{(score * 100).toFixed(2)}%</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* AutoML Tournament Leaderboard Table */}
                {trainingResult.tournament && trainingResult.tournament.length > 0 && (
                  <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                    <h4 style={{ margin: '0 0 12px 0', fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                      🏆 AutoML Candidate Tournament Leaderboard
                    </h4>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, textAlign: 'left' }}>
                        <thead>
                          <tr style={{ background: '#1e293b', color: '#94a3b8' }}>
                            <th style={{ padding: '6px 10px' }}>Rank</th>
                            <th style={{ padding: '6px 10px' }}>Algorithm</th>
                            <th style={{ padding: '6px 10px' }}>Model Family</th>
                            <th style={{ padding: '6px 10px' }}>Primary Score</th>
                            <th style={{ padding: '6px 10px' }}>Fit Time</th>
                            <th style={{ padding: '6px 10px' }}>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {trainingResult.tournament.map((cand) => {
                            const isWinner = cand.rank === 1;
                            return (
                              <tr key={cand.algorithm_id} style={{
                                borderBottom: '1px solid #1e293b',
                                background: isWinner ? 'rgba(16, 185, 129, 0.12)' : 'transparent'
                              }}>
                                <td style={{ padding: '8px 10px', fontWeight: 800, color: isWinner ? '#10b981' : '#94a3b8' }}>
                                  #{cand.rank}
                                </td>
                                <td style={{ padding: '8px 10px', fontWeight: 700, color: '#f8fafc' }}>
                                  {cand.algorithm_name}
                                </td>
                                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>
                                  {cand.family}
                                </td>
                                <td style={{ padding: '8px 10px', fontWeight: 700, color: isWinner ? '#10b981' : '#38bdf8' }}>
                                  {(cand.primary_score * 100).toFixed(2)}% ({cand.primary_metric})
                                </td>
                                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>
                                  {cand.fit_time_ms.toFixed(1)}ms
                                </td>
                                <td style={{ padding: '8px 10px' }}>
                                  <span style={{
                                    fontSize: 10,
                                    fontWeight: 700,
                                    padding: '2px 6px',
                                    borderRadius: 4,
                                    background: isWinner ? '#064e3b' : '#1e293b',
                                    color: isWinner ? '#34d399' : '#94a3b8'
                                  }}>
                                    {isWinner ? 'Selected Best' : 'Evaluated'}
                                  </span>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Feature Importance Ranking */}
                {trainingResult.feature_importances && trainingResult.feature_importances.length > 0 && (
                  <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                    <h4 style={{ margin: '0 0 12px 0', fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                      🌲 Feature Importance Weights (Relative Contribution)
                    </h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                      {trainingResult.feature_importances.slice(0, 10).map((item) => (
                        <div key={item.feature} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12 }}>
                          <span style={{ width: 140, color: '#f8fafc', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {item.feature}
                          </span>
                          <div style={{ flex: 1, background: '#1e293b', height: 8, borderRadius: 4, overflow: 'hidden' }}>
                            <div style={{
                              width: `${Math.min(item.importance * 100, 100)}%`,
                              background: '#0ea5e9',
                              height: '100%',
                              borderRadius: 4
                            }} />
                          </div>
                          <span style={{ width: 55, color: '#38bdf8', fontWeight: 700, textAlign: 'right' }}>
                            {(item.importance * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Interactive Live Inference & Prediction Sandbox */}
                <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                  <div style={{ marginBottom: 12 }}>
                    <h4 style={{ margin: 0, fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                      Interactive Model Inference & Prediction Sandbox
                    </h4>
                    <p style={{ margin: 0, fontSize: 11, color: '#94a3b8' }}>
                      Test live prediction against the persisted model artifact pipeline in real-time.
                    </p>
                  </div>

                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                    gap: 10,
                    marginBottom: 14
                  }}>
                    {(trainingResult.features || []).map(feat => (
                      <div key={feat}>
                        <label style={{ display: 'block', fontSize: 10, fontWeight: 700, color: '#94a3b8', marginBottom: 4 }}>
                          {feat}
                        </label>
                        <input
                          type="text"
                          value={interactivePredictionInput[feat] !== undefined ? interactivePredictionInput[feat] : ''}
                          onChange={(e) => setInteractivePredictionInput({
                            ...interactivePredictionInput,
                            [feat]: e.target.value
                          })}
                          placeholder="value"
                          style={{
                            width: '100%',
                            background: '#1e293b',
                            color: '#f8fafc',
                            border: '1px solid #334155',
                            borderRadius: 4,
                            padding: '6px 8px',
                            fontSize: 11
                          }}
                        />
                      </div>
                    ))}
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <button
                      onClick={handleRunInteractivePrediction}
                      disabled={isPredicting}
                      style={{
                        background: '#0ea5e9',
                        color: '#ffffff',
                        border: 'none',
                        borderRadius: 6,
                        padding: '8px 16px',
                        fontSize: 12,
                        fontWeight: 700,
                        cursor: isPredicting ? 'not-allowed' : 'pointer'
                      }}
                    >
                      {isPredicting ? 'Evaluating Inference...' : 'Run Live Inference'}
                    </button>

                    {interactivePredictionResult && (
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        background: '#064e3b',
                        padding: '6px 12px',
                        borderRadius: 6,
                        border: '1px solid #059669',
                        fontSize: 12,
                        color: '#34d399',
                        fontWeight: 700
                      }}>
                        <span>Predicted {interactivePredictionResult.target}:</span>
                        <strong style={{ fontSize: 14, color: '#ffffff' }}>
                          {interactivePredictionResult.predictions?.[0]}
                        </strong>
                        <span style={{ fontSize: 10, color: '#a7f3d0' }}>
                          ({interactivePredictionResult.prediction_time_ms?.toFixed(1)}ms)
                        </span>
                      </div>
                    )}

                    {predictionError && (
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                        background: '#450a0a',
                        padding: '6px 12px',
                        borderRadius: 6,
                        border: '1px solid #ef4444',
                        fontSize: 11,
                        color: '#fca5a5',
                        fontWeight: 600
                      }}>
                        <AlertCircle size={14} color="#ef4444" />
                        <span>{predictionError}</span>
                      </div>
                    )}
                  </div>
                </div>

              </div>
            )}

            {!trainingResult && (
              <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 30, textAlign: 'center', color: '#64748b' }}>
                <Sparkles size={36} style={{ margin: '0 auto 10px', opacity: 0.5 }} />
                <p style={{ margin: 0, fontSize: 13 }}>
                  Select a target column and click <strong>"Train Model"</strong> above to execute a real supervised machine learning pipeline.
                </p>
              </div>
            )}

          </div>
        )}

        {/* ======================================================== */}
        {/* TAB 5: PHASE 5 REAL UNSUPERVISED LEARNING STUDIO         */}
        {/* ======================================================== */}
        {activeTab === 'unsupervised' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Top Subtab Switcher: Clustering vs PCA */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              background: '#0f172a',
              border: '1px solid #1e293b',
              borderRadius: 10,
              padding: '10px 16px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <button
                  onClick={() => setUnsupSubTab('clustering')}
                  style={{
                    background: unsupSubTab === 'clustering' ? '#0ea5e9' : '#1e293b',
                    color: '#ffffff',
                    border: '1px solid #334155',
                    borderRadius: 6,
                    padding: '8px 16px',
                    fontWeight: 700,
                    fontSize: 12,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6
                  }}
                >
                  <Grid size={14} />
                  <span>🎯 K-Means Clustering</span>
                </button>

                <button
                  onClick={() => setUnsupSubTab('pca')}
                  style={{
                    background: unsupSubTab === 'pca' ? '#0ea5e9' : '#1e293b',
                    color: '#ffffff',
                    border: '1px solid #334155',
                    borderRadius: 6,
                    padding: '8px 16px',
                    fontWeight: 700,
                    fontSize: 12,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6
                  }}
                >
                  <Cpu size={14} />
                  <span>📐 PCA Dimensionality Reduction</span>
                </button>
              </div>

              {unsupSubTab === 'clustering' && lastClusterAuditEvent && (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  background: '#064e3b',
                  color: '#34d399',
                  padding: '4px 10px',
                  borderRadius: 14,
                  fontSize: 11,
                  fontWeight: 700,
                  border: '1px solid #059669'
                }}>
                  <Activity size={12} />
                  <span>Audit: {lastClusterAuditEvent.event_id}</span>
                  <span style={{ color: '#a7f3d0' }}>({(lastClusterAuditEvent.execution_time_ms || 0).toFixed(1)}ms)</span>
                </div>
              )}

              {unsupSubTab === 'pca' && lastPcaAuditEvent && (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  background: '#064e3b',
                  color: '#34d399',
                  padding: '4px 10px',
                  borderRadius: 14,
                  fontSize: 11,
                  fontWeight: 700,
                  border: '1px solid #059669'
                }}>
                  <Activity size={12} />
                  <span>Audit: {lastPcaAuditEvent.event_id}</span>
                  <span style={{ color: '#a7f3d0' }}>({(lastPcaAuditEvent.execution_time_ms || 0).toFixed(1)}ms)</span>
                </div>
              )}
            </div>

            {/* ---------------------------------------------------- */}
            {/* SUBTAB 1: K-MEANS CLUSTERING                         */}
            {/* ---------------------------------------------------- */}
            {unsupSubTab === 'clustering' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {/* Configuration Card */}
                <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                  <div style={{ marginBottom: 14 }}>
                    <h3 style={{ margin: 0, fontSize: 14, fontWeight: 800, color: '#f8fafc' }}>
                      K-Means Clustering Configuration
                    </h3>
                    <p style={{ margin: '2px 0 0', fontSize: 11, color: '#94a3b8' }}>
                      Partitions numerical observations into k distinct Voronoi cells using Euclidean distance and inertia minimization.
                    </p>
                  </div>

                  {/* Feature Selection */}
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                      <label style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8' }}>
                        Numerical Features ({clusterFeatures.length} selected) *
                      </label>
                      <div style={{ display: 'flex', gap: 8 }}>
                        <button
                          onClick={() => setClusterFeatures([...numericColumns])}
                          style={{ background: 'transparent', border: 'none', color: '#38bdf8', fontSize: 11, cursor: 'pointer', fontWeight: 600 }}
                        >
                          Select All ({numericColumns.length})
                        </button>
                        <span style={{ color: '#475569' }}>|</span>
                        <button
                          onClick={() => setClusterFeatures([])}
                          style={{ background: 'transparent', border: 'none', color: '#94a3b8', fontSize: 11, cursor: 'pointer', fontWeight: 600 }}
                        >
                          Clear
                        </button>
                      </div>
                    </div>

                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, maxHeight: 110, overflowY: 'auto', padding: 4, background: '#090d16', borderRadius: 6, border: '1px solid #1e293b' }}>
                      {numericColumns.map(col => {
                        const isSelected = clusterFeatures.includes(col);
                        return (
                          <button
                            key={col}
                            onClick={() => {
                              if (isSelected) {
                                setClusterFeatures(clusterFeatures.filter(c => c !== col));
                              } else {
                                setClusterFeatures([...clusterFeatures, col]);
                              }
                            }}
                            style={{
                              background: isSelected ? '#0284c7' : '#1e293b',
                              color: isSelected ? '#ffffff' : '#94a3b8',
                              border: isSelected ? '1px solid #38bdf8' : '1px solid #334155',
                              borderRadius: 4,
                              padding: '4px 8px',
                              fontSize: 11,
                              fontWeight: 600,
                              cursor: 'pointer'
                            }}
                          >
                            {col}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Controls: k slider & scaling dropdown */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14, marginBottom: 16 }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                        <label style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8' }}>Number of Clusters (k)</label>
                        <span style={{ background: '#0ea5e9', color: '#fff', padding: '1px 6px', borderRadius: 4, fontSize: 11, fontWeight: 700 }}>
                          k = {clusterK}
                        </span>
                      </div>
                      <input
                        type="range"
                        min="2"
                        max={Math.min(Math.max((rowCount || 10) - 1, 2), 12)}
                        value={clusterK}
                        onChange={(e) => setClusterK(parseInt(e.target.value, 10))}
                        style={{ width: '100%', accentColor: '#0ea5e9' }}
                      />
                    </div>

                    <div>
                      <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#94a3b8', marginBottom: 6 }}>
                        Feature Preprocessing
                      </label>
                      <select
                        value={clusterScaling}
                        onChange={(e) => setClusterScaling(e.target.value)}
                        style={{
                          width: '100%',
                          background: '#1e293b',
                          color: '#f8fafc',
                          border: '1px solid #334155',
                          borderRadius: 6,
                          padding: '7px 10px',
                          fontSize: 12,
                          fontWeight: 600
                        }}
                      >
                        <option value="standard">StandardScaler (Zero mean, unit variance — Recommended)</option>
                        <option value="none">None (Raw numerical scale)</option>
                      </select>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <button
                      onClick={handleRunKMeans}
                      disabled={isClustering}
                      style={{
                        background: !isClustering ? '#10b981' : '#1e293b',
                        color: !isClustering ? '#ffffff' : '#64748b',
                        border: 'none',
                        borderRadius: 6,
                        padding: '10px 18px',
                        fontSize: 12,
                        fontWeight: 700,
                        cursor: clusterFeatures.length > 0 && !isClustering ? 'pointer' : 'not-allowed',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8
                      }}
                    >
                      <Play size={14} />
                      <span>{isClustering ? 'Fitting K-Means...' : 'Train K-Means Clusters'}</span>
                    </button>

                    <button
                      onClick={handleComputeElbow}
                      disabled={clusterFeatures.length === 0 || isComputingElbow}
                      style={{
                        background: '#1e293b',
                        color: clusterFeatures.length > 0 && !isComputingElbow ? '#38bdf8' : '#64748b',
                        border: '1px solid #334155',
                        borderRadius: 6,
                        padding: '10px 16px',
                        fontSize: 12,
                        fontWeight: 700,
                        cursor: clusterFeatures.length > 0 && !isComputingElbow ? 'pointer' : 'not-allowed',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8
                      }}
                    >
                      <TrendingUp size={14} />
                      <span>{isComputingElbow ? 'Evaluating Elbow...' : 'Evaluate Elbow Curve (k=2..8)'}</span>
                    </button>
                  </div>
                </div>

                {/* Error Banner */}
                {clusterError && (
                  <div style={{
                    background: '#450a0a',
                    border: '1px solid #ef4444',
                    borderRadius: 8,
                    padding: '12px 16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    color: '#fca5a5',
                    fontSize: 12,
                    fontWeight: 600
                  }}>
                    <AlertCircle size={18} color="#ef4444" />
                    <span>{clusterError}</span>
                  </div>
                )}

                {/* Elbow Curve Panel */}
                {elbowCurve && elbowCurve.length > 0 && (
                  <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                      <div>
                        <h4 style={{ margin: 0, fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                          Elbow Curve & Silhouette Diagnostic Evaluation
                        </h4>
                        <p style={{ margin: '2px 0 0', fontSize: 11, color: '#94a3b8' }}>
                          Compares Within-Cluster Sum of Squares (Inertia) vs Silhouette separation across candidate k.
                        </p>
                      </div>
                      <button
                        onClick={() => setElbowCurve(null)}
                        style={{ background: 'transparent', border: 'none', color: '#64748b', fontSize: 11, cursor: 'pointer' }}
                      >
                        Dismiss
                      </button>
                    </div>

                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                        <thead>
                          <tr style={{ background: '#1e293b', color: '#94a3b8', textAlign: 'left' }}>
                            <th style={{ padding: '8px 12px' }}>Clusters (k)</th>
                            <th style={{ padding: '8px 12px' }}>Inertia (WCSS)</th>
                            <th style={{ padding: '8px 12px' }}>Silhouette Score</th>
                            <th style={{ padding: '8px 12px' }}>Separation Quality</th>
                            <th style={{ padding: '8px 12px', textAlign: 'right' }}>Action</th>
                          </tr>
                        </thead>
                        <tbody>
                          {elbowCurve.map((row) => {
                            const sil = row.silhouette_score;
                            const qualityBadge = sil > 0.5 ? '#10b981' : (sil > 0.25 ? '#38bdf8' : '#f59e0b');
                            const qualityText = sil > 0.5 ? 'Strong Separation' : (sil > 0.25 ? 'Moderate' : 'Weak / Overlapping');
                            return (
                              <tr key={row.k} style={{ borderBottom: '1px solid #1e293b' }}>
                                <td style={{ padding: '8px 12px', fontWeight: 700, color: '#f8fafc' }}>k = {row.k}</td>
                                <td style={{ padding: '8px 12px', color: '#cbd5e1' }}>{row.inertia?.toFixed(2)}</td>
                                <td style={{ padding: '8px 12px', fontWeight: 700, color: qualityBadge }}>{sil?.toFixed(3)}</td>
                                <td style={{ padding: '8px 12px' }}>
                                  <span style={{ fontSize: 10, fontWeight: 700, color: qualityBadge, background: `${qualityBadge}22`, padding: '2px 8px', borderRadius: 10 }}>
                                    {qualityText}
                                  </span>
                                </td>
                                <td style={{ padding: '8px 12px', textAlign: 'right' }}>
                                  <button
                                    onClick={() => setClusterK(row.k)}
                                    style={{
                                      background: clusterK === row.k ? '#0ea5e9' : '#1e293b',
                                      color: '#ffffff',
                                      border: '1px solid #334155',
                                      borderRadius: 4,
                                      padding: '3px 8px',
                                      fontSize: 11,
                                      cursor: 'pointer',
                                      fontWeight: 600
                                    }}
                                  >
                                    {clusterK === row.k ? 'Selected' : 'Use k=' + row.k}
                                  </button>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Cluster Results */}
                {clusterResult && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    {/* Analytical Finding Box */}
                    {clusterResult.finding && (
                      <div style={{
                        background: '#0c1a2e',
                        borderLeft: '4px solid #0ea5e9',
                        borderRadius: 8,
                        padding: '12px 16px',
                        color: '#bae6fd',
                        fontSize: 12,
                        lineHeight: 1.5
                      }}>
                        <strong>Analytical Summary:</strong> {clusterResult.finding}
                      </div>
                    )}

                    {/* KPI Cards Row */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Clusters Formed</span>
                        <h4 style={{ margin: '4px 0 0', fontSize: 20, fontWeight: 800, color: '#38bdf8' }}>
                          k = {clusterResult.n_clusters}
                        </h4>
                        <span style={{ fontSize: 10, color: '#64748b' }}>Across {clusterResult.selected_features?.length} features</span>
                      </div>

                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Inertia (WCSS)</span>
                        <h4 style={{ margin: '4px 0 0', fontSize: 20, fontWeight: 800, color: '#f8fafc' }}>
                          {clusterResult.inertia?.toFixed(2)}
                        </h4>
                        <span style={{ fontSize: 10, color: '#64748b' }}>Within-cluster sum of squares</span>
                      </div>

                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Silhouette Score</span>
                        <h4 style={{
                          margin: '4px 0 0',
                          fontSize: 20,
                          fontWeight: 800,
                          color: (clusterResult.silhouette_score || 0) > 0.4 ? '#10b981' : ((clusterResult.silhouette_score || 0) > 0.2 ? '#38bdf8' : '#f59e0b')
                        }}>
                          {clusterResult.silhouette_score !== null ? clusterResult.silhouette_score?.toFixed(3) : 'N/A'}
                        </h4>
                        <span style={{ fontSize: 10, color: '#64748b' }}>Range: [-1.0 to +1.0]</span>
                      </div>

                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Preprocessing</span>
                        <h4 style={{ margin: '4px 0 0', fontSize: 16, fontWeight: 800, color: '#a78bfa' }}>
                          {clusterResult.scaling === 'standard' ? 'StandardScaler' : 'Raw Metrics'}
                        </h4>
                        <span style={{ fontSize: 10, color: '#64748b' }}>Preserves active dataset</span>
                      </div>
                    </div>

                    {/* Cluster Sizes Breakdown */}
                    {clusterResult.cluster_sizes && (
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 16 }}>
                        <h4 style={{ margin: '0 0 10px 0', fontSize: 12, fontWeight: 700, color: '#94a3b8' }}>
                          Cluster Membership Distribution
                        </h4>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                          {Object.entries(clusterResult.cluster_sizes).map(([label, count]) => {
                            const total = Object.values(clusterResult.cluster_sizes).reduce((a, b) => a + b, 0);
                            const pct = total > 0 ? ((count / total) * 100).toFixed(1) : 0;
                            return (
                              <div key={label} style={{
                                background: '#1e293b',
                                border: '1px solid #334155',
                                borderRadius: 6,
                                padding: '8px 14px',
                                minWidth: 120
                              }}>
                                <span style={{ fontSize: 11, color: '#38bdf8', fontWeight: 700 }}>Cluster {label}</span>
                                <div style={{ fontSize: 16, fontWeight: 800, color: '#f8fafc', marginTop: 2 }}>
                                  {count} <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 500 }}>({pct}%)</span>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* Scatter Plot Image from Backend */}
                    {clusterResult.plot_image_base64 && (
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                        <h4 style={{ margin: '0 0 12px 0', fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                          Cluster Spatial Distribution Visualization
                        </h4>
                        <div style={{ display: 'flex', justifyContent: 'center', background: '#090d16', padding: 12, borderRadius: 8 }}>
                          <img
                            src={`data:image/png;base64,${clusterResult.plot_image_base64}`}
                            alt="K-Means Cluster Scatter Plot"
                            style={{ maxWidth: '100%', maxHeight: 420, objectFit: 'contain', borderRadius: 6 }}
                          />
                        </div>
                      </div>
                    )}

                    {/* Centroids Table */}
                    {clusterResult.centroids && clusterResult.centroids.length > 0 && (
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                          Cluster Centroid Coordinates (Feature Means)
                        </h4>
                        <div style={{ overflowX: 'auto' }}>
                          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                            <thead>
                              <tr style={{ background: '#1e293b', color: '#94a3b8', textAlign: 'left' }}>
                                <th style={{ padding: '8px 10px' }}>Cluster</th>
                                {clusterResult.selected_features.map(f => (
                                  <th key={f} style={{ padding: '8px 10px' }}>{f}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {clusterResult.centroids.map((row, idx) => (
                                <tr key={idx} style={{ borderBottom: '1px solid #1e293b' }}>
                                  <td style={{ padding: '8px 10px', fontWeight: 700, color: '#38bdf8' }}>
                                    Cluster {row.cluster_label !== undefined ? row.cluster_label : idx}
                                  </td>
                                  {clusterResult.selected_features.map(f => (
                                    <td key={f} style={{ padding: '8px 10px', color: '#cbd5e1' }}>
                                      {typeof row[f] === 'number' ? row[f].toFixed(4) : (row[f] || '-')}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {/* Reproducible Python Pipeline Script */}
                    {clusterResult.reproducible_code && (
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 16 }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                          <span style={{ fontSize: 12, fontWeight: 700, color: '#94a3b8' }}>Reproducible Python Pipeline</span>
                          <button
                            onClick={() => handleCopyCode(clusterResult.reproducible_code, 'kmeans')}
                            style={{
                              background: '#1e293b',
                              color: copiedCodeType === 'kmeans' ? '#34d399' : '#94a3b8',
                              border: '1px solid #334155',
                              borderRadius: 4,
                              padding: '4px 10px',
                              fontSize: 11,
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: 6
                            }}
                          >
                            {copiedCodeType === 'kmeans' ? <Check size={12} /> : <Copy size={12} />}
                            <span>{copiedCodeType === 'kmeans' ? 'Copied!' : 'Copy Code'}</span>
                          </button>
                        </div>
                        <pre style={{
                          background: '#090d16',
                          color: '#38bdf8',
                          padding: 12,
                          borderRadius: 6,
                          fontSize: 11,
                          fontFamily: 'monospace',
                          overflowX: 'auto',
                          margin: 0
                        }}>
                          {clusterResult.reproducible_code}
                        </pre>
                      </div>
                    )}
                  </div>
                )}

                {/* Empty State */}
                {!clusterResult && !isClustering && (
                  <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 36, textAlign: 'center', color: '#64748b' }}>
                    <Grid size={40} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
                    <h4 style={{ margin: '0 0 6px 0', fontSize: 14, color: '#f8fafc' }}>K-Means Clustering Ready</h4>
                    <p style={{ margin: 0, fontSize: 12, maxWidth: 450, marginInline: 'auto' }}>
                      Select numerical features and cluster count above, then click <strong>"Train K-Means Clusters"</strong> to evaluate real centroids, inertia, and silhouette separation.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* ---------------------------------------------------- */}
            {/* SUBTAB 2: PCA DIMENSIONALITY REDUCTION               */}
            {/* ---------------------------------------------------- */}
            {unsupSubTab === 'pca' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {/* Configuration Card */}
                <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                  <div style={{ marginBottom: 14 }}>
                    <h3 style={{ margin: 0, fontSize: 14, fontWeight: 800, color: '#f8fafc' }}>
                      Principal Component Analysis (PCA) Configuration
                    </h3>
                    <p style={{ margin: '2px 0 0', fontSize: 11, color: '#94a3b8' }}>
                      Performs orthogonal linear transformation via Singular Value Decomposition (SVD) to maximize variance along uncorrelated components.
                    </p>
                  </div>

                  {/* Feature Selection */}
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                      <label style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8' }}>
                        Numerical Features ({pcaFeatures.length} selected, min 2) *
                      </label>
                      <div style={{ display: 'flex', gap: 8 }}>
                        <button
                          onClick={() => setPcaFeatures([...numericColumns])}
                          style={{ background: 'transparent', border: 'none', color: '#38bdf8', fontSize: 11, cursor: 'pointer', fontWeight: 600 }}
                        >
                          Select All ({numericColumns.length})
                        </button>
                        <span style={{ color: '#475569' }}>|</span>
                        <button
                          onClick={() => setPcaFeatures([])}
                          style={{ background: 'transparent', border: 'none', color: '#94a3b8', fontSize: 11, cursor: 'pointer', fontWeight: 600 }}
                        >
                          Clear
                        </button>
                      </div>
                    </div>

                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, maxHeight: 110, overflowY: 'auto', padding: 4, background: '#090d16', borderRadius: 6, border: '1px solid #1e293b' }}>
                      {numericColumns.map(col => {
                        const isSelected = pcaFeatures.includes(col);
                        return (
                          <button
                            key={col}
                            onClick={() => {
                              if (isSelected) {
                                setPcaFeatures(pcaFeatures.filter(c => c !== col));
                              } else {
                                setPcaFeatures([...pcaFeatures, col]);
                              }
                            }}
                            style={{
                              background: isSelected ? '#0284c7' : '#1e293b',
                              color: isSelected ? '#ffffff' : '#94a3b8',
                              border: isSelected ? '1px solid #38bdf8' : '1px solid #334155',
                              borderRadius: 4,
                              padding: '4px 8px',
                              fontSize: 11,
                              fontWeight: 600,
                              cursor: 'pointer'
                            }}
                          >
                            {col}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Controls: n_components & scaling */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14, marginBottom: 16 }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                        <label style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8' }}>Principal Components (n)</label>
                        <span style={{ background: '#0ea5e9', color: '#fff', padding: '1px 6px', borderRadius: 4, fontSize: 11, fontWeight: 700 }}>
                          n = {pcaComponents}
                        </span>
                      </div>
                      <input
                        type="range"
                        min="1"
                        max={Math.max(Math.min(pcaFeatures.length, 10), 1)}
                        value={pcaComponents}
                        onChange={(e) => setPcaComponents(parseInt(e.target.value, 10))}
                        disabled={pcaFeatures.length < 2}
                        style={{ width: '100%', accentColor: '#0ea5e9' }}
                      />
                    </div>

                    <div>
                      <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#94a3b8', marginBottom: 6 }}>
                        Preprocessing / Scaling
                      </label>
                      <select
                        value={pcaScaling}
                        onChange={(e) => setPcaScaling(e.target.value)}
                        style={{
                          width: '100%',
                          background: '#1e293b',
                          color: '#f8fafc',
                          border: '1px solid #334155',
                          borderRadius: 6,
                          padding: '7px 10px',
                          fontSize: 12,
                          fontWeight: 600
                        }}
                      >
                        <option value="standard">StandardScaler (Zero mean, unit variance — Recommended)</option>
                        <option value="none">None (Covariance PCA without unit variance)</option>
                      </select>
                    </div>
                  </div>

                  {/* Action Button */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <button
                      onClick={handleRunPCA}
                      disabled={isPca}
                      style={{
                        background: !isPca ? '#0ea5e9' : '#1e293b',
                        color: !isPca ? '#ffffff' : '#64748b',
                        border: 'none',
                        borderRadius: 6,
                        padding: '10px 18px',
                        fontSize: 12,
                        fontWeight: 700,
                        cursor: pcaFeatures.length >= 2 && !isPca ? 'pointer' : 'not-allowed',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8
                      }}
                    >
                      <Play size={14} />
                      <span>{isPca ? 'Computing PCA Decomposition...' : 'Execute PCA Decomposition'}</span>
                    </button>
                    {pcaFeatures.length < 2 && (
                      <span style={{ fontSize: 11, color: '#94a3b8' }}>Select at least 2 features to run PCA.</span>
                    )}
                  </div>
                </div>

                {/* Error Banner */}
                {pcaError && (
                  <div style={{
                    background: '#450a0a',
                    border: '1px solid #ef4444',
                    borderRadius: 8,
                    padding: '12px 16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    color: '#fca5a5',
                    fontSize: 12,
                    fontWeight: 600
                  }}>
                    <AlertCircle size={18} color="#ef4444" />
                    <span>{pcaError}</span>
                  </div>
                )}

                {/* PCA Results */}
                {pcaResult && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    {/* Analytical Finding Box */}
                    {pcaResult.finding && (
                      <div style={{
                        background: '#0c1a2e',
                        borderLeft: '4px solid #0ea5e9',
                        borderRadius: 8,
                        padding: '12px 16px',
                        color: '#bae6fd',
                        fontSize: 12,
                        lineHeight: 1.5
                      }}>
                        <strong>Analytical Summary:</strong> {pcaResult.finding}
                      </div>
                    )}

                    {/* KPI Cards Row */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Components Retained</span>
                        <h4 style={{ margin: '4px 0 0', fontSize: 20, fontWeight: 800, color: '#38bdf8' }}>
                          n = {pcaResult.n_components}
                        </h4>
                        <span style={{ fontSize: 10, color: '#64748b' }}>From {pcaResult.selected_features?.length} features</span>
                      </div>

                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Cumulative Explained Variance</span>
                        <h4 style={{ margin: '4px 0 0', fontSize: 20, fontWeight: 800, color: '#10b981' }}>
                          {pcaResult.cumulative_variance_ratio && pcaResult.cumulative_variance_ratio.length > 0
                            ? (pcaResult.cumulative_variance_ratio[pcaResult.cumulative_variance_ratio.length - 1] * 100).toFixed(1) + '%'
                            : 'N/A'}
                        </h4>
                        <span style={{ fontSize: 10, color: '#64748b' }}>Total information captured</span>
                      </div>

                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>PC1 Variance Ratio</span>
                        <h4 style={{ margin: '4px 0 0', fontSize: 20, fontWeight: 800, color: '#f8fafc' }}>
                          {pcaResult.explained_variance_ratio && pcaResult.explained_variance_ratio.length > 0
                            ? (pcaResult.explained_variance_ratio[0] * 100).toFixed(1) + '%'
                            : 'N/A'}
                        </h4>
                        <span style={{ fontSize: 10, color: '#64748b' }}>Dominant orthogonal axis</span>
                      </div>

                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Preprocessing</span>
                        <h4 style={{ margin: '4px 0 0', fontSize: 16, fontWeight: 800, color: '#a78bfa' }}>
                          {pcaResult.scaling === 'standard' ? 'StandardScaler' : 'Raw Covariance'}
                        </h4>
                        <span style={{ fontSize: 10, color: '#64748b' }}>Zero dataset mutation</span>
                      </div>
                    </div>

                    {/* Scree Plot Visualization */}
                    {pcaResult.plot_image_base64 && (
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                        <h4 style={{ margin: '0 0 12px 0', fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                          PCA Scree Plot & Cumulative Variance Curve
                        </h4>
                        <div style={{ display: 'flex', justifyContent: 'center', background: '#090d16', padding: 12, borderRadius: 8 }}>
                          <img
                            src={`data:image/png;base64,${pcaResult.plot_image_base64}`}
                            alt="PCA Scree Plot"
                            style={{ maxWidth: '100%', maxHeight: 420, objectFit: 'contain', borderRadius: 6 }}
                          />
                        </div>
                      </div>
                    )}

                    {/* Component Variance Breakdown Table */}
                    {pcaResult.component_summary && (
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                          Principal Component Variance Breakdown
                        </h4>
                        <div style={{ overflowX: 'auto' }}>
                          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                            <thead>
                              <tr style={{ background: '#1e293b', color: '#94a3b8', textAlign: 'left' }}>
                                <th style={{ padding: '8px 12px' }}>Component</th>
                                <th style={{ padding: '8px 12px' }}>Individual Variance Ratio</th>
                                <th style={{ padding: '8px 12px' }}>Cumulative Variance Ratio</th>
                                <th style={{ padding: '8px 12px' }}>Raw Variance (Eigenvalue)</th>
                              </tr>
                            </thead>
                            <tbody>
                              {pcaResult.component_summary.map((row, idx) => {
                                const indPct = (row.explained_variance_ratio * 100).toFixed(2);
                                const cumPct = (row.cumulative_variance_ratio * 100).toFixed(2);
                                const rawVar = pcaResult.explained_variance?.[idx]?.toFixed(4) || '-';
                                return (
                                  <tr key={idx} style={{ borderBottom: '1px solid #1e293b' }}>
                                    <td style={{ padding: '8px 12px', fontWeight: 700, color: '#38bdf8' }}>{row.component}</td>
                                    <td style={{ padding: '8px 12px', color: '#cbd5e1' }}>{indPct}%</td>
                                    <td style={{ padding: '8px 12px', fontWeight: 700, color: '#10b981' }}>{cumPct}%</td>
                                    <td style={{ padding: '8px 12px', color: '#94a3b8' }}>{rawVar}</td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {/* Feature Loadings Matrix (Eigenvectors) */}
                    {pcaResult.feature_loadings && pcaResult.feature_loadings.length > 0 && (
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                        <h4 style={{ margin: '0 0 6px 0', fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                          Feature Loadings Matrix (Eigenvectors)
                        </h4>
                        <p style={{ margin: '0 0 10px 0', fontSize: 11, color: '#94a3b8' }}>
                          Weights indicating the direction and magnitude of feature contribution to each principal axis.
                        </p>
                        <div style={{ overflowX: 'auto' }}>
                          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                            <thead>
                              <tr style={{ background: '#1e293b', color: '#94a3b8', textAlign: 'left' }}>
                                <th style={{ padding: '8px 10px' }}>Component</th>
                                {pcaResult.selected_features.map(f => (
                                  <th key={f} style={{ padding: '8px 10px' }}>{f}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {pcaResult.feature_loadings.map((row, idx) => (
                                <tr key={idx} style={{ borderBottom: '1px solid #1e293b' }}>
                                  <td style={{ padding: '8px 10px', fontWeight: 700, color: '#38bdf8' }}>{row.component}</td>
                                  {pcaResult.selected_features.map(f => {
                                    const val = row[f];
                                    const isPos = typeof val === 'number' && val >= 0;
                                    return (
                                      <td key={f} style={{ padding: '8px 10px', color: isPos ? '#34d399' : '#f87171', fontWeight: 600 }}>
                                        {typeof val === 'number' ? val.toFixed(4) : (val || '-')}
                                      </td>
                                    );
                                  })}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {/* Sample Projected Coordinates Preview */}
                    {pcaResult.sample_coordinates && pcaResult.sample_coordinates.length > 0 && (
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                          Sample Projected Coordinates Preview (First 15 Rows)
                        </h4>
                        <div style={{ overflowX: 'auto' }}>
                          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                            <thead>
                              <tr style={{ background: '#1e293b', color: '#94a3b8', textAlign: 'left' }}>
                                <th style={{ padding: '8px 10px' }}>Row Index</th>
                                {Object.keys(pcaResult.sample_coordinates[0] || {}).map(pc => (
                                  <th key={pc} style={{ padding: '8px 10px' }}>{pc}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {pcaResult.sample_coordinates.slice(0, 15).map((row, idx) => (
                                <tr key={idx} style={{ borderBottom: '1px solid #1e293b' }}>
                                  <td style={{ padding: '8px 10px', color: '#64748b' }}>#{idx + 1}</td>
                                  {Object.keys(row).map(pc => (
                                    <td key={pc} style={{ padding: '8px 10px', color: '#cbd5e1' }}>
                                      {typeof row[pc] === 'number' ? row[pc].toFixed(4) : row[pc]}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {/* Reproducible Python Pipeline Script */}
                    {pcaResult.reproducible_code && (
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 16 }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                          <span style={{ fontSize: 12, fontWeight: 700, color: '#94a3b8' }}>Reproducible Python Pipeline</span>
                          <button
                            onClick={() => handleCopyCode(pcaResult.reproducible_code, 'pca')}
                            style={{
                              background: '#1e293b',
                              color: copiedCodeType === 'pca' ? '#34d399' : '#94a3b8',
                              border: '1px solid #334155',
                              borderRadius: 4,
                              padding: '4px 10px',
                              fontSize: 11,
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: 6
                            }}
                          >
                            {copiedCodeType === 'pca' ? <Check size={12} /> : <Copy size={12} />}
                            <span>{copiedCodeType === 'pca' ? 'Copied!' : 'Copy Code'}</span>
                          </button>
                        </div>
                        <pre style={{
                          background: '#090d16',
                          color: '#38bdf8',
                          padding: 12,
                          borderRadius: 6,
                          fontSize: 11,
                          fontFamily: 'monospace',
                          overflowX: 'auto',
                          margin: 0
                        }}>
                          {pcaResult.reproducible_code}
                        </pre>
                      </div>
                    )}
                  </div>
                )}

                {/* Empty State */}
                {!pcaResult && !isPca && (
                  <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 36, textAlign: 'center', color: '#64748b' }}>
                    <Cpu size={40} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
                    <h4 style={{ margin: '0 0 6px 0', fontSize: 14, color: '#f8fafc' }}>PCA Dimensionality Reduction Ready</h4>
                    <p style={{ margin: 0, fontSize: 12, maxWidth: 450, marginInline: 'auto' }}>
                      Select at least 2 numerical features above, then click <strong>"Execute PCA Decomposition"</strong> to compute scree variance ratios, feature loadings, and orthogonal principal components.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ======================================================== */}
        {/* TAB 6: Time-Series & Forecasting Studio (Phase 6)        */}
        {/* ======================================================== */}
        {activeTab === 'timeseries' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Top Subtab Selector + Audit Badge */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={() => setTsSubTab('forecast')}
                  style={{
                    background: tsSubTab === 'forecast' ? '#0ea5e9' : '#1e293b',
                    color: '#ffffff',
                    border: '1px solid #334155',
                    borderRadius: 6,
                    padding: '8px 16px',
                    fontWeight: 700,
                    fontSize: 12,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6
                  }}
                >
                  <TrendingUp size={14} />
                  <span>⏱️ Forecasting (ARIMA / SARIMAX / AutoReg / Naive)</span>
                </button>

                <button
                  onClick={() => setTsSubTab('diagnostics')}
                  style={{
                    background: tsSubTab === 'diagnostics' ? '#0ea5e9' : '#1e293b',
                    color: '#ffffff',
                    border: '1px solid #334155',
                    borderRadius: 6,
                    padding: '8px 16px',
                    fontWeight: 700,
                    fontSize: 12,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6
                  }}
                >
                  <Activity size={14} />
                  <span>📊 Statistical Diagnostics (ADF, KPSS, ACF, PACF, Decomposition)</span>
                </button>
              </div>

              {lastForecastAuditEvent && (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  background: '#064e3b',
                  color: '#34d399',
                  padding: '4px 10px',
                  borderRadius: 14,
                  fontSize: 11,
                  fontWeight: 700,
                  border: '1px solid #059669'
                }}>
                  <Activity size={12} />
                  <span>Audit: {lastForecastAuditEvent.event_id}</span>
                  <span style={{ color: '#a7f3d0' }}>({(lastForecastAuditEvent.execution_time_ms || 0).toFixed(1)}ms)</span>
                </div>
              )}
            </div>

            {/* Time-Series Setup & Column Grounding Card */}
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
              <div style={{ marginBottom: 14 }}>
                <h3 style={{ margin: 0, fontSize: 14, fontWeight: 800, color: '#f8fafc' }}>
                  Temporal Index & Target Selection
                </h3>
                <p style={{ margin: '2px 0 0', fontSize: 11, color: '#94a3b8' }}>
                  Explicit schema grounding: Select a temporal column for calendar indexing and a continuous numerical target to analyze and forecast.
                </p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14, marginBottom: 14 }}>
                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#94a3b8', marginBottom: 6 }}>
                    Temporal / Datetime Column *
                  </label>
                  <select
                    value={tsTimeCol}
                    onChange={(e) => setTsTimeCol(e.target.value)}
                    style={{
                      width: '100%',
                      background: '#1e293b',
                      color: '#f8fafc',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 10px',
                      fontSize: 12,
                      fontWeight: 600
                    }}
                  >
                    <option value="">-- Select Temporal Column --</option>
                    {allColumns.map(col => (
                      <option key={col.name || col} value={col.name || col}>{col.name || col}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#94a3b8', marginBottom: 6 }}>
                    Target Metric Column *
                  </label>
                  <select
                    value={tsTargetCol}
                    onChange={(e) => setTsTargetCol(e.target.value)}
                    style={{
                      width: '100%',
                      background: '#1e293b',
                      color: '#f8fafc',
                      border: '1px solid #334155',
                      borderRadius: 6,
                      padding: '8px 10px',
                      fontSize: 12,
                      fontWeight: 600
                    }}
                  >
                    <option value="">-- Select Numeric Target --</option>
                    {numericColumns.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <button
                  onClick={handleValidateTimeSeries}
                  disabled={isValidatingTs}
                  style={{
                    background: '#1e293b',
                    color: '#38bdf8',
                    border: '1px solid #334155',
                    borderRadius: 6,
                    padding: '8px 14px',
                    fontSize: 12,
                    fontWeight: 700,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6
                  }}
                >
                  <CheckCircle2 size={14} />
                  <span>{isValidatingTs ? 'Validating...' : 'Validate Temporal Series'}</span>
                </button>
              </div>

              {/* Validation Error Banner */}
              {tsValidateError && (
                <div style={{
                  marginTop: 12,
                  background: '#450a0a',
                  border: '1px solid #ef4444',
                  borderRadius: 6,
                  padding: '10px 14px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  color: '#fca5a5',
                  fontSize: 12,
                  fontWeight: 600
                }}>
                  <AlertCircle size={16} color="#ef4444" />
                  <span>{tsValidateError}</span>
                </div>
              )}

              {/* Validation Status Badges */}
              {tsValidation && (
                <div style={{
                  marginTop: 14,
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                  gap: 10,
                  background: '#090d16',
                  padding: 12,
                  borderRadius: 8,
                  border: '1px solid #1e293b'
                }}>
                  <div>
                    <span style={{ fontSize: 10, color: '#64748b' }}>Usable Observations</span>
                    <div style={{ fontSize: 14, fontWeight: 800, color: '#38bdf8' }}>{tsValidation.usable_observations}</div>
                  </div>
                  <div>
                    <span style={{ fontSize: 10, color: '#64748b' }}>Detected Frequency</span>
                    <div style={{ fontSize: 14, fontWeight: 800, color: '#10b981' }}>{tsValidation.detected_frequency || 'Irregular'}</div>
                  </div>
                  <div>
                    <span style={{ fontSize: 10, color: '#64748b' }}>Temporal Ordering</span>
                    <div style={{ fontSize: 14, fontWeight: 800, color: tsValidation.is_chronologically_sorted ? '#10b981' : '#f59e0b' }}>
                      {tsValidation.is_chronologically_sorted ? 'Monotonic Ascending' : 'Unsorted'}
                    </div>
                  </div>
                  <div>
                    <span style={{ fontSize: 10, color: '#64748b' }}>Duplicate Timestamps</span>
                    <div style={{ fontSize: 14, fontWeight: 800, color: tsValidation.has_duplicate_timestamps ? '#ef4444' : '#10b981' }}>
                      {tsValidation.has_duplicate_timestamps ? `${tsValidation.duplicate_timestamp_count} Duplicates` : 'None (Unique)'}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* ---------------------------------------------------- */}
            {/* SUBTAB 1: TIME-SERIES FORECASTING                    */}
            {/* ---------------------------------------------------- */}
            {tsSubTab === 'forecast' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {/* Forecast Configuration Card */}
                <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                  <div style={{ marginBottom: 14 }}>
                    <h3 style={{ margin: 0, fontSize: 14, fontWeight: 800, color: '#f8fafc' }}>
                      Forecasting Model Architecture & Horizon
                    </h3>
                    <p style={{ margin: '2px 0 0', fontSize: 11, color: '#94a3b8' }}>
                      Fits mathematical time-series model and evaluates out-of-sample accuracy on a strict chronological holdout window without future leakage.
                    </p>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14, marginBottom: 14 }}>
                    <div>
                      <label style={{ display: 'block', fontSize: 11, fontWeight: 700, color: '#94a3b8', marginBottom: 6 }}>
                        Forecasting Algorithm
                      </label>
                      <select
                        value={tsModel}
                        onChange={(e) => setTsModel(e.target.value)}
                        style={{
                          width: '100%',
                          background: '#1e293b',
                          color: '#f8fafc',
                          border: '1px solid #334155',
                          borderRadius: 6,
                          padding: '8px 10px',
                          fontSize: 12,
                          fontWeight: 600
                        }}
                      >
                        <option value="arima">ARIMA (p, d, q) — Autoregressive Integrated Moving Average</option>
                        <option value="sarimax">SARIMAX (p,d,q)x(P,D,Q,s) — Seasonal ARIMA with Exogenous Support</option>
                        <option value="autoreg">AutoReg (lags) — Autoregressive Univariate Model</option>
                        <option value="naive">Naive Baseline — Last-Observed Value Persistence</option>
                      </select>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                        <label style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8' }}>Forecast Horizon (Steps)</label>
                        <span style={{ background: '#0ea5e9', color: '#fff', padding: '1px 6px', borderRadius: 4, fontSize: 11, fontWeight: 700 }}>
                          h = {tsHorizon} steps
                        </span>
                      </div>
                      <input
                        type="range"
                        min="1"
                        max="30"
                        value={tsHorizon}
                        onChange={(e) => setTsHorizon(parseInt(e.target.value, 10))}
                        style={{ width: '100%', accentColor: '#0ea5e9' }}
                      />
                    </div>
                  </div>

                  {/* Model Specific Parameter Controls */}
                  {tsModel === 'arima' && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, background: '#090d16', padding: 12, borderRadius: 8, border: '1px solid #1e293b', marginBottom: 14 }}>
                      <div>
                        <label style={{ fontSize: 10, color: '#94a3b8', fontWeight: 600 }}>AR Order (p)</label>
                        <input
                          type="number"
                          min="0"
                          max="5"
                          value={tsParams.p ?? 1}
                          onChange={(e) => setTsParams({ ...tsParams, p: parseInt(e.target.value, 10) })}
                          style={{ width: '100%', background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: 4, padding: '4px 8px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 10, color: '#94a3b8', fontWeight: 600 }}>Differencing (d)</label>
                        <input
                          type="number"
                          min="0"
                          max="2"
                          value={tsParams.d ?? 1}
                          onChange={(e) => setTsParams({ ...tsParams, d: parseInt(e.target.value, 10) })}
                          style={{ width: '100%', background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: 4, padding: '4px 8px', fontSize: 12 }}
                        />
                      </div>
                      <div>
                        <label style={{ fontSize: 10, color: '#94a3b8', fontWeight: 600 }}>MA Order (q)</label>
                        <input
                          type="number"
                          min="0"
                          max="5"
                          value={tsParams.q ?? 1}
                          onChange={(e) => setTsParams({ ...tsParams, q: parseInt(e.target.value, 10) })}
                          style={{ width: '100%', background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: 4, padding: '4px 8px', fontSize: 12 }}
                        />
                      </div>
                    </div>
                  )}

                  {tsModel === 'sarimax' && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))', gap: 10, background: '#090d16', padding: 12, borderRadius: 8, border: '1px solid #1e293b', marginBottom: 14 }}>
                      <div>
                        <label style={{ fontSize: 10, color: '#94a3b8', fontWeight: 600 }}>p</label>
                        <input type="number" min="0" max="3" value={tsParams.p ?? 1} onChange={(e) => setTsParams({ ...tsParams, p: parseInt(e.target.value, 10) })} style={{ width: '100%', background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: 4, padding: '4px 8px', fontSize: 12 }} />
                      </div>
                      <div>
                        <label style={{ fontSize: 10, color: '#94a3b8', fontWeight: 600 }}>d</label>
                        <input type="number" min="0" max="2" value={tsParams.d ?? 1} onChange={(e) => setTsParams({ ...tsParams, d: parseInt(e.target.value, 10) })} style={{ width: '100%', background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: 4, padding: '4px 8px', fontSize: 12 }} />
                      </div>
                      <div>
                        <label style={{ fontSize: 10, color: '#94a3b8', fontWeight: 600 }}>q</label>
                        <input type="number" min="0" max="3" value={tsParams.q ?? 1} onChange={(e) => setTsParams({ ...tsParams, q: parseInt(e.target.value, 10) })} style={{ width: '100%', background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: 4, padding: '4px 8px', fontSize: 12 }} />
                      </div>
                      <div>
                        <label style={{ fontSize: 10, color: '#94a3b8', fontWeight: 600 }}>P (Seasonal)</label>
                        <input type="number" min="0" max="2" value={tsParams.P ?? 1} onChange={(e) => setTsParams({ ...tsParams, P: parseInt(e.target.value, 10) })} style={{ width: '100%', background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: 4, padding: '4px 8px', fontSize: 12 }} />
                      </div>
                      <div>
                        <label style={{ fontSize: 10, color: '#94a3b8', fontWeight: 600 }}>D (Seasonal)</label>
                        <input type="number" min="0" max="1" value={tsParams.D ?? 0} onChange={(e) => setTsParams({ ...tsParams, D: parseInt(e.target.value, 10) })} style={{ width: '100%', background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: 4, padding: '4px 8px', fontSize: 12 }} />
                      </div>
                      <div>
                        <label style={{ fontSize: 10, color: '#94a3b8', fontWeight: 600 }}>Q (Seasonal)</label>
                        <input type="number" min="0" max="2" value={tsParams.Q ?? 1} onChange={(e) => setTsParams({ ...tsParams, Q: parseInt(e.target.value, 10) })} style={{ width: '100%', background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: 4, padding: '4px 8px', fontSize: 12 }} />
                      </div>
                      <div>
                        <label style={{ fontSize: 10, color: '#94a3b8', fontWeight: 600 }}>s (Period)</label>
                        <input type="number" min="2" max="52" value={tsParams.s ?? 7} onChange={(e) => setTsParams({ ...tsParams, s: parseInt(e.target.value, 10) })} style={{ width: '100%', background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: 4, padding: '4px 8px', fontSize: 12 }} />
                      </div>
                    </div>
                  )}

                  {tsModel === 'autoreg' && (
                    <div style={{ maxWidth: 200, background: '#090d16', padding: 12, borderRadius: 8, border: '1px solid #1e293b', marginBottom: 14 }}>
                      <label style={{ fontSize: 10, color: '#94a3b8', fontWeight: 600 }}>Autoregressive Lags</label>
                      <input
                        type="number"
                        min="1"
                        max="10"
                        value={tsParams.lags ?? 2}
                        onChange={(e) => setTsParams({ ...tsParams, lags: parseInt(e.target.value, 10) })}
                        style={{ width: '100%', background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: 4, padding: '4px 8px', fontSize: 12 }}
                      />
                    </div>
                  )}

                  {/* Action Button */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <button
                      onClick={handleRunForecast}
                      disabled={isForecasting}
                      style={{
                        background: !isForecasting ? '#10b981' : '#1e293b',
                        color: !isForecasting ? '#ffffff' : '#64748b',
                        border: 'none',
                        borderRadius: 6,
                        padding: '10px 18px',
                        fontSize: 12,
                        fontWeight: 700,
                        cursor: !isForecasting ? 'pointer' : 'not-allowed',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8
                      }}
                    >
                      <Play size={14} />
                      <span>{isForecasting ? 'Fitting & Forecasting...' : 'Train & Generate Forecast'}</span>
                    </button>
                    {(!tsTimeCol || !tsTargetCol) && (
                      <span style={{ fontSize: 11, color: '#94a3b8' }}>Please select temporal and target columns above.</span>
                    )}
                  </div>
                </div>

                {/* Error Banner */}
                {forecastError && (
                  <div style={{
                    background: '#450a0a',
                    border: '1px solid #ef4444',
                    borderRadius: 8,
                    padding: '12px 16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    color: '#fca5a5',
                    fontSize: 12,
                    fontWeight: 600
                  }}>
                    <AlertCircle size={18} color="#ef4444" />
                    <span>{forecastError}</span>
                  </div>
                )}

                {/* Forecast Results */}
                {forecastResult && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    {/* Analytical Finding Box */}
                    {forecastResult.finding && (
                      <div style={{
                        background: '#0c1a2e',
                        borderLeft: '4px solid #0ea5e9',
                        borderRadius: 8,
                        padding: '12px 16px',
                        color: '#bae6fd',
                        fontSize: 12,
                        lineHeight: 1.5
                      }}>
                        <strong>Analytical Finding:</strong> {forecastResult.finding}
                      </div>
                    )}

                    {/* KPI Cards Row */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Validation RMSE</span>
                        <h4 style={{ margin: '4px 0 0', fontSize: 20, fontWeight: 800, color: '#10b981' }}>
                          {forecastResult.metrics?.rmse?.toFixed(3) || '—'}
                        </h4>
                        <span style={{ fontSize: 10, color: '#64748b' }}>
                          vs Naive Baseline: {forecastResult.baseline_metrics?.rmse?.toFixed(3) || '—'}
                        </span>
                      </div>

                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Validation MAE</span>
                        <h4 style={{ margin: '4px 0 0', fontSize: 20, fontWeight: 800, color: '#f8fafc' }}>
                          {forecastResult.metrics?.mae?.toFixed(3) || '—'}
                        </h4>
                        <span style={{ fontSize: 10, color: '#64748b' }}>Mean absolute error</span>
                      </div>

                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Forecast Horizon</span>
                        <h4 style={{ margin: '4px 0 0', fontSize: 20, fontWeight: 800, color: '#38bdf8' }}>
                          {forecastResult.horizon} Periods
                        </h4>
                        <span style={{ fontSize: 10, color: '#64748b' }}>With 95% prediction intervals</span>
                      </div>

                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: 14 }}>
                        <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Validation Strategy</span>
                        <h4 style={{ margin: '4px 0 0', fontSize: 16, fontWeight: 800, color: '#a78bfa' }}>
                          Chronological Split
                        </h4>
                        <span style={{ fontSize: 10, color: '#64748b' }}>Zero future leakage</span>
                      </div>
                    </div>

                    {/* Forecast Visualization Chart */}
                    {forecastResult.plot_base64 && (
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                        <h4 style={{ margin: '0 0 12px 0', fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                          Historical vs Forecast Projections & 95% Confidence Bounds
                        </h4>
                        <div style={{ display: 'flex', justifyContent: 'center', background: '#090d16', padding: 12, borderRadius: 8 }}>
                          <img
                            src={`data:image/png;base64,${forecastResult.plot_base64}`}
                            alt="Time-Series Forecast Plot"
                            style={{ maxWidth: '100%', maxHeight: 420, objectFit: 'contain', borderRadius: 6 }}
                          />
                        </div>
                      </div>
                    )}

                    {/* Forecast Step Projections Table */}
                    {forecastResult.forecast_records && (
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                        <h4 style={{ margin: '0 0 10px 0', fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                          Projected Observations & 95% Confidence Intervals
                        </h4>
                        <div style={{ overflowX: 'auto' }}>
                          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                            <thead>
                              <tr style={{ background: '#1e293b', color: '#94a3b8', textAlign: 'left' }}>
                                <th style={{ padding: '8px 12px' }}>Step</th>
                                <th style={{ padding: '8px 12px' }}>Future Timestamp</th>
                                <th style={{ padding: '8px 12px' }}>Point Forecast</th>
                                <th style={{ padding: '8px 12px' }}>Lower Bound (95% CI)</th>
                                <th style={{ padding: '8px 12px' }}>Upper Bound (95% CI)</th>
                              </tr>
                            </thead>
                            <tbody>
                              {forecastResult.forecast_records.map((row) => (
                                <tr key={row.step} style={{ borderBottom: '1px solid #1e293b' }}>
                                  <td style={{ padding: '8px 12px', fontWeight: 700, color: '#38bdf8' }}>+{row.step}</td>
                                  <td style={{ padding: '8px 12px', color: '#cbd5e1' }}>{row.timestamp}</td>
                                  <td style={{ padding: '8px 12px', fontWeight: 700, color: '#10b981' }}>{row.forecast.toFixed(4)}</td>
                                  <td style={{ padding: '8px 12px', color: '#94a3b8' }}>{row.lower_ci_95.toFixed(4)}</td>
                                  <td style={{ padding: '8px 12px', color: '#94a3b8' }}>{row.upper_ci_95.toFixed(4)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {/* Reproducible Python Script */}
                    {forecastResult.reproducible_code && (
                      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 16 }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                          <span style={{ fontSize: 12, fontWeight: 700, color: '#94a3b8' }}>Reproducible Python Pipeline</span>
                          <button
                            onClick={() => handleCopyCode(forecastResult.reproducible_code, 'forecast')}
                            style={{
                              background: '#1e293b',
                              color: copiedCodeType === 'forecast' ? '#34d399' : '#94a3b8',
                              border: '1px solid #334155',
                              borderRadius: 4,
                              padding: '4px 10px',
                              fontSize: 11,
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: 6
                            }}
                          >
                            {copiedCodeType === 'forecast' ? <Check size={12} /> : <Copy size={12} />}
                            <span>{copiedCodeType === 'forecast' ? 'Copied!' : 'Copy Code'}</span>
                          </button>
                        </div>
                        <pre style={{
                          background: '#090d16',
                          color: '#38bdf8',
                          padding: 12,
                          borderRadius: 6,
                          fontSize: 11,
                          fontFamily: 'monospace',
                          overflowX: 'auto',
                          margin: 0
                        }}>
                          {forecastResult.reproducible_code}
                        </pre>
                      </div>
                    )}
                  </div>
                )}

                {/* Empty State */}
                {!forecastResult && !isForecasting && (
                  <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 36, textAlign: 'center', color: '#64748b' }}>
                    <TrendingUp size={40} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
                    <h4 style={{ margin: '0 0 6px 0', fontSize: 14, color: '#f8fafc' }}>Time-Series Forecasting Studio Ready</h4>
                    <p style={{ margin: 0, fontSize: 12, maxWidth: 450, marginInline: 'auto' }}>
                      Select a temporal column and target metric above, choose your model (ARIMA, SARIMAX, AutoReg, Naive), then click <strong>"Train & Generate Forecast"</strong>.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* ---------------------------------------------------- */}
            {/* SUBTAB 2: STATISTICAL DIAGNOSTICS (ADF, KPSS, ACF)   */}
            {/* ---------------------------------------------------- */}
            {tsSubTab === 'diagnostics' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {/* Stationarity Tests (ADF & KPSS) Card */}
                <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                    <div>
                      <h4 style={{ margin: 0, fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                        Stationarity Unit Root Tests (ADF & KPSS)
                      </h4>
                      <p style={{ margin: '2px 0 0', fontSize: 11, color: '#94a3b8' }}>
                        Joint hypothesis testing to determine whether series requires differencing (d) or detrending before modeling.
                      </p>
                    </div>
                    <button
                      onClick={handleRunStationarity}
                      disabled={isTestingStationarity}
                      style={{
                        background: '#1e293b',
                        color: '#38bdf8',
                        border: '1px solid #334155',
                        borderRadius: 6,
                        padding: '8px 14px',
                        fontSize: 12,
                        fontWeight: 700,
                        cursor: 'pointer'
                      }}
                    >
                      {isTestingStationarity ? 'Computing...' : 'Run Stationarity Diagnostics'}
                    </button>
                  </div>

                  {stationarityError && (
                    <div style={{ background: '#450a0a', border: '1px solid #ef4444', borderRadius: 6, padding: '8px 12px', color: '#fca5a5', fontSize: 11, marginBottom: 12 }}>
                      {stationarityError}
                    </div>
                  )}

                  {stationarityResult && (
                    <div>
                      <div style={{
                        background: stationarityResult.conclusion === 'STATIONARY' ? '#064e3b' : '#451a03',
                        border: `1px solid ${stationarityResult.conclusion === 'STATIONARY' ? '#059669' : '#d97706'}`,
                        borderRadius: 8,
                        padding: '10px 14px',
                        marginBottom: 12,
                        color: stationarityResult.conclusion === 'STATIONARY' ? '#a7f3d0' : '#fde68a',
                        fontSize: 12
                      }}>
                        <strong>Scientific Assessment: {stationarityResult.conclusion}</strong> — {stationarityResult.interpretation}
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12 }}>
                        <div style={{ background: '#090d16', padding: 12, borderRadius: 8, border: '1px solid #1e293b' }}>
                          <span style={{ fontSize: 11, color: '#38bdf8', fontWeight: 700 }}>Augmented Dickey-Fuller (ADF)</span>
                          <div style={{ fontSize: 18, fontWeight: 800, color: '#f8fafc', marginTop: 4 }}>
                            p = {stationarityResult.adf_test.p_value?.toFixed(4)}
                          </div>
                          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
                            Test Stat: {stationarityResult.adf_test.test_statistic?.toFixed(3)} (1% Crit: {stationarityResult.adf_test.critical_values?.['1%']?.toFixed(2)})
                          </div>
                          <div style={{ fontSize: 11, fontWeight: 700, color: stationarityResult.adf_test.is_stationary ? '#10b981' : '#ef4444', marginTop: 4 }}>
                            {stationarityResult.adf_test.is_stationary ? '✓ Reject H0: Stationary' : '✗ Cannot Reject H0: Unit Root Present'}
                          </div>
                        </div>

                        <div style={{ background: '#090d16', padding: 12, borderRadius: 8, border: '1px solid #1e293b' }}>
                          <span style={{ fontSize: 11, color: '#10b981', fontWeight: 700 }}>KPSS Stationarity Test</span>
                          <div style={{ fontSize: 18, fontWeight: 800, color: '#f8fafc', marginTop: 4 }}>
                            p = {stationarityResult.kpss_test.p_value?.toFixed(4)}
                          </div>
                          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
                            Test Stat: {stationarityResult.kpss_test.test_statistic?.toFixed(3)} (5% Crit: {stationarityResult.kpss_test.critical_values?.['5%']?.toFixed(2)})
                          </div>
                          <div style={{ fontSize: 11, fontWeight: 700, color: stationarityResult.kpss_test.is_stationary ? '#10b981' : '#ef4444', marginTop: 4 }}>
                            {stationarityResult.kpss_test.is_stationary ? '✓ Fail to Reject H0: Stationary' : '✗ Reject H0: Non-Stationary'}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Autocorrelation (ACF & PACF) Card */}
                <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                    <div>
                      <h4 style={{ margin: 0, fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                        Autocorrelation (ACF) & Partial Autocorrelation (PACF)
                      </h4>
                      <p style={{ margin: '2px 0 0', fontSize: 11, color: '#94a3b8' }}>
                        Identifies autoregressive (p) and moving average (q) orders with 95% Bartlett confidence limits (±1.96/√N).
                      </p>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ fontSize: 11, color: '#94a3b8' }}>Max Lags:</span>
                        <input
                          type="number"
                          min="4"
                          max="40"
                          value={maxLags}
                          onChange={(e) => setMaxLags(parseInt(e.target.value, 10))}
                          style={{ width: 55, background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: 4, padding: '4px 6px', fontSize: 11 }}
                        />
                      </div>
                      <button
                        onClick={handleRunAcfPacf}
                        disabled={isTestingAcf}
                        style={{
                          background: '#1e293b',
                          color: '#38bdf8',
                          border: '1px solid #334155',
                          borderRadius: 6,
                          padding: '8px 14px',
                          fontSize: 12,
                          fontWeight: 700,
                          cursor: 'pointer'
                        }}
                      >
                        {isTestingAcf ? 'Computing...' : 'Evaluate ACF & PACF'}
                      </button>
                    </div>
                  </div>

                  {acfError && (
                    <div style={{ background: '#450a0a', border: '1px solid #ef4444', borderRadius: 6, padding: '8px 12px', color: '#fca5a5', fontSize: 11, marginBottom: 12 }}>
                      {acfError}
                    </div>
                  )}

                  {acfResult && (
                    <div>
                      {acfResult.plot_base64 && (
                        <div style={{ display: 'flex', justifyContent: 'center', background: '#090d16', padding: 12, borderRadius: 8, marginBottom: 12 }}>
                          <img
                            src={`data:image/png;base64,${acfResult.plot_base64}`}
                            alt="ACF and PACF Plots"
                            style={{ maxWidth: '100%', maxHeight: 360, objectFit: 'contain', borderRadius: 6 }}
                          />
                        </div>
                      )}

                      <div style={{ display: 'flex', gap: 12, fontSize: 11 }}>
                        <div style={{ background: '#090d16', padding: 10, borderRadius: 6, border: '1px solid #1e293b', flex: 1 }}>
                          <strong style={{ color: '#38bdf8' }}>Significant ACF Lags (q candidate):</strong>
                          <span style={{ color: '#cbd5e1', marginLeft: 6 }}>
                            {acfResult.significant_acf_lags?.length > 0 ? acfResult.significant_acf_lags.join(', ') : 'None'}
                          </span>
                        </div>
                        <div style={{ background: '#090d16', padding: 10, borderRadius: 6, border: '1px solid #1e293b', flex: 1 }}>
                          <strong style={{ color: '#10b981' }}>Significant PACF Lags (p candidate):</strong>
                          <span style={{ color: '#cbd5e1', marginLeft: 6 }}>
                            {acfResult.significant_pacf_lags?.length > 0 ? acfResult.significant_pacf_lags.join(', ') : 'None'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Seasonal Decomposition Card */}
                <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 18 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                    <div>
                      <h4 style={{ margin: 0, fontSize: 13, fontWeight: 700, color: '#f8fafc' }}>
                        Seasonal Trend & Noise Decomposition
                      </h4>
                      <p style={{ margin: '2px 0 0', fontSize: 11, color: '#94a3b8' }}>
                        Isolates underlying secular trend, periodic seasonality, and irregular noise residuals.
                      </p>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <select
                        value={decompModel}
                        onChange={(e) => setDecompModel(e.target.value)}
                        style={{ background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: 4, padding: '4px 8px', fontSize: 11 }}
                      >
                        <option value="additive">Additive</option>
                        <option value="multiplicative">Multiplicative</option>
                      </select>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <span style={{ fontSize: 11, color: '#94a3b8' }}>Period:</span>
                        <input
                          type="number"
                          min="2"
                          max="52"
                          value={decompPeriod}
                          onChange={(e) => setDecompPeriod(parseInt(e.target.value, 10))}
                          style={{ width: 50, background: '#1e293b', color: '#f8fafc', border: '1px solid #334155', borderRadius: 4, padding: '4px 6px', fontSize: 11 }}
                        />
                      </div>
                      <button
                        onClick={handleRunDecomposition}
                        disabled={isDecomposing}
                        style={{
                          background: '#1e293b',
                          color: '#38bdf8',
                          border: '1px solid #334155',
                          borderRadius: 6,
                          padding: '8px 14px',
                          fontSize: 12,
                          fontWeight: 700,
                          cursor: 'pointer'
                        }}
                      >
                        {isDecomposing ? 'Decomposing...' : 'Decompose'}
                      </button>
                    </div>
                  </div>

                  {decompError && (
                    <div style={{ background: '#450a0a', border: '1px solid #ef4444', borderRadius: 6, padding: '8px 12px', color: '#fca5a5', fontSize: 11, marginBottom: 12 }}>
                      {decompError}
                    </div>
                  )}

                  {decompResult && decompResult.plot_base64 && (
                    <div style={{ display: 'flex', justifyContent: 'center', background: '#090d16', padding: 12, borderRadius: 8 }}>
                      <img
                        src={`data:image/png;base64,${decompResult.plot_base64}`}
                        alt="Time-Series Decomposition Plot"
                        style={{ maxWidth: '100%', maxHeight: 450, objectFit: 'contain', borderRadius: 6 }}
                      />
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ======================================================== */}
        {/* TAB 5: Pipeline History Tree (Preserved)                 */}
        {/* ======================================================== */}
        {activeTab === 'tree' && (
          <div style={{ maxWidth: 800, margin: '0 auto' }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: 22 }}>
              <h3 style={{ margin: '0 0 16px 0', fontSize: 15, fontWeight: 800, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 8 }}>
                <Layers size={18} color="#0ea5e9" /> Transformation Pipeline Step Tree
              </h3>

              {pipelineSteps && pipelineSteps.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {pipelineSteps.map((step, idx) => (
                    <div key={step.step_id || idx} style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 12,
                      background: '#1e293b',
                      padding: 14,
                      borderRadius: 8,
                      borderLeft: '4px solid #0ea5e9'
                    }}>
                      <div style={{
                        background: '#0ea5e9',
                        color: '#fff',
                        borderRadius: '50%',
                        width: 22,
                        height: 22,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 11,
                        fontWeight: 800
                      }}>
                        {idx + 1}
                      </div>
                      <div style={{ flex: 1 }}>
                        <strong style={{ fontSize: 13, color: '#f8fafc' }}>
                          {step.operation || step.title || 'Pipeline Operation'}
                        </strong>
                        <p style={{ margin: '4px 0 0 0', fontSize: 11, color: '#94a3b8' }}>
                          {step.description || JSON.stringify(step.params || {})}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', margin: '20px 0' }}>
                  No transformations applied yet. Upload a dataset or execute an operation to see the pipeline tree!
                </p>
              )}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
