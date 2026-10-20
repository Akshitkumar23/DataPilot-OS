/**
 * DataPilot OS — Frontend Response Contract Normalizer.
 * Enforces pure state transitions based on canonical API response:
 * - execute: SUCCEEDED with analytical artifacts
 * - clarify: NEEDS_CLARIFICATION with explicit candidate options
 * - unsupported: UNSUPPORTED with capability guidance
 * - unsafe: UNSAFE with safety / leakage guard details
 * - failed: FAILED with stage diagnostics
 */

export const DECISION_TYPES = {
  EXECUTE: 'execute',
  CLARIFY: 'clarify',
  UNSUPPORTED: 'unsupported',
  UNSAFE: 'unsafe',
  FAILED: 'failed',
};

export const STATUS_TYPES = {
  SUCCEEDED: 'SUCCEEDED',
  NEEDS_CLARIFICATION: 'NEEDS_CLARIFICATION',
  UNSUPPORTED: 'UNSUPPORTED',
  UNSAFE: 'UNSAFE',
  FAILED: 'FAILED',
};

export function normalizeResponseContract(data) {
  if (!data || typeof data !== 'object') {
    return {
      status: STATUS_TYPES.FAILED,
      decision: DECISION_TYPES.FAILED,
      error: { code: 'INVALID_RESPONSE', message: 'Received empty or non-object response from engine.' },
    };
  }

  const rawStatus = (data.status || (data.result?.status) || 'SUCCEEDED').toUpperCase().trim();
  let decision = data.decision ? String(data.decision).toLowerCase().trim() : null;

  if (!decision) {
    if (rawStatus === 'SUCCEEDED') decision = DECISION_TYPES.EXECUTE;
    else if (rawStatus === 'NEEDS_CLARIFICATION') decision = DECISION_TYPES.CLARIFY;
    else if (rawStatus === 'UNSUPPORTED') decision = DECISION_TYPES.UNSUPPORTED;
    else if (rawStatus === 'UNSAFE') decision = DECISION_TYPES.UNSAFE;
    else if (rawStatus === 'FAILED') {
      const errCode = (data.error?.code || data.error_code || '').toUpperCase();
      if (errCode.includes('LEAKAGE') || errCode.includes('UNSAFE')) {
        decision = DECISION_TYPES.UNSAFE;
      } else if (errCode.includes('UNSUPPORTED')) {
        decision = DECISION_TYPES.UNSUPPORTED;
      } else if (errCode.includes('CLARIF') || errCode.includes('AMBIGUOUS')) {
        decision = DECISION_TYPES.CLARIFY;
      } else {
        decision = DECISION_TYPES.FAILED;
      }
    } else {
      decision = DECISION_TYPES.FAILED;
    }
  }

  const errorObj = data.error || (data.error_code ? { code: data.error_code, message: data.message } : null);
  const candidateColumns = data.error?.candidate_columns || data.candidate_columns || [];
  const slotFilling = data.slot_filling || data.error?.slot_filling || data.slotFilling || null;
  const inferredBindings = data.inferred_bindings || data.result?.inferred_bindings || [];
  const isDatasetWide = Boolean(data.is_dataset_wide || data.result?.is_dataset_wide);

  return {
    requestId: data.request_id || null,
    status: rawStatus,
    decision,
    analysisSpec: data.analysis_spec || data.plan || null,
    executionGraph: data.execution_graph || null,
    result: data.result || null,
    manifest: data.manifest || data.result?.manifest || null,
    validation: data.validation || data.result?.validation || null,
    clarification: data.clarification || null,
    candidateColumns,
    slotFilling,
    inferredBindings,
    isDatasetWide,
    error: errorObj,
  };
}
