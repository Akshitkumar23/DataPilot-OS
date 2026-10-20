/**
 * Convert the canonical DS-OS query response into the one presentation contract
 * used by the renderer.  The analytical engine remains the source of truth;
 * this module only makes its typed evidence renderable by the UI.
 */

const isRecord = (value) => value && typeof value === 'object' && !Array.isArray(value);

const asArray = (value) => Array.isArray(value) ? value : [];

const asDisplayValue = (value) => {
  if (value === null || value === undefined) return undefined;
  if (Array.isArray(value)) return value.join(', ');
  if (isRecord(value)) return JSON.stringify(value);
  return value;
};

function createEvidenceMetrics(manifest, records) {
  const metrics = {};
  if (isRecord(manifest)) {
    for (const [key, value] of Object.entries(manifest)) {
      const displayValue = asDisplayValue(value);
      if (displayValue !== undefined && displayValue !== '' && key !== 'warnings' && key !== 'assumptions') {
        metrics[key] = displayValue;
      }
    }
  }

  // A one-row typed result normally contains its directly computed statistics
  // (p-values, scores, coefficients, etc.). Surface scalar evidence as metrics
  // without inventing any dataset- or domain-specific fields.
  if (records.length === 1 && isRecord(records[0])) {
    for (const [key, value] of Object.entries(records[0])) {
      if (value === null || value === undefined || isRecord(value) || Array.isArray(value)) continue;
      metrics[key] = value;
    }
  }
  return metrics;
}

function createValidationReport(validation) {
  if (!isRecord(validation) || typeof validation.is_valid !== 'boolean') return null;
  const issues = asArray(validation.issues);
  return {
    evidence_sufficient: validation.is_valid,
    checks: [
      {
        passed: validation.is_valid,
        name: 'Semantic result contract',
        description: validation.is_valid
          ? 'The executed result satisfied the requested analytical contract.'
          : 'The executed result did not satisfy the requested analytical contract.'
      }
    ],
    missing_evidence: issues,
    metrics: isRecord(validation.metrics) ? validation.metrics : {}
  };
}

export function createQueryPresentation(body) {
  const result = isRecord(body?.result) ? body.result : (isRecord(body) ? body : {});
  const data = isRecord(result.data) ? result.data : {};
  const records = asArray(data.records || data.rows || data.data);
  const columns = asArray(data.columns).length
    ? data.columns
    : (records.length > 0 && isRecord(records[0]) ? Object.keys(records[0]) : []);
  const manifest = isRecord(result.manifest) ? result.manifest : null;
  const validationReport = createValidationReport(result.validation);
  const outputTables = records.length > 0 || columns.length > 0
    ? [{
      name: manifest?.operator ? `${String(manifest.operator).replace(/_/g, ' ')} result` : 'Analysis result',
      columns,
      rows: records,
      row_count: data.row_count ?? records.length
    }]
    : [];
  // The final payload is also the last sequence step. Include preceding
  // artifacts once, instead of silently hiding them behind the final result.
  const precedingSteps = asArray(result.steps).slice(0, -1).filter(isRecord);
  const stepPresentations = precedingSteps.map(step => createQueryPresentation({ result: { ...step, steps: [] } }));
  const sequenceTables = stepPresentations.flatMap((step, index) =>
    step.output_tables.map(table => ({ ...table, name: `Step ${index + 1}: ${table.name}` })));
  const plotImages = [...new Set([...stepPresentations.map(step => step.plot_image), result.plot_image].filter(Boolean))];

  const statusVal = body?.status || result.status || 'SUCCEEDED';
  let decisionVal = body?.decision || result.decision || null;
  if (!decisionVal) {
    if (statusVal === 'SUCCEEDED') decisionVal = 'execute';
    else if (statusVal === 'NEEDS_CLARIFICATION') decisionVal = 'clarify';
    else if (statusVal === 'UNSUPPORTED') decisionVal = 'unsupported';
    else if (statusVal === 'UNSAFE') decisionVal = 'unsafe';
    else decisionVal = 'unsafe';
  }

  const resolvedManifest = isRecord(body?.manifest) ? body.manifest : manifest;
  const resolvedValidation = isRecord(body?.validation) ? body.validation : (result.validation || null);
  const resolvedAnalysisSpec = body?.analysis_spec || body?.plan || result.analysis_spec || result.plan || null;
  const resolvedExecutionGraph = body?.execution_graph || result.execution_graph || null;
  const resolvedError = isRecord(body?.error) ? body.error : (isRecord(result.error) ? result.error : null);

  return {
    request_id: body?.request_id || result.request_id || null,
    status: statusVal,
    decision: decisionVal,
    execution_mode: 'typed_operator',
    analysis_spec: resolvedAnalysisSpec,
    execution_graph: resolvedExecutionGraph,
    output: result.response || result.answer || result.written_finding || null,
    summary: result.response || result.answer || result.written_finding || null,
    plot_image: result.plot_image || null,
    plot_images: plotImages,
    plotly_spec: result.plotly_spec || null,
    output_tables: [...sequenceTables, ...outputTables],
    step_findings: stepPresentations.map((step, index) => ({ step: index + 1, summary: step.summary })),
    metrics: {
      ...createEvidenceMetrics(resolvedManifest, records),
      ...(isRecord(data.extended_metrics) ? data.extended_metrics : {})
    },
    extended_metrics: isRecord(data.extended_metrics) ? data.extended_metrics : null,
    cv_folds: asArray(data.cv_folds),
    cv_mean: data.cv_mean ?? null,
    residual_diagnostics: isRecord(data.residual_diagnostics)
      ? data.residual_diagnostics
      : (isRecord(result.residual_diagnostics) ? result.residual_diagnostics : (isRecord(body?.residual_diagnostics) ? body.residual_diagnostics : null)),
    threshold_tuning_payload: data.threshold_tuning_payload || result.threshold_tuning_payload || body?.threshold_tuning_payload || null,
    statistical_test: isRecord(data.statistical_test) ? data.statistical_test : (isRecord(result.statistical_test) ? result.statistical_test : null),
    executed_sql: (result.sql || data.sql || data.executed_sql || result.executed_sql || (resolvedManifest?.operator === 'sql_query' && body?.sql)) || null,
    sql_metadata: (result.sql || data.sql || data.executed_sql || result.executed_sql) ? {
      execution_time_ms: result.execution_time_ms ?? data.execution_time_ms ?? null,
      row_count: result.row_count ?? data.row_count ?? (Array.isArray(records) ? records.length : null),
      total_pages: result.total_pages ?? null,
      page: result.page ?? null
    } : null,
    explain_plan: result.explain_plan || body?.explain_plan || data.explain_plan || null,
    is_source_native: result.is_source_native ?? body?.is_source_native ?? null,
    source_storage: result.source_storage || body?.source_storage || null,
    data: data,
    manifest: resolvedManifest,
    validation: resolvedValidation,
    validationReport,
    artifacts: isRecord(result.artifacts) ? result.artifacts : {},
    clarification: body?.clarification || result.clarification || null,
    error: resolvedError,
    reproducible_code: result.code || null,
    timing: result.artifacts?.execution_time_sec !== undefined
      ? `${result.artifacts.execution_time_sec}s`
      : null
  };
}

export function createManualExecutionPresentation(result, code) {
  const succeeded = result?.success === true;
  const metrics = Object.fromEntries(
    Object.entries({
      row_count: result?.row_count,
      column_count: result?.column_count,
      execution_time_sec: result?.execution_time_sec
    }).filter(([, value]) => value !== null && value !== undefined)
  );
  return {
    request_id: `manual-${Date.now()}`,
    status: succeeded ? 'SUCCEEDED' : 'FAILED',
    execution_mode: 'manual_python',
    output: null,
    stdout: result?.output || '',
    stderr: succeeded ? null : (result?.message || result?.error || 'Manual Python execution failed.'),
    plot_image: result?.plot_image || null,
    output_tables: asArray(result?.output_tables),
    metrics,
    code,
    timing: result?.execution_time_sec !== undefined ? `${result.execution_time_sec}s` : null
  };
}
