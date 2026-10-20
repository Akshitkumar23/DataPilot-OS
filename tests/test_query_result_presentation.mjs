import assert from 'node:assert/strict';
import { createManualExecutionPresentation, createQueryPresentation } from '../src/utils/queryResultPresentation.js';

const canonical = {
  request_id: 'req-ui-contract',
  status: 'SUCCEEDED',
  result: {
    response: 'Computed a scientifically validated result.',
    data: {
      columns: ['feature_x', 'pearson_r'],
      records: [{ feature_x: 'x', pearson_r: -0.83 }],
      row_count: 1
    },
    plot_image: 'aGVsbG8=',
    manifest: {
      operator: 'correlation_analysis',
      columns_used: ['x', 'y'],
      statistics_computed: ['pearson_correlation'],
      chart_type: 'scatter',
      row_count_in: 200,
      row_count_out: 1,
      assumptions: [],
      warnings: []
    },
    validation: { is_valid: true, issues: [], metrics: { manifest_operator: 'correlation_analysis' } },
    artifacts: { execution_time_sec: 0.42 }
  }
};

const presentation = createQueryPresentation(canonical);
assert.equal(presentation.execution_mode, 'typed_operator');
assert.equal(presentation.output_tables.length, 1);
assert.deepEqual(presentation.output_tables[0].columns, ['feature_x', 'pearson_r']);
assert.equal(presentation.output_tables[0].rows[0].pearson_r, -0.83);
assert.equal(presentation.metrics.pearson_r, -0.83);
assert.equal(presentation.validationReport.evidence_sufficient, true);
assert.equal(presentation.manifest.chart_type, 'scatter');

const manualFailure = createManualExecutionPresentation({ success: false, message: 'Syntax error' }, 'bad code');
assert.equal(manualFailure.status, 'FAILED');
assert.equal(manualFailure.stderr, 'Syntax error');

console.log('query-result-presentation: PASS');

const sequence = createQueryPresentation({ result: {
  ...canonical.result,
  steps: [
    { data: { columns: ['n'], records: [{ n: 4 }] }, plot_image: 'first-image', response: 'First step' },
    canonical.result,
  ],
} });
assert.equal(sequence.output_tables.length, 2);
assert.equal(sequence.output_tables[0].name, 'Step 1: Analysis result');
assert.deepEqual(sequence.plot_images, ['first-image', 'aGVsbG8=']);
assert.equal(sequence.step_findings[0].summary, 'First step');

// Advanced workbench metadata test
const advancedRecord = createQueryPresentation({
  status: 'SUCCEEDED',
  result: {
    data: { columns: ['y_pred'], records: [{ y_pred: 10.5 }] },
    explain_plan: 'EXPLAIN SCAN df',
    residual_diagnostics: { rmse: 1.25, durbin_watson: 1.98 },
    is_source_native: true,
    source_storage: 'source_native'
  }
});
assert.equal(advancedRecord.explain_plan, 'EXPLAIN SCAN df');
assert.equal(advancedRecord.residual_diagnostics.rmse, 1.25);
assert.equal(advancedRecord.residual_diagnostics.durbin_watson, 1.98);
assert.equal(advancedRecord.is_source_native, true);
assert.equal(advancedRecord.source_storage, 'source_native');

// FW-02 & SQL query presentation test
const sqlAndThresholdRecord = createQueryPresentation({
  status: 'SUCCEEDED',
  result: {
    sql: 'SELECT * FROM iris WHERE petal_length > 1.5',
    execution_time_ms: 8.4,
    row_count: 50,
    data: {
      columns: ['target'],
      records: [{ target: 1 }],
      threshold_tuning_payload: {
        y_test: [0, 1, 1, 0],
        test_probs: [0.1, 0.8, 0.7, 0.3],
        classes: ['0', '1'],
        positive_class: '1',
        baseline_threshold: 0.50,
        brier_score: 0.05,
        roc_auc: 1.0,
        pr_auc: 1.0
      }
    }
  }
});
assert.equal(sqlAndThresholdRecord.executed_sql, 'SELECT * FROM iris WHERE petal_length > 1.5');
assert.equal(sqlAndThresholdRecord.sql_metadata.execution_time_ms, 8.4);
assert.equal(sqlAndThresholdRecord.sql_metadata.row_count, 50);
assert.equal(sqlAndThresholdRecord.threshold_tuning_payload.brier_score, 0.05);
assert.equal(sqlAndThresholdRecord.threshold_tuning_payload.classes.length, 2);

console.log('query-result-presentation (advanced workbench & FW-02/SQL): PASS');
