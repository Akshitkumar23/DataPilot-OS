"""Corrected bilingual curation seeds for every registered analytical tool.

These are AI-authored requests on observed public data, not human-reviewed
labels. No training split is assigned until independent holdouts are curated.
"""
import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.build_contract_verified_corpus import RECIPES, PROVENANCE
from training.planner_prompt import fingerprint
from training.reference_execution import execute_reference

# Offline reviewed wording, never imported by the runtime query router.
WORDING = {
 'filter_data': ('Return all rows where `age` > 30; do not truncate.', '`age` 30 se zyada wali saari rows do; limit mat lagao.'),
 'sort_data': ('Sort rows by `balance` descending and return the first 20.', '`balance` descending sort karke pehli 20 rows do.'),
 'conditional_rate': ('For each `job`, divide rows where `y` equals "yes" by all rows in that job. Keep jobs with at least 100 rows, sort rate_percent descending, return at most 20 groups and draw a bar chart.', 'Har `job` mein `y`="yes" rows ko us job ki total rows se divide karke percentage do. Kam se kam 100 rows wale groups lo, rate_percent descending sort karo, top 20 aur bar chart do.'),
 'group_data': ('Compute mean `balance` grouped by `marital`, with a bar chart.', '`marital` ke har group ka mean `balance` aur bar chart do.'),
 'aggregate_data': ('Count rows where `housing` equals "yes".', '`housing`="yes" wali rows ki ginti do.'),
 'audit_metric_distribution': ('Audit `balance` for rows where `housing` equals "yes"; include a box plot.', '`housing`="yes" rows mein `balance` distribution audit karo aur box plot banao.'),
 'correlate_data': ('Compute pairwise Pearson correlation between `alcohol` and `density`, with a scatter plot. Use a linear relationship, no smoother and no assumed direction.', '`alcohol` aur `density` ki pairwise Pearson correlation aur scatter plot do. Linear relation evaluate karo; smoother nahi aur direction pehle se assume nahi karni.'),
 'forecast_data': ('Using `date` and `Appliances`, forecast the next 24 intervals at 10-minute frequency with Holt linear trend and a line chart.', '`date` aur `Appliances` se agle 24 intervals ka forecast 10-minute frequency par Holt linear trend se karo aur line chart do.'),
 'clean_data': ('Drop rows with actual missing/null values in `workclass`; do not impute values.', '`workclass` mein actual missing/null value wali rows drop karo; imputation mat karo.'),
 'detect_anomalies': ('Use Isolation Forest on `residual sugar` and `alcohol` to detect anomalous rows.', '`residual sugar` aur `alcohol` par Isolation Forest lagakar anomalous rows detect karo.'),
 'eda_data': ('Produce a tabular exploratory profile for `Appliances` and `T1`; no chart.', '`Appliances` aur `T1` ka tabular EDA profile do; chart nahi.'),
 'train_model': ('For retrospective explanation, train a classification model for `income` using only `age`, `education_num`, and `hours_per_week`; report a confusion matrix. Do not present it as a prospective model.', 'Retrospective explanation ke liye `income` classifier train karo; sirf `age`, `education_num`, `hours_per_week` features lo aur confusion matrix do. Ise prospective model mat batao.'),
 'cluster_data': ('Create 3 clusters from `alcohol`, `residual sugar`, and `density`, using standard scaling.', '`alcohol`, `residual sugar`, `density` ki standard scaling karke 3 clusters banao.'),
 'statistical_test': ('Use one-way ANOVA at alpha 0.05 to compare `alcohol` across groups defined by `quality`.', '`quality` ke groups mein `alcohol` compare karne ke liye one-way ANOVA alpha 0.05 par karo.'),
 'categorical_outcome_association': ('For retrospective association only, compare numeric `duration` across categories of `y` with a box plot. Do not claim prospective prediction.', 'Sirf retrospective association ke liye numeric `duration` ko `y` categories mein compare karke box plot do. Prospective prediction ka claim nahi karna.'),
 'pca_analysis': ('Reduce `alcohol`, `density`, and `pH` to 2 principal components with PCA.', '`alcohol`, `density`, `pH` par PCA karke 2 principal components do.'),
 'explain_features': ('Explain associations of `age`, `education_num`, and `hours_per_week` with target `income`, using only those predictors for retrospective explanation.', 'Retrospective explanation mein target `income` ke saath sirf `age`, `education_num`, `hours_per_week` predictors ke associations explain karo.'),
 'autocorrelation': ('Compute autocorrelation of `Appliances` through 24 lags in the supplied observation order.', 'Diye gaye observation order mein `Appliances` ki autocorrelation 24 lags tak nikaalo.'),
 'decompose_time_series': ('Order by `date` and decompose `Appliances` with an additive model and seasonal period of 144 observations.', '`date` order mein `Appliances` ka additive decomposition karo; seasonal period 144 observations lo.'),
 'event_window_analysis': ('Order observations by `date`. Treat each row with `lights` > 0 as an event; analyze `Appliances` over the 24 observations preceding events, with a box plot.', '`date` se observations order karo. Har `lights` > 0 row ko event maano; events se pehle ki 24 observations mein `Appliances` analyze karke box plot banao.'),
}


def seeds():
    records = []
    for recipe in RECIPES:
        command = {'tool': recipe['tool'], 'arguments': deepcopy(recipe['args'])}
        family = fingerprint([recipe['dataset'], command])
        for language, query in zip(('en', 'hinglish'), WORDING[recipe['tool']]):
            records.append({'schema_version': 'planner-sft-v2', 'case_id': f'core-{recipe["tool"]}-{language}',
                            'dataset_id': recipe['dataset'], 'query': query, 'language': language,
                            'task_family_id': family, 'required_columns': recipe['columns'],
                            'expected_decision': 'tool', 'expected_commands': [command],
                            'messages': [{'role': 'user', 'content': query}, {'role': 'assistant', 'content': json.dumps({'mode': 'tool', 'commands': [command]})}],
                            'curation_status': 'ai-authored-review-required', 'human_reviewed': False,
                            'split_assignment': 'pending-family-level-curation'})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a curation workspace')
    from modules.capability_registry import CapabilityRegistry
    registry = CapabilityRegistry(auto_bootstrap=True)
    records = seeds()
    analytical = {s.name for s in registry.list_tool_specs() if s.category not in {'general_chat', 'clarification', 'clarify'}}
    if analytical != {r['expected_commands'][0]['tool'] for r in records}:
        raise ValueError('Live registry coverage differs from seed inventory')
    provenance = json.loads(PROVENANCE.read_text(encoding='utf-8'))
    datasets = {d['id']: d for d in provenance['datasets']}
    for r in records:
        cmd = r['expected_commands'][0]
        registry.validate_tool_call(cmd['tool'], cmd['arguments'])
        if not set(r['required_columns']) <= set(datasets[r['dataset_id']]['columns']):
            raise ValueError('Reference columns absent from source')
    args.output.mkdir(parents=True)
    (args.output / 'candidates.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n', encoding='utf-8')
    evidence, frames = [], {}
    if args.execute:
        import pandas as pd
        for r in records[::2]:
            try:
                did = r['dataset_id']
                if did not in frames:
                    path = ROOT / str(datasets[did]['file']).replace('\\', '/')
                    if hashlib.sha256(path.read_bytes()).hexdigest() != datasets[did]['sha256']:
                        raise ValueError('Dataset integrity failure')
                    frames[did] = pd.read_csv(path)
                outputs = execute_reference(r, frames[did])
                evidence.append({'case_id': r['case_id'], 'task_family_id': r['task_family_id'], 'passed': True,
                                 'manifest': outputs[-1].manifest.model_dump(mode='json')})
            except Exception as exc:
                evidence.append({'case_id': r['case_id'], 'task_family_id': r['task_family_id'], 'passed': False, 'error': str(exc)})
            print(f"Checked {r['case_id']}: {evidence[-1]['passed']}", flush=True)
    report = {'analytical_tools_covered': len(analytical), 'independent_reference_plans': len(records)//2,
              'language_variants': len(records), 'execution_checked': len(evidence),
              'execution_passed': sum(e['passed'] for e in evidence), 'results': evidence,
              'training_ready': False, 'human_reviewed': 0,
              'limits': ['Execution/manifest check is not independent numerical verification.',
                         'Hinglish variants need semantic review; only English variants were executed.',
                         'No train/validation/test split assigned. No clarification or multi-turn coverage claimed.']}
    (args.output / 'coverage-and-evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
