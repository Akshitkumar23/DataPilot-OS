"""Execute sequential reference plans with per-step result contracts."""
from training.corpus_contract import reference_commands


def execute_reference(record, dataframe):
    from backend.ds_engine.command_resolver import CommandResolver
    from backend.ds_engine.dataset_profiler import DatasetProfiler
    from backend.ds_engine.typed_operators import TypedOperatorRegistry
    from backend.ds_engine.result_validator import SemanticResultValidator
    from modules.query_orchestrator import _map_intent_to_v2
    commands = reference_commands(record)
    if not commands:
        raise ValueError('Clarification is not executable')
    current = dataframe.copy(deep=True)
    outputs = []
    for index, command in enumerate(commands):
        resolved = CommandResolver.resolve_command(command, df=current, raw_query=record['query'])
        if not resolved.is_valid or not resolved.semantic_intent:
            raise ValueError(f'Step {index + 1} resolver: {resolved.error_code} {resolved.error_message}')
        intent = _map_intent_to_v2(resolved.semantic_intent, record['query'], df=current,
                                  validated_tool_args=command['arguments'])
        output = TypedOperatorRegistry.execute_intent(current, intent, DatasetProfiler.profile(current))
        if output.status != 'SUCCEEDED':
            raise ValueError(f'Step {index + 1} execution: {output.error_code} {output.error_message}')
        validation = SemanticResultValidator.validate_output(intent, output, input_row_count=len(current))
        if not validation.is_valid:
            raise ValueError(f'Step {index + 1} result: {validation.issues}')
        outputs.append(output)
        if index < len(commands) - 1:
            if output.dataframe is None:
                raise ValueError(f'Step {index + 1} does not produce a dataframe for the next step')
            current = output.dataframe.copy(deep=True)
    return outputs
