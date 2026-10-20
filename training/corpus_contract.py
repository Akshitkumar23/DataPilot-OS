"""Versioned reference-plan access shared by corpus tooling."""


def reference_commands(record):
    if record.get('expected_decision') == 'clarification':
        if 'expected_command' in record or 'expected_commands' in record:
            raise ValueError('Clarification cannot contain executable references')
        return []
    if 'expected_command' in record and 'expected_commands' in record:
        raise ValueError('Provide one reference representation, not both')
    if 'expected_commands' in record:
        if record.get('schema_version') != 'planner-sft-v2':
            raise ValueError('expected_commands requires planner-sft-v2')
        commands = record['expected_commands']
    else:
        commands = [record.get('expected_command')]
    if not isinstance(commands, list) or not commands:
        raise ValueError('Reference plan must contain at least one command')
    for command in commands:
        if (not isinstance(command, dict) or set(command) != {'tool', 'arguments'}
                or not isinstance(command['tool'], str) or not command['tool']
                or not isinstance(command['arguments'], dict)):
            raise ValueError('Each reference command requires only tool and object arguments')
    return commands


def reference_family(record):
    commands = reference_commands(record)
    return 'clarification' if not commands else 'multi_step' if len(commands) > 1 else commands[0]['tool']


def reference_identity(record):
    commands = reference_commands(record)
    return commands if commands else {'clarify': record['query']}
