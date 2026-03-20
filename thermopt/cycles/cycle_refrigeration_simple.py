from . import cycle_refrigeration_recuperated


def evaluate_cycle(
    variables,
    parameters,
    constraints,
    objective_function,
):

    output = cycle_refrigeration_recuperated.evaluate_cycle(
        variables,
        parameters,
        constraints,
        objective_function,
        recuperated=False,
    )

    return output
