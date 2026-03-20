from . import cycle_power_recuperated_dual_pressure_single_brine


def evaluate_cycle(
    variables,
    parameters,
    constraints,
    objective_function,
):
    """
    Non-recuperated dual-pressure ORC with a single shared brine source.

    All brine from the wells is throttled to the same pressure and flows
    sequentially through the HP evaporator, LP evaporator, and preheater.

    Delegates to ``cycle_power_recuperated_dual_pressure_single_brine``
    with ``recuperated=False``.
    """
    return cycle_power_recuperated_dual_pressure_single_brine.evaluate_cycle(
        variables,
        parameters,
        constraints,
        objective_function,
        recuperated=False,
    )
