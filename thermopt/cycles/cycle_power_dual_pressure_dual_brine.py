from . import cycle_power_recuperated_dual_pressure_dual_brine


def evaluate_cycle(
    variables,
    parameters,
    constraints,
    objective_function,
):
    """
    Non-recuperated dual-pressure ORC with two separate brine sources.

    The HP brine feeds only the HP evaporator; the LP brine feeds the
    LP evaporator and preheater in series. Each brine stream has its
    own reinjection pump.

    Delegates to ``cycle_power_recuperated_dual_pressure_dual_brine``
    with ``recuperated=False``.
    """
    return cycle_power_recuperated_dual_pressure_dual_brine.evaluate_cycle(
        variables,
        parameters,
        constraints,
        objective_function,
        recuperated=False,
    )
