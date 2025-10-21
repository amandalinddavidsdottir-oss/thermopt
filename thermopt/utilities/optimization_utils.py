import re
import yaml
import numbers
import numpy as np
import jax.numpy as jnp

from . import numerics as num


# TODO: @Lasse: We need some of these functions for the cycle optimization, and eventually also for turbomachinery optimization
# TODO: We have to discuss how to handle "rendered" variables


def render_and_evaluate(expression, data):
    """
    Render variables prefixed with '$' in an expression and evaluate the resulting expression.

    This function processes an input string `expr`, identifying all occurrences of variables
    indicated by a leading '$' symbol. Each such variable is resolved to its value from the
    provided `context` dictionary. The expression with all variables resolved is then evaluated
    and the result is returned.

    This function is useful to render strings defined in a YAML configuration file to values
    that are calculated within the code and stored in a dicitonary.

    Parameters
    ----------
    expr : str
        The expression string containing variables to be rendered. Variables in the
        expression are expected to be prefixed with a '$' symbol.
    data : dict
        A dictionary containing variables and their corresponding values. These variables
        are used to render values in the expression.

    Returns
    -------
    The result of evaluating the rendered expression. The type of the result depends on the
    expression.

    Notes
    -----
    - `pattern`: A regular expression pattern used to identify variables within the expression.
      Variables are expected to be in the format `$variableName`, potentially with dot-separated
      sub-properties (e.g., `$variable.property`).

    - `replace_with_value`: An inner function that takes a regex match object and returns
      the value of the variable from `context`. `match.group(1)` returns the first captured
      group from the matched text, which in this case is the variable name excluding the
      leading '$' symbol. For example, in `$variableName`, `match.group(1)` would return
      `variableName`.

    - The function uses Python's `eval` for evaluation, which should be used cautiously as
      it can execute arbitrary code. Ensure that the context and expressions are from a trusted
      source.
    """

    # Return the value directly when it is not a string (e.g, float or integer)
    if not isinstance(expression, str):
        return expression

    # Pattern to find $variable expressions
    pattern = re.compile(r"\$(\w+(\.\w+)*)")

    # Function to replace each match with its resolved value
    def replace_with_value(match):
        nested_key = match.group(1)
        try:
            value = render_nested_value(nested_key, data)
            if isinstance(value, np.ndarray) or isinstance(value, jnp.ndarray):
                return "np.array(" + repr(value.tolist()) + ")"
            else:
                return repr(value)
        except KeyError:
            raise KeyError(
                f"Variable '{nested_key}' not found in the provided data context."
            )

    try:
        # Replace all $variable with their actual values
        resolved_expr = pattern.sub(replace_with_value, expression)

        # Check if any unresolved variables remain
        if "$" in resolved_expr:
            raise ValueError(f"Unresolved variable in expression: '{resolved_expr}'")

        # Now evaluate the expression
        return eval(resolved_expr)

    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in '{expression}': {e}")
    except Exception as e:
        # Enhanced error message
        raise TypeError(
            f"Error evaluating expression '{expression}': {e}.\n"
            "If the expression is meant to use data from the configuration, "
            "ensure each variable is prefixed with '$'. For example, use '$variable_name' "
            "instead of 'variable_name'."
        )


def render_nested_value(nested_key, data):
    """
    Retrieves a value from a nested structure (dictionaries or objects with attributes) using a dot-separated key.

    This function is designed to navigate through a combination of dictionaries and objects. For an object to be
    compatible with this function, it must implement a `keys()` method that returns its attribute names.

    This function is intended as a subroutine of the more genera ``render_expression``

    Parameters
    ----------
    nested_key : str
        A dot-separated key string that specifies the path in the structure.
        For example, 'level1.level2.key' will retrieve data['level1']['level2']['key'] if data is a dictionary,
        or data.level1.level2.key if data is an object or a combination of dictionaries and objects.

    data : dict or object
        The starting dictionary or object from which to retrieve the value. This can be a nested structure
        of dictionaries and objects.

    Returns
    -------
    value
        The value retrieved from the nested structure using the specified key.
        The type of the value depends on what is stored at the specified key in the structure.

    Raises
    ------
    KeyError
        If the specified nested key is not found in the data structure. The error message includes the part
        of the path that was successfully traversed and the available keys or attributes at the last valid level.
    """
    keys = nested_key.split(".")
    value = data
    traversed_path = []

    for key in keys:
        if isinstance(value, dict):
            # Handle dictionary-like objects
            if key in value:
                traversed_path.append(key)
                value = value[key]
            else:
                valid_keys = ", ".join(value.keys())
                traversed_path_str = (
                    ".".join(traversed_path) if traversed_path else "root"
                )
                raise KeyError(
                    f"Nested key '{key}' not found at '{traversed_path_str}'. Available keys: {valid_keys}"
                )
        elif hasattr(value, key):
            # Handle objects with attributes
            traversed_path.append(key)
            value = getattr(value, key)
        else:
            traversed_path_str = ".".join(traversed_path)
            available_keys = ", ".join(value.keys())
            raise KeyError(
                f"Key '{key}' not found in object at '{traversed_path_str}'. Available keys: {available_keys}"
            )

    if not num.is_numeric(value):
        raise ValueError(
            f"The key '{nested_key}' is not numeric. Key value is: {value}"
        )

    return value


# def evaluate_constraints(data, constraints):
#     """
#     Evaluates the constraints based on the provided data and constraint definitions.

#     Parameters
#     ----------
#     data : dict
#         A dictionary containing performance data against which the constraints will be evaluated.
#     constraints : list of dicts
#         A list of constraint definitions, where each constraint is defined as a dictionary.
#         Each dictionary must have 'variable' (str), 'type' (str, one of '=', '>', '<'), and 'value' (numeric).

#     Returns
#     -------
#     tuple of numpy.ndarray
#         Returns two numpy arrays: the first is an array of equality constraints, and the second is an array of
#         inequality constraints. These arrays are flattened and concatenated from the evaluated constraint values.

#     Raises
#     ------
#     ValueError
#         If an unknown constraint type is specified in the constraints list.
#     """
#     # Initialize constraint lists
#     c_eq = []  # Equality constraints
#     c_ineq = []  # Inequality constraints

#     # Loop over all constraint from configuration file
#     for constraint in constraints:
#         name = constraint["variable"]
#         constraint_type = constraint["type"]
#         target = constraint["value"]
#         normalize = constraint.get("normalize", False)

#         # Get the performance value for the given variable name
#         current = render_and_evaluate(name, data)
#         if isinstance(target, str):  # Try to render variable when not a number
#             target = render_and_evaluate(target, data)

#         # Evaluate constraint
#         # mismatch = current - target
#         mismatch = target - current

#         # Normalize constraint according to specifications
#         normalize_factor = normalize if num.is_numeric(normalize) else target
#         if normalize is not False:
#             if normalize_factor == 0:
#                 raise ValueError(
#                     f"Cannot normalize constraint '{name} {constraint_type} {target}' because the normalization factor is '{normalize_factor}' (division by zero)."
#                 )
#             mismatch /= normalize_factor

#         # Add constraints to lists
#         if constraint_type == "=":
#             c_eq.append(mismatch)
#         elif constraint_type == ">":
#             c_ineq.append(mismatch)
#         elif constraint_type == "<":
#             # Change sign because optimizer handles c_ineq > 0
#             c_ineq.append(-mismatch)
#         else:
#             raise ValueError(f"Unknown constraint type: {constraint_type}")

#     # Flatten and concatenate constraints
#     c_eq = np.hstack([np.atleast_1d(item) for item in c_eq]) if c_eq else np.array([])
#     c_ineq = (
#         np.hstack([np.atleast_1d(item) for item in c_ineq]) if c_ineq else np.array([])
#     )

#     return c_eq, c_ineq

def evaluate_constraints(data, constraints):
    """
    Evaluates the constraints based on the provided data and constraint definitions.

    Parameters
    ----------
    data : dict
        A dictionary containing performance data against which the constraints will be evaluated.
    constraints : list of dicts
        A list of constraint definitions, where each constraint is defined as a dictionary.
        Each dictionary must have 'variable' (str), 'type' (str, one of '=', '>', '<'), and 'value' (numeric).

    Returns
    -------
    tuple:
        - NumPy array of equality constraint residuals
        - NumPy array of inequality constraint residuals
        - List of dicts with detailed constraint evaluation
    """
    # Initialize constraint lists
    c_eq = []
    c_ineq = []
    output = []

    # Loop over all constraint from configuration file
    for constraint in constraints:
        name_expr = constraint["variable"]
        constraint_type = constraint["type"]
        target = constraint["value"]
        normalize = constraint.get("normalize", False)

        # Evaluate constraint value
        current = render_and_evaluate(name_expr, data)
        if isinstance(target, str):
            target = render_and_evaluate(target, data)

        # Always treat current as array for uniform handling
        current = np.atleast_1d(current)

        for i, curr_val in enumerate(current):
            # mismatch = target - curr_val
            mismatch = curr_val - target

            if normalize is not False:
                normalize_factor = normalize if num.is_numeric(normalize) else target
                if normalize_factor == 0:
                    raise ValueError(
                        f"Cannot normalize constraint '{name_expr}[{i}] {constraint_type} {target}' "
                        f"because normalization factor is 0."
                    )
                norm_mismatch = mismatch / normalize_factor
            else:
                norm_mismatch = mismatch
                normalize_factor = None

            # Determine satisfaction
            tol = 1e-4
            if constraint_type == "=":
                satisfied = np.isclose(norm_mismatch, 0.0, atol=tol)
                c_eq.append(norm_mismatch)

            elif constraint_type == ">":
                satisfied = norm_mismatch > -tol or np.isclose(norm_mismatch, 0.0, atol=tol)
                c_ineq.append(-norm_mismatch)  # We flip sign: constraint is norm_mismatch ≥ 0 → -norm_mismatch ≤ 0

            elif constraint_type == "<":
                satisfied = norm_mismatch < tol or np.isclose(norm_mismatch, 0.0, atol=tol)
                c_ineq.append(norm_mismatch)  # Constraint is norm_mismatch ≤ 0

            else:
                raise ValueError(f"Unknown constraint type: {constraint_type}")


            # Add index if current is an array
            name_out = f"{name_expr}[{i}]" if current.size > 1 else name_expr

            output.append({
                "name": name_out,
                "value": curr_val,
                "type": constraint_type,
                "target": target,
                "mismatch": mismatch,
                "normalized_mismatch": norm_mismatch,
                "satisfied": satisfied,
                "normalize": normalize_factor,
            })

    return c_eq, c_ineq, output

      

def evaluate_objective_function(data, objective_function):
    """
    Evaluates the objective function based on the provided data and configuration.

    Parameters
    ----------
    data : dict
        A dictionary containing performance data against which the objective function will be evaluated.
    objective_function : dict
        A dictionary defining the objective function. It must have 'variable' (str)
        and 'type' (str, either 'minimize' or 'maximize').

    Returns
    -------
    float
        The value of the objective function, adjusted for optimization. Positive for minimization and
        negative for maximization.

    Raises
    ------
    ValueError
        If an unknown objective function type is specified in the configuration.
    """

    # Get the performance value for the given variable name
    name = objective_function["variable"]
    type = objective_function["type"]
    value = render_and_evaluate(name, data)

    # Ensure it represents a scalar
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(
            f"The objective function '{name}' must be scalar, "
            f"but got array with shape {value.shape} and value {value}."
        )

    if type == "minimize":
        return value
    elif type == "maximize":
        return -value
    else:
        raise ValueError(f"Unknown objective function type: {type}")
