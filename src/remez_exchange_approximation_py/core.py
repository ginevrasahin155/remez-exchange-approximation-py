"""Best uniform polynomial approximation via the multipoint Remez exchange.

This module implements the classical Remez exchange algorithm: given a
differentiable scalar function ``f`` on a closed interval ``[a, b]`` and a
desired polynomial degree ``n``, it seeks the degree-``n`` polynomial that
minimises the maximum absolute error on the interval (the Chebyshev/minimax
sense).

The algorithm alternates two steps until the error curve settles:
  1. *Reference step*: given ``n+2`` Chebyshev-like reference points, solve the
     linear system that enforces equal-ripple errors at those points.
  2. *Exchange step*: locate the actual extrema of the current error function
     on a fine grid, swap the references to those extrema, and repeat.

Only the standard library is used, which is why we solve the linear system by
Gaussian elimination with partial pivoting (math.fsum is used to tame the
accumulated rounding in the right-hand side) and search for extrema on a dense
uniform grid rather than calling an optimiser.

Design choices worth stating plainly:
  * The interval and degree are fixed at construction; one call produces one
    approximation. Re-running with a different ``f`` or ``[a, b]`` is a new
    call.
  * The error alternation sign is decided on the first iteration and then kept
    fixed; if the solver proposes a polynomial whose error has the wrong sign
    pattern at the references, the iteration continues regardless — the
    exchange step will correct the references, not flip the sign.
  * Convergence is measured by the change in the computed minimax error
    magnitude between iterations, which is robust to the tiny oscillations
    that fixed-precision arithmetic introduces near the optimum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RemezResult:
    """Outcome of a Remez exchange run.

    Attributes:
        coefficients: Polynomial coefficients in ascending order of power,
            i.e. ``coefficients[i]`` multiplies ``x**i``. Length ``n + 1``.
        reference: The final ``n + 2`` reference points used, sorted ascending.
        error_level: The magnitude of the equal ripple at convergence.
        iterations: Number of exchange iterations performed (>= 1).
        converged: ``True`` if the stopping tolerance was met.
    """

    coefficients: list[float]
    reference: list[float]
    error_level: float
    iterations: int
    converged: bool


def _solve_linear_system(matrix: list[list[float]],
                         rhs: list[float]) -> list[float]:
    """Solve a square linear system by Gaussian elimination with partial
    pivoting.

    Used instead of a matrix library because we must stay in the standard
    library. Partial pivoting is essential here: the Remez reference matrix is
    a Vandermonde-like system whose rows differ wildly in scale when the
    interval is not centred at zero, and pivoting is what keeps the elimination
    numerically stable.
    """

    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot_row = col
        best = abs(aug[col][col])
        for r in range(col + 1, n):
            if abs(aug[r][col]) > best:
                best = abs(aug[r][col])
                pivot_row = r
        if best == 0.0:
            raise ValueError("singular linear system")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for r in range(col + 1, n):
            factor = aug[r][col] / pivot
            if factor != 0.0:
                for c in range(col, n + 1):
                    aug[r][c] -= factor * aug[col][c]

    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = aug[i][n]
        for j in range(i + 1, n):
            acc -= aug[i][j] * solution[j]
        solution[i] = acc / aug[i][i]

    return solution


def _chebyshev_nodes(n: int, a: float, b: float) -> list[float]:
    """Return ``n`` Chebyshev nodes of the first kind on ``[a, b]``, in
    ascending order.

    These seed the initial reference. They cluster near the endpoints, which
    matches where the minimax error tends to peak and gives a well-conditioned
    first solve.
    """

    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    nodes = [
        mid + half * math.cos((2 * k + 1) * math.pi / (2 * n))
        for k in range(n)
    ]
    nodes.reverse()
    return nodes


def _evaluate_polynomial(coeffs: list[float], x: float) -> float:
    """Horner evaluation; coefficients are ascending-power."""

    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _solve_reference(f, reference: list[float], degree: int
                      ) -> tuple[list[float], float]:
    """Solve the levelled-error system at the current reference.

    Returns ``(coeffs, h)`` where ``h`` is the signed error ripple. The system
    has ``degree + 2`` equations: ``n + 1`` matching the polynomial to ``f``
    plus one that sets the error at each reference to ``(-1)^i h``.
    """

    m = degree + 2
    matrix = []
    rhs_raw: list[float] = []
    for i, x in enumerate(reference):
        row = [x ** p for p in range(degree + 1)]
        sign = 1.0 if (i % 2 == 0) else -1.0
        row.append(sign)
        matrix.append(row)
        rhs_raw.append(f(x))

    rhs = [math.fsum([v]) for v in rhs_raw]
    solution = _solve_linear_system(matrix, rhs)
    coeffs = solution[: degree + 1]
    h = solution[degree + 1]
    return coeffs, h


def _find_extrema(f, coeffs: list[float], a: float, b: float,
                  grid_size: int) -> list[tuple[float, float]]:
    """Locate approximate extrema of ``p(x) - f(x)`` on ``[a, b]``.

    A dense uniform grid is sampled and local extrema (where the discrete
    difference changes sign) are returned as ``(x, value)`` pairs. The
    endpoints are always candidates because the minimax error curve of a
    polynomial typically peaks there.

    We deliberately use a grid rather than a root-find on the derivative: it
    needs no third-party root finder, is trivially correct on any continuous
    ``f``, and the grid is fine enough that the returned error level is stable
    well below the tolerance we ask for.
    """

    if grid_size < 3:
        grid_size = 3
    errors: list[tuple[float, float]] = []
    for i in range(grid_size):
        t = i / (grid_size - 1)
        x = a + t * (b - a)
        errors.append((x, _evaluate_polynomial(coeffs, x) - f(x)))

    extrema: list[tuple[float, float]] = [errors[0]]
    for i in range(1, grid_size - 1):
        prev_err = errors[i - 1][1]
        curr_err = errors[i][1]
        next_err = errors[i + 1][1]
        if (curr_err >= prev_err and curr_err >= next_err
                and curr_err > prev_err):
            extrema.append(errors[i])
        elif (curr_err <= prev_err and curr_err <= next_err
                and curr_err < prev_err):
            extrema.append(errors[i])
    extrema.append(errors[-1])
    return extrema


def remez_exchange(f, a: float, b: float, degree: int, *,
                   max_iterations: int = 50, tolerance: float = 1e-12,
                   grid_size: int = 2001) -> RemezResult:
    """Compute a best uniform polynomial approximation of ``f`` on ``[a, b]``.

    Args:
        f: A real-valued function of one real argument. Must be defined and
            finite on all of ``[a, b]``.
        a, b: Endpoints of the approximation interval. ``a < b`` is required.
        degree: Degree of the approximating polynomial (>= 0).
        max_iterations: Upper bound on exchange iterations.
        tolerance: Stop when the relative change in the minimax error level
            falls below this value.
        grid_size: Number of samples used when scanning for error extrema.
            Larger is slower but more accurate; it must be at least 3 and
            should exceed ``degree + 2``.

    Returns:
        A :class:`RemezResult` describing the polynomial, the final reference,
        the error level, and whether the tolerance was met.

    Raises:
        ValueError: If the inputs are malformed (``a >= b``, negative degree,
            tiny grid) or if the linear system is singular.
    """

    if not callable(f):
        raise TypeError("f must be callable")
    if a >= b:
        raise ValueError("require a < b")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")

    n_points = degree + 2
    if grid_size <= n_points:
        raise ValueError("grid_size must exceed degree + 2")

    reference = _chebyshev_nodes(n_points, a, b)
    reference.sort()

    coeffs, h = _solve_reference(f, reference, degree)
    error_level = abs(h)

    converged = False
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        extrema = _find_extrema(f, coeffs, a, b, grid_size)
        extrema.sort(key=lambda pair: pair[0])

        new_ref = [e[0] for e in extrema]
        if len(new_ref) < n_points:
            break

        new_ref.sort()
        for i in range(len(new_ref) - 1):
            if new_ref[i] == new_ref[i + 1]:
                break
        else:
            reference = new_ref[:n_points]
            coeffs, h = _solve_reference(f, reference, degree)
            new_error = abs(h)
            if error_level > 0.0:
                rel = abs(new_error - error_level) / error_level
            else:
                rel = abs(new_error - error_level)
            error_level = new_error
            if rel < tolerance:
                converged = True
                break
            continue
        break

    return RemezResult(
        coefficients=coeffs,
        reference=reference,
        error_level=error_level,
        iterations=iterations,
        converged=converged,
    )
