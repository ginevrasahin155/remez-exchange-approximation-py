# remez-exchange-approximation-py

Constructs a best uniform (minimax) polynomial approximation of a scalar
function on a closed interval using the classical multipoint Remez exchange
iteration.

```python
from remez_exchange_approximation_py import remez_exchange

result = remez_exchange(lambda x: x * x, -1.0, 1.0, 2)
print(result.coefficients)  # ascending powers, length degree + 1
print(result.error_level)   # magnitude of the equal ripple
print(result.reference)      # final n+2 exchange points
```

`remez_exchange(f, a, b, degree, *, max_iterations=50, tolerance=1e-12,
grid_size=2001)` returns a `RemezResult` with `coefficients`, `reference`,
`error_level`, `iterations`, and `converged`. Coefficients are ascending-power
(`coefficients[i]` multiplies `x**i`).

## Why this exists

The minimax polynomial is the degree-`n` polynomial that minimises the worst-
case error on an interval. Computing it requires iterating the Remez
exchange: solve for a polynomial whose error alternates sign at `n+2`
reference points, then move the references to the actual extrema of the error.

This library does that with only the standard library. The trade-off is
precision and speed: the linear system is solved by Gaussian elimination
with partial pivoting, and extrema are located by scanning a dense uniform
grid rather than by root-finding on the derivative. That is enough to reach
the default 1e-12 relative tolerance on well-behaved functions on short
intervals, without pulling in numpy or an optimisation package.

## The awkward edge

The grid-based extrema search means the reported `error_level` is the ripple
at the final reference set, while the true maximum error on the interval may
be marginally larger because an extremum can fall between grid samples. If
you need a guarantee, re-evaluate the error on a finer grid yourself and use
that as the figure of merit. Increasing `grid_size` narrows the gap at the
cost of runtime.

The function `f` must be defined and finite on all of `[a, b]`. There is no
sampling fallback for infinities or NaNs.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

## Limitations

Values are coerced to floats, so very large integers lose precision. If you need
exact integer aggregates over a window, this is the wrong tool.

