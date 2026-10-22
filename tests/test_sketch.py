import numpy as np

from src.sketch import Sketch


def test_rademacher_shape():
    """Check output shape for a Rademacher sketch."""
    n, k, m = 100, 20, 5
    W = np.random.default_rng(42).standard_normal((n, m))
    sketch = Sketch("rademacher", k=k, n=n, seed=42)
    S = sketch.apply(W)
    assert S.shape == (k, m)
    print("PASS: Rademacher output shape")


def test_srht_shape():
    """Check output shape for an SRHT sketch."""
    n, k, m = 100, 20, 5
    W = np.random.default_rng(42).standard_normal((n, m))
    sketch = Sketch("srht", k=k, n=n, seed=42)
    S = sketch.apply(W)
    assert S.shape == (k, m)
    print("PASS: SRHT output shape")


def test_reproducibility():
    """Same seed and input should produce the same sketch."""
    n, k, m = 64, 16, 4
    W = np.random.default_rng(1).standard_normal((n, m))
    for kind in ("rademacher", "srht"):
        sketch1 = Sketch(kind, k=k, n=n, seed=123)
        sketch2 = Sketch(kind, k=k, n=n, seed=123)
        assert np.allclose(sketch1.apply(W), sketch2.apply(W))
        print(f"PASS: {kind} reproducibility")


def test_invalid_dimensions():
    """Invalid dimensions should raise ValueError."""
    try:
        Sketch("srht", k=0, n=100)
    except ValueError:
        print("PASS: Invalid sketch dimension rejected")
    else:
        raise AssertionError("Expected ValueError for k=0")

    try:
        Sketch("srht", k=10, n=100).apply(np.ones((50, 3)))
    except ValueError:
        print("PASS: Invalid input row count rejected")
    else:
        raise AssertionError("Expected ValueError for mismatched rows")


def test_srht_implicit():
    """SRHT should not materialize an explicit Theta matrix."""
    sketch = Sketch("srht", k=16, n=64, seed=42)
    try:
        _ = sketch.Theta
    except AttributeError:
        print("PASS: SRHT remains implicit")
    else:
        raise AssertionError("SRHT should not expose an explicit Theta")


if __name__ == "__main__":
    test_rademacher_shape()
    test_srht_shape()
    test_reproducibility()
    test_invalid_dimensions()
    test_srht_implicit()
    print("\nAll sketch tests passed!")