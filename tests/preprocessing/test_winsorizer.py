import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):
    """
    Test the Winsorizer transformer with different quantile settings.
    """
    # Generate random data from a normal distribution
    X = np.random.normal(0, 1, 1000)

    # Initialize the Winsorizer with the given quantiles
    winsorizer = Winsorizer(lower_quantile=lower_quantile, upper_quantile=upper_quantile)

    # Fit the Winsorizer to the data
    winsorizer.fit(X)

    # Ensure the quantiles are correctly computed
    expected_lower_quantile = np.quantile(X, lower_quantile)
    expected_upper_quantile = np.quantile(X, upper_quantile)

    # Check if computed quantiles match expected values
    assert np.isclose(winsorizer.lower_quantile_, expected_lower_quantile), (
        f"Expected lower quantile: {expected_lower_quantile}, "
        f"but got: {winsorizer.lower_quantile_}"
    )
    assert np.isclose(winsorizer.upper_quantile_, expected_upper_quantile), (
        f"Expected upper quantile: {expected_upper_quantile}, "
        f"but got: {winsorizer.upper_quantile_}"
    )

    # Transform the data
    X_transformed = winsorizer.transform(X)

    # Check that all values are clipped within the quantile range
    assert np.all(X_transformed >= expected_lower_quantile), (
        f"Values below lower quantile: {X_transformed[X_transformed < expected_lower_quantile]}"
    )
    assert np.all(X_transformed <= expected_upper_quantile), (
        f"Values above upper quantile: {X_transformed[X_transformed > expected_upper_quantile]}"
    )

    # Special case: If lower and upper quantiles are the same, all values should be clipped to that quantile
    if lower_quantile == upper_quantile:
        assert np.all(X_transformed == expected_lower_quantile), (
            f"Expected all values to be {expected_lower_quantile}, "
            f"but got different values."
        )
