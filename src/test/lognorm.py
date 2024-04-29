import matplotlib.pyplot as plt
import numpy as np

def test_lognormal():
    def lognormal(median, sigma, shift, min_val, max_val):
        while True:
            # Adjusted for the Python function - this assumes median is e^mu
            z = np.log(median)
            x = z + sigma * np.sqrt(-2 * np.log(np.random.rand()))
            value = np.exp(x) + shift  # Shift the distribution
            # Check if the generated value is within the specified range
            if min_val <= value <= max_val:
                return value

    # Example usage:
    # This would generate a single value from the adjusted lognormal distribution.
    values = []
    for i in range(5000):
        values.append(lognormal(median=np.exp(2), sigma=2.5, shift=1000-20, min_val=1000-100, max_val=1000+1200))
    fig, ax = plt.subplots()
    ax.hist(values, bins=500)
    plt.show()

test_lognormal()