import numpy as np
import matplotlib.pyplot as plt


def generate_uniform_radial_distribution(num_points, r_min, r_max):
    """
    Generate a uniform radial distribution across a circular area.

    Parameters:
    num_points (int): Number of points to generate.
    r_min (float): Minimum radius of the distribution.
    r_max (float): Maximum radius of the distribution.

    Returns:
    numpy.ndarray: Array of radii representing a uniform radial distribution.
    """
    # Generate uniform random numbers between 0 and 1, then scale them by r_max**2
    random_squares = np.random.default_rng().random() * r_max ** 2
    # Take the square root to get uniform distribution over a circle's area
    radii = np.sqrt(random_squares)

    return radii


# Example usage:
num_points = 1000
r_min = 0
r_max = 10
angle = np.pi / 4  # 45 degrees in radians
radii = []
for i in range(num_points):
    radii.append(generate_uniform_radial_distribution(num_points, r_min, r_max))

# Binning the radii
num_bins = 36
bin_edges = np.linspace(r_min, r_max, num_bins + 1)
bin_counts, _ = np.histogram(radii, bins=bin_edges)

# Prepare to generate the 2D plot
angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)  # Full circle of angles
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

for i in range(num_bins):
    count = bin_counts[i]
    radius = np.mean([bin_edges[i], bin_edges[i + 1]])  # Average radius for the bin

    # Randomly choose 'count' angles from the angle array
    chosen_angles = np.random.choice(angles, size=count, replace=False)

    # Plot each point at the randomly chosen angle with the same radial distance
    ax.scatter(chosen_angles, np.full_like(chosen_angles, radius), alpha=0.5, s=10)

ax.set_title('2D Radial Distribution from 1D Slice Spread Across Angles')
plt.show()

fig, ax = plt.subplots()
ax.hist(radii, bins=36)
ax.grid(True)
plt.show()