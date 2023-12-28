import numpy as np

def create_sphere(num_points=300, radius=1.0, rand=0.1):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    randx = np.random.uniform(-rand, rand, num_points)
    randy = np.random.uniform(-rand, rand, num_points)
    randz = np.random.uniform(-rand, rand, num_points)
    x = radius * np.sin(phi) * np.cos(theta) + randx
    y = radius * np.sin(phi) * np.sin(theta) + randy
    z = radius * np.cos(phi) + randz
    return np.column_stack((x, y, z))

def create_circle(num_points=300, radius=1.0, rand=0.1):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    randx = np.random.uniform(-rand, rand, num_points)
    randy = np.random.uniform(-rand, rand, num_points)
    randz = np.random.uniform(-rand, rand, num_points)
    x = radius * np.cos(theta) + randx
    y = radius * np.sin(theta) + randy
    z = np.zeros(num_points) + randz
    return np.column_stack((x, y, z))

def create_line_segment(num_points=100, length=1.0, rand=0.1):
    x = np.random.uniform(0, length, num_points) 
    y = np.random.uniform(-rand, rand, num_points)
    z = np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x, y, z))

def create_torus(num_points=300, R=1.0, r=0.3, rand=0.1):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = (R + r * np.cos(phi)) * np.cos(theta) + np.random.uniform(-rand, rand, num_points)
    y = (R + r * np.cos(phi)) * np.sin(theta) + np.random.uniform(-rand, rand, num_points)
    z = r * np.sin(phi) + np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x, y, z))

def create_flat_disc(num_points=300, radius=1.0, rand=0.1):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = radius * np.random.rand(num_points) * np.cos(theta)
    y = radius * np.random.rand(num_points) * np.sin(theta)
    z = np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x, y, z))

def create_ellipsoid(num_points=300, a=1.0, b=0.8, c=0.6):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    x = a * np.sin(phi) * np.cos(theta)
    y = b * np.sin(phi) * np.sin(theta)
    z = c * np.cos(phi) 
    return np.column_stack((x, y, z))

def create_perturbed_3_disc(num_points=300, perturbation=0.5):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, np.pi, num_points)
    x_disc = np.sin(theta) * np.cos(phi)
    y_disc = np.sin(theta) * np.sin(phi)
    z_disc = np.cos(theta)
    # Deform the 3-disc to create a perturbed shape
    x_perturbed = x_disc + perturbation * np.random.normal(size=num_points)
    y_perturbed = y_disc + perturbation * np.random.normal(size=num_points)
    z_perturbed = z_disc + perturbation * np.random.normal(size=num_points)
    return np.column_stack((x_perturbed, y_perturbed, z_perturbed))
