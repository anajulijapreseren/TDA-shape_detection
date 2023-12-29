import numpy as np
import pickle
import random

rand = 0.05

def create_4d_sphere(num_points=300, radius=1.0, rand=rand):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    psi = np.random.uniform(0, 2 * np.pi, num_points)
    randw = np.random.uniform(-rand, rand, num_points)
    randx = np.random.uniform(-rand, rand, num_points)
    randy = np.random.uniform(-rand, rand, num_points)
    randz = np.random.uniform(-rand, rand, num_points)
    x = radius * np.sin(phi) * np.cos(theta) + randx
    y = radius * np.sin(phi) * np.sin(theta) + randy
    z = radius * np.cos(phi) + randz
    w = psi + randw
    return np.column_stack((x, y, z, w))

def create_4d_circle(num_points=300, radius=1.0, rand=rand):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    psi = np.random.uniform(0, 2 * np.pi, num_points)
    randw = np.random.uniform(-rand, rand, num_points)
    randx = np.random.uniform(-rand, rand, num_points)
    randy = np.random.uniform(-rand, rand, num_points)
    randz = np.random.uniform(-rand, rand, num_points)
    x = radius * np.cos(theta) + randx
    y = radius * np.sin(theta) + randy
    z = np.zeros(num_points) + randz
    w = psi + randw
    return np.column_stack((x, y, z, w))

def create_4d_line_segment(num_points=300, length=2.0, rand=rand):
    x = np.random.uniform(-length/2, length/2, num_points)
    y = np.random.uniform(-rand, rand, num_points)
    z = np.random.uniform(-rand, rand, num_points)
    w = np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x, y, z, w))

def create_4d_torus(num_points=300, R=1.0, r=0.3, rand=rand):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    psi = np.random.uniform(0, 2 * np.pi, num_points)
    randw = np.random.uniform(-rand, rand, num_points)
    randx = np.random.uniform(-rand, rand, num_points)
    randy = np.random.uniform(-rand, rand, num_points)
    randz = np.random.uniform(-rand, rand, num_points)
    x = (R + r * np.cos(phi)) * np.cos(theta) + randx
    y = (R + r * np.cos(phi)) * np.sin(theta) + randy
    z = r * np.sin(phi) + randz
    w = psi + randw
    return np.column_stack((x, y, z, w))

def create_4d_flat_disc(num_points=300, radius=1.0, rand=rand):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    psi = np.random.uniform(0, 2 * np.pi, num_points)
    randw = np.random.uniform(-rand, rand, num_points)
    randx = np.random.uniform(-rand, rand, num_points)
    randy = np.random.uniform(-rand, rand, num_points)
    randz = np.random.uniform(-rand, rand, num_points)
    x = radius * np.random.rand(num_points) * np.cos(theta) + randx
    y = radius * np.random.rand(num_points) * np.sin(theta) + randy
    z = np.zeros(num_points) + randz
    w = psi + randw
    return np.column_stack((x, y, z, w))

def create_4d_ellipsoid(num_points=300, a=1.0, b=0.6, c=0.5, rand=rand):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    psi = np.random.uniform(0, 2 * np.pi, num_points)
    randw = np.random.uniform(-rand, rand, num_points)
    randx = np.random.uniform(-rand, rand, num_points)
    randy = np.random.uniform(-rand, rand, num_points)
    randz = np.random.uniform(-rand, rand, num_points)
    x = a * np.sin(phi) * np.cos(theta) + randx
    y = b * np.sin(phi) * np.sin(theta) + randy
    z = c * np.cos(phi) + randz
    w = psi + randw
    return np.column_stack((x, y, z, w))

def create_4d_perturbed_3_disc(num_points=300, a=1, b=0.8, c=0.6, rand=rand):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, np.pi, num_points)
    psi = np.random.uniform(0, 2 * np.pi, num_points)
    randw = np.random.uniform(-rand, rand, num_points)
    randx = np.random.uniform(-rand, rand, num_points)
    randy = np.random.uniform(-rand, rand, num_points)
    randz = np.random.uniform(-rand, rand, num_points)
    x_disc = a * np.random.rand(num_points) * np.sin(theta) * np.cos(phi)
    y_disc = b * np.random.rand(num_points) * np.sin(theta) * np.sin(phi)
    z_disc = c * np.random.rand(num_points) * np.cos(theta)
    x_perturbed = x_disc + randx
    y_perturbed = y_disc + randy
    z_perturbed = z_disc + randz
    w = psi + randw
    return np.column_stack((x_perturbed, y_perturbed, z_perturbed, w))

def main():
    shape_generators = {
        '4d_sphere': create_4d_sphere,
        '4d_circle': create_4d_circle,
        '4d_line_segment': create_4d_line_segment,
        '4d_torus': create_4d_torus,
        '4d_flat_disc': create_4d_flat_disc,
        '4d_ellipsoid': create_4d_ellipsoid,
        '4d_perturbed_3_disc': create_4d_perturbed_3_disc
    }

    shape_data = []
    label_mapping = {}
    parameters_data = {}

    # Define lists of parameter values for variation
    abc_list = [[1, 0.8, 0.2], [1, 0.7, 0.4], [1, 0.6, 0.4], [1, 0.6, 0.3]]
    torus_list = [[1, 0.3], [1, 0.4], [1, 0.5], [1, 0.6]]
    other_list = [0.8, 0.9, 1, 1.1]

    for label, generator in enumerate(shape_generators.items()):
        shape_name, shape_func = generator
        label_mapping[shape_name] = label
        for i in range(30):
            n = 400 + (((i//4) - 2) * 50)
            if shape_name == '4d_ellipsoid' or shape_name == '4d_perturbed_3_disc':
                a, b, c = random.choice(abc_list)
                if shape_name == '4d_perturbed_3_disc':
                    n += 300
                point_cloud = shape_func(num_points=n, a=a, b=b, c=c)
                par = (f"num_points: {n}, a={a}, b={b}, c={c}")
            elif shape_name == '4d_torus':
                R, r = random.choice(torus_list)
                point_cloud = shape_func(num_points=n, R=R, r=r)
                par = (f"num_points: {n}, R={R}, r={r}")
            else:
                r = random.choice(other_list)
                point_cloud = shape_func(num_points=n, radius=r)
                par = (f"num_points: {n}, r={r}")
            index = label * 20 + i
            parameters_data[index] = par
            shape_data.append((index, point_cloud, label))

    with open('m_4d_shapes_data.pkl', 'wb') as file:
        pickle.dump(shape_data, file)

    with open('m_4d_label_mapping.txt', 'w') as file:
        for shape_name, label in label_mapping.items():
            file.write(f"{shape_name}:{label}\n")

    with open('m_4d_parameters_data.txt', 'w') as file:
        for index, data in parameters_data.items():
            file.write(f"{index}: {data}\n")

    print("4D shapes and labels saved successfully.")

if __name__ == "__main__":
    main()


