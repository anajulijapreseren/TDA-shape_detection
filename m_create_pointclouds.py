import numpy as np
import pickle

rand = 0.05

def create_sphere(num_points=300, radius=1.0, rand=rand):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    randx = np.random.uniform(-rand, rand, num_points)
    randy = np.random.uniform(-rand, rand, num_points)
    randz = np.random.uniform(-rand, rand, num_points)
    x = radius * np.sin(phi) * np.cos(theta) + randx
    y = radius * np.sin(phi) * np.sin(theta) + randy
    z = radius * np.cos(phi) + randz
    return np.column_stack((x, y, z))

def create_circle(num_points=300, radius=1.0, rand=rand):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    randx = np.random.uniform(-rand, rand, num_points)
    randy = np.random.uniform(-rand, rand, num_points)
    randz = np.random.uniform(-rand, rand, num_points)
    x = radius * np.cos(theta) + randx
    y = radius * np.sin(theta) + randy
    z = np.zeros(num_points) + randz
    return np.column_stack((x, y, z))

def create_line_segment(num_points=300, radius=1.0, rand=rand):
    x = np.random.uniform(-radius/2, radius/2, num_points) 
    y = np.random.uniform(-rand, rand, num_points)
    z = np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x, y, z))

def create_torus(num_points=300, R=1.0, r=0.3, rand=rand):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = (R + r * np.cos(phi)) * np.cos(theta) + np.random.uniform(-rand, rand, num_points)
    y = (R + r * np.cos(phi)) * np.sin(theta) + np.random.uniform(-rand, rand, num_points)
    z = r * np.sin(phi) + np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x, y, z))

def create_flat_disc(num_points=300, radius=1.0, rand=rand):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = radius * np.random.rand(num_points) * np.cos(theta)
    y = radius * np.random.rand(num_points) * np.sin(theta)
    z = np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x, y, z))

def create_ellipsoid(num_points=300, a=1.0, b=0.6, c=0.5, rand=rand):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    x = a * np.sin(phi) * np.cos(theta) + np.random.uniform(-rand, rand, num_points)
    y = b * np.sin(phi) * np.sin(theta) + np.random.uniform(-rand, rand, num_points)
    z = c * np.cos(phi) + np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x, y, z))


def create_perturbed_3_disc(num_points=300, a=1, b=0.8, c=0.6, rand=rand):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, np.pi, num_points)
    x_disc = a * np.random.rand(num_points) * np.sin(theta) * np.cos(phi)
    y_disc = b * np.random.rand(num_points) * np.sin(theta) * np.sin(phi)
    z_disc = c * np.random.rand(num_points) * np.cos(theta)
    # Deform to create a perturbed shape
    x_perturbed = x_disc + np.random.uniform(-rand, rand, num_points)
    y_perturbed = y_disc + np.random.uniform(-rand, rand, num_points)
    z_perturbed = z_disc + np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x_perturbed, y_perturbed, z_perturbed))






#tu notri dodaš zadeve, ki jih na novo napišeš, vse te oblike bo generiralo in jih tudi označilo z 0,1,2...glede na obliko
#kako je označena kakšna oblika si lahko pogledaš v label_mapping.txt

def main():
    shape_generators = {
        'sphere': create_sphere,
        'circle': create_circle,
        'line_segment': create_line_segment,
        'torus': create_torus,
        'flat_disc': create_flat_disc,
        'ellipsoid': create_ellipsoid,
        'perturbed_3_disc': create_perturbed_3_disc

    }

    shape_data = []
    label_mapping = {}
    parameters_data = {}
    
    abc_list = [[1, 0.8, 0.2], [1, 0.7, 0.4], [1, 0.6, 0.4], [1, 0.6, 0.3]]
    torus_list = [[1, 0.3], [1, 0.4], [1, 0.5], [1, 0.6]]
    other_list = [0.8, 0.9, 1, 1.1]

    for label, generator in enumerate(shape_generators.items()):
        shape_name, shape_func = generator
        label_mapping[shape_name] = label
        #tu je koliko vsake oblike bo zgeneriralo (to da je vsake oblike enako veliko je dobro za trening modela)
        for i in range(20):
            n = 400 + (((i//4) - 2) * 50) 
            if shape_name == 'ellipsoid' or shape_name == 'perturbed_3_disc':
                a, b, c = abc_list[i%4]
                point_cloud = shape_func(num_points=n, a=a, b=b, c=c)
                par = (f"num_points: {n}, a={a}, b={b}, c={c}")
            elif shape_name == 'torus':
                R, r = torus_list[i%4]
                point_cloud = shape_func(num_points=n, R=R, r=r)
                par = (f"num_points: {n}, R={R}, r={r}")
            else:
                r = other_list[i%4]
                point_cloud = shape_func(num_points=n, radius=r)
                par = (f"num_points: {n}, r={r}")
            index = label * 20 + i
            parameters_data[index] = par
            shape_data.append((index, point_cloud, label))  
            


    with open('m_shapes_data.pkl', 'wb') as file:
        pickle.dump(shape_data, file)

    with open('m_label_mapping.txt', 'w') as file:
        for shape_name, label in label_mapping.items():
            file.write(f"{shape_name}:{label}\n")
    
    with open('m_parameters_data.txt', 'w') as file:
        for index, data in parameters_data.items():
            file.write(f"{index}: {data}\n")

    print("Shapes and labels saved successfully.")

if __name__ == "__main__":
    main()
