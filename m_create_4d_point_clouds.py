import numpy as np
import pickle

rand=0.05

def create_4d_sphere(num_points=400, r=1, rand=rand):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    alpha = np.random.uniform(0, 2 * np.pi, num_points)
    
    x = r * np.cos(phi) + np.random.uniform(-rand, rand, num_points)
    y = r * np.sin(phi) * np.cos(theta) + np.random.uniform(-rand, rand, num_points)
    z = r * np.sin(phi) * np.sin(theta) * np.cos(alpha) + np.random.uniform(-rand, rand, num_points)
    w = r * np.sin(phi) * np.sin(theta) * np.sin(alpha) + np.random.uniform(-rand, rand, num_points)

    sphere = np.column_stack((x, y, z, w))
    return sphere



def create_4d_line_segment(num_points=400, r=1, rand=rand):
    x = np.random.uniform(-r/2, r/2, num_points) 
    y = np.random.uniform(-rand/5, rand/5, num_points)
    z = np.random.uniform(-rand/5, rand/5, num_points)
    w = np.random.uniform(-rand/5, rand/5, num_points)
    return np.column_stack((x, y, z, w))


def create_4d_ellipsoid(num_points=400, a=1, b=0.8, c=0.5, d=0.3, rand=rand):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    alpha = np.random.uniform(0, 2 * np.pi, num_points)
    x = a * np.cos(phi) + np.random.uniform(-rand, rand, num_points)
    y = b * np.sin(phi) * np.cos(theta) + np.random.uniform(-rand, rand, num_points)
    z = c * np.sin(phi) * np.sin(theta) * np.cos(alpha) + np.random.uniform(-rand, rand, num_points)
    w = d * np.sin(phi) * np.sin(theta) * np.sin(alpha) + np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x, y, z, w))

def create_4d_4_disc(num_points=400, r=1, rand=rand):
    'polna 4d sfera'
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    alpha = np.random.uniform(0, 2 * np.pi, num_points)
    x = r * np.random.rand(num_points)* np.cos(phi) + np.random.uniform(-rand, rand, num_points)
    y = r * np.random.rand(num_points)* np.sin(phi) * np.cos(theta) + np.random.uniform(-rand, rand, num_points)
    z = r * np.random.rand(num_points)* np.sin(phi) * np.sin(theta) * np.cos(alpha) + np.random.uniform(-rand, rand, num_points)
    w = r * np.random.rand(num_points)* np.sin(phi) * np.sin(theta) * np.sin(alpha) + np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x, y, z, w))

def create_4d_perturbed_4_disc(num_points=400, a=1, b=0.8, c=0.5, d=0.3, rand=rand):
    'krompirjasta polna 4d sfera'
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    alpha = np.random.uniform(0, 2 * np.pi, num_points)
    
    x = a * np.random.rand(num_points) * np.cos(phi) + np.random.uniform(-rand, rand, num_points)
    y = b * np.random.rand(num_points) * np.sin(phi) * np.cos(theta) + np.random.uniform(-rand, rand, num_points)
    z = c * np.random.rand(num_points) * np.sin(phi) * np.sin(theta) * np.cos(alpha) + np.random.uniform(-rand, rand, num_points)
    w = d * np.random.rand(num_points) * np.sin(phi) * np.sin(theta) * np.sin(alpha) + np.random.uniform(-rand, rand, num_points)
    return np.column_stack((x, y, z, w))




def main():
    shape_generators = {
        '4D_sphere': create_4d_sphere,
        '4D_line_segment': create_4d_line_segment,
        '4d_4_disc': create_4d_4_disc,
        '4d_perturbed_4_disc': create_4d_perturbed_4_disc,
        '4d_ellipsoid': create_4d_ellipsoid
    }

    shape_data = []
    label_mapping = {}
    parameters_data = {}
    
    abcd_list = [[1, 0.8, 0.2, 0.4], [1, 0.7, 0.4, 0.4],  [1, 0.6, 0.3, 0.5]]
    r_list = [0.9, 1, 1.1]
   
    for label, generator in enumerate(shape_generators.items()):
        shape_name, shape_func = generator
        label_mapping[shape_name] = label

        for i in range(30):
            n = 400 + (((i//6) - 2) * 50) 
            if shape_name in ['4d_ellipsoid', '4d_perturbed_4_disc']:
                a, b, c, d = abcd_list[i%3]
                if shape_name == '4d_perturbed_4_disc':
                    n += 300
                point_cloud = shape_func(num_points=n, a=a, b=b, c=c, d=d)
                par = (f"num_points: {n}, a={a}, b={b}, c={c}, d={d}")
            else:
                r = r_list[i%3]
                point_cloud = shape_func(num_points=n, r=r)
                par = (f"num_points: {n}, r={r}")
            index = label * 30 + i
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

    print("Shapes and labels saved successfully.")

if __name__ == "__main__":
    main()


