import numpy as np
import pickle

#DEFINING DIFFERENT SHAPES:
#TO DO:verjetno je treba malo spremeniti določene oblike, točke je treba še malo pretresti, trenutno so res narejene da točno
#sledijo obliki, tega nočemo, morajo malo odstopati v vseh treh dimenzijah
#fino bi bilo zgenerirati z različnim številom točk (ne premajhnim, ker bo to pokvarilo model, stvar mora še vedno izgledati kot krogla/črta...),
#preveč točk ne sme biti, ker potem predolgo računa stvari
#spremeniti te radije, da imamo različno velike
#tu je mišljeno da se vse te oblike zgenerira enkrat, hkrati in se jih shrani v spodaj napisano datoteko. 
#Tako da bo treba morda popraviti kak del kode, tako da bo generiralo npr različno velike krogle in črte, ne da bi morali na roko 
#spreminjati ta parameter (torej treba bo neke dodati neke spremenljivke ki bodo določale različne velikosti ipd.)

def create_sphere(num_points=300, radius=1.0):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack((x, y, z))

def create_circle(num_points=300, radius=1.0):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros(num_points)
    return np.column_stack((x, y, z))

def create_line_segment(num_points=100, length=1.0):
    t = np.random.uniform(0, length, num_points)
    return np.column_stack((t, np.zeros(num_points), np.zeros(num_points)))

#tu notri dodaš zadeve, ki jih na novo napišeš, vse te oblike bo generiralo in jih tudi označilo z 0,1,2...glede na obliko
#kako je označena kakšna oblika si lahko pogledaš v label_mapping.txt
def main():
    shape_generators = {
        'sphere': create_sphere,
        'circle': create_circle,
        'line_segment': create_line_segment
    }

    shape_data = []
    label_mapping = {}

    for label, generator in enumerate(shape_generators.items()):
        shape_name, shape_func = generator
        label_mapping[shape_name] = label
        #tu je koliko vsake oblike bo zgeneriralo (to da je vsake oblike enako veliko je dobro za trening modela)
        for _ in range(5):
            point_cloud = shape_func()
            shape_data.append((point_cloud, label))

    with open('Shape_detection/shapes_data.pkl', 'wb') as file:
        pickle.dump(shape_data, file)

    with open('Shape_detection/label_mapping.txt', 'w') as file:
        for shape_name, label in label_mapping.items():
            file.write(f"{shape_name}:{label}\n")

    print("Shapes and labels saved successfully.")

if __name__ == "__main__":
    main()
