import math
import re
import numpy as np
import sympy as sp

file_path = 'Documents/Técnico/MMCom/trabalho1_7.txt'  #write the path to your file

with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

E = float(lines[5].split()[2])
A = float(lines[6].split()[2])

E_Pas = E * (10**9)
A_m2 = A * (10**-6)

#-----print-dos-valores-lidos-------
print("E =", E, "GPa")
print("A =", A, "mm^2")
print('\n')
#-----------------------------------

coordinates_start_line = lines.index('Coordenadas\n') + 1
matrix_start_line = lines.index('Matriz de elementos\n') + 1
frontier_start_line = lines.index('Condições Fronteira\n') + 1
forces_start_line = lines.index('Forças Aplicadas\n') + 1


coordinates = {}
for line in lines[coordinates_start_line:matrix_start_line]:
    match = re.match(r'([0-9.]+)\s+x\s+=\s+([0-9.]+)\s+;\s+y\s+=\s+([0-9.]+)', line)
    if match:
        label, x, y = match.groups()
        coordinates[label] = (float(x), float(y))

print(coordinates)
print('\n')

matrix_lines = []
for line in lines[matrix_start_line:frontier_start_line]:
    if any(char.isdigit() for char in line):
        matrix_lines.append(line.strip().split())
    else:
        break 

matrix = np.array([[int(element) for element in line] for line in matrix_lines])

print(matrix)
print('\n')

frontier_values = []
for line in lines[frontier_start_line:forces_start_line]:
    if any(char.isdigit() for char in line):
        frontier_line = line.strip().split()[1:]
        frontier_values.extend(frontier_line)
    else:
        break

frontiers = [int(value) for value in frontier_values]

displacments = [sp.symbols('u{}'.format(i)) for i, value in enumerate(frontiers) if value == 1]
displacments_vector = np.array([displacments.pop(0) if value == 1 else 0 for value in frontiers])


reactions = [1 if value == 0 else 0 for value in frontiers]

reactions2 = [sp.symbols('R{}'.format(i)) for i, value in enumerate(reactions) if value == 1]
reactions_vector = [reactions2.pop(0) if value == 1 else 0 for value in reactions]

print(displacments_vector)
print('\n')


forces_values = []
for line in lines[forces_start_line:]:
    if any(char.isdigit() or char == '-' for char in line):
        forces_line = line.strip().split()[1:]
        forces_values.extend(forces_line)
    else:
        break

forces = [int(value) for value in forces_values]

forces_vector = np.array(forces) * 10**3  + np.array(reactions_vector)

print(forces_vector)
print('\n')


def calculate_distance(coordinates, point1, point2):
    x1, y1 = coordinates[point1]
    x2, y2 = coordinates[point2]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def calculate_angle(coordinates, joint1, joint2):
    x1, y1 = coordinates[joint1]
    x2, y2 = coordinates[joint2]
    angle = np.arctan2(y2 - y1, x2 - x1)
    
    if angle > np.pi / 2:
        angle -= np.pi
    elif angle <= -np.pi / 2:
        angle += np.pi
    
    return angle


def calculate_stiffness_matrix(joint1, joint2):

    theta = calculate_angle(coordinates, joint1, joint2)

    c = np.cos(theta)
    s = np.sin(theta)

    k_upper_triangle = np.array( [ [c**2,  c*s , -c**2,  -c*s ],
                                   [  0 , s**2 , -c*s , -s**2 ],
                                   [  0 ,   0  , c**2 ,   c*s ],
                                   [  0 ,   0  ,   0  ,  s**2 ] ] )
    
    k_upper_triangle = np.round(k_upper_triangle, decimals=5)

    h = calculate_distance(coordinates, joint1, joint2)
    k_upper_triangle *= E_Pas * A_m2 / h

    
    print(h)
    print(theta * (180 / math.pi))
    k_local = k_upper_triangle + np.triu(k_upper_triangle, k=1).T

    return k_local



def calculate_big_stiffness_matrix(matrix):

    big_K = [[0 for _ in range(len(matrix)*2)] for _ in range(len(matrix)*2)]

    for i in range(len(matrix)):
        for j in range(i + 1 , len(matrix)):

            if matrix[i][j] == 1:

                joint1 = str(i + 1)
                joint2 = str(j + 1)
                small_k = calculate_stiffness_matrix(joint1, joint2)
                print(small_k)
                print('\n')

                big_K[(i+1)*2-2][(i+1)*2-2] += small_k[0][0]
                big_K[(i+1)*2-2][(i+1)*2-1] += small_k[0][1]
                big_K[(i+1)*2-2][(j+1)*2-2] += small_k[0][2]
                big_K[(i+1)*2-2][(j+1)*2-1] += small_k[0][3]
                big_K[(i+1)*2-1][(i+1)*2-2] += small_k[1][0]
                big_K[(i+1)*2-1][(i+1)*2-1] += small_k[1][1]
                big_K[(i+1)*2-1][(j+1)*2-2] += small_k[1][2]
                big_K[(i+1)*2-1][(j+1)*2-1] += small_k[1][3]
                big_K[(j+1)*2-2][(i+1)*2-2] += small_k[2][0]
                big_K[(j+1)*2-2][(i+1)*2-1] += small_k[2][1]
                big_K[(j+1)*2-2][(j+1)*2-2] += small_k[2][2]
                big_K[(j+1)*2-2][(j+1)*2-1] += small_k[2][3]
                big_K[(j+1)*2-1][(i+1)*2-2] += small_k[3][0]
                big_K[(j+1)*2-1][(i+1)*2-1] += small_k[3][1]
                big_K[(j+1)*2-1][(j+1)*2-2] += small_k[3][2]
                big_K[(j+1)*2-1][(j+1)*2-1] += small_k[3][3]
    
    return big_K

big_K = np.array(calculate_big_stiffness_matrix(matrix))

for row in big_K:
    print(row)
print('\n')

def transform_to_equations(A, X, B):
    equations = []
    for i in range(len(A)):
        equation = sp.Eq(sum(A[i][j] * X[j] for j in range(len(X))), B[i])
        equations.append(equation)
    return equations


equations = transform_to_equations(big_K, displacments_vector, forces_vector)

for row in equations:
    print(row)
print('\n')

def solve_equations(A, X, B):
    equations = transform_to_equations(A, X, B)
    
    X_unknown = [x for x in X if isinstance(x, sp.Symbol)]
    B_unknown = [b for b in B if isinstance(b, sp.Symbol)]
       
    solutions = sp.solve(equations, X_unknown + B_unknown)
    
    solved_X = [solutions[x] if x in X_unknown else x for x in X]
    solved_B = [solutions[b] if b in B_unknown else b for b in B]
    
    return solved_X, solved_B

displacment_solutions, forces_solution = solve_equations(big_K, displacments_vector, forces_vector)

print(displacment_solutions)
print('\n')
print(forces_solution)
print('\n')