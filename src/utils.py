import csv

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

def read_initial_conditions(filename) -> dict:
    with open(filename, mode='r', newline='') as file:
        ic_dict: dict[str, float] = {}
        csv_reader = csv.reader(file, delimiter=';')
        for row in csv_reader:
            ic_dict.update({ row[0]: float(row[1]) })

    return ic_dict

def draw_bias(axes, point, labels, mult = 1., dcm = None):
    bias_3d = np.identity(3) * mult
    if dcm is not None:
        bias_3d = np.dot(dcm, bias_3d)
    bias_3d += point

    for i, (clr, lbl) in enumerate(zip(('r', 'g', 'b'), labels)):
        axes.plot(*np.column_stack((point, bias_3d[:, i])), color=clr, linewidth=2)
        axes.text(*bias_3d[:, i], lbl)

def draw_state(r_eci, v_eci, a_eci):
    # Настройка окна с plot3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Очистка и настройка графика
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    draw_bias(ax, (0, 0, 0), (r'$x_B$', r'$y_B$', r'$z_B$'), 1.4)
    ax.quiver(0, 0, 0, *(r_eci/la.norm(r_eci)), color='black')
    ax.quiver(*(r_eci / la.norm(r_eci)), *(v_eci / la.norm(v_eci)), color='red')
    ax.quiver(*(r_eci / la.norm(r_eci)), *(a_eci / la.norm(a_eci)), color='green')

    plt.show()
