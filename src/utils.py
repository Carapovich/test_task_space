import csv

import matplotlib.pyplot as plt
import numpy as np

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

def show_bias_in_ecef(dcm_body2ecef):
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

    draw_bias(ax, (0, 0, 0), (r'$x_B$', r'$y_B$', r'$z_B$'), 1.4, dcm_body2ecef)
    draw_bias(ax, (-2, -2, -2), (r'$x_W$', r'$y_W$', r'$z_W$'))
    plt.show()
