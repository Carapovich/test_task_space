import csv
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D

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
        axes.plot(*np.column_stack((point, bias_3d[i, :])), color=clr, linewidth=2)
        axes.text(*bias_3d[i, :], lbl, fontsize=12)

def draw_state(axes, rv_vecs):
    if not hasattr(draw_state, 'xyz_lim'):
        draw_state.xyz_lim = float( la.norm(rv_vecs[:3]) / 1e6 )

    # Очистка и настройка графика
    # axes.clear()
    axes.set_xlim([-2 * draw_state.xyz_lim, 2 * draw_state.xyz_lim])
    axes.set_ylim([-2 * draw_state.xyz_lim, 2 * draw_state.xyz_lim])
    axes.set_zlim([-2 * draw_state.xyz_lim, 2 * draw_state.xyz_lim])
    axes.set_box_aspect([1, 1, 1])
    axes.set_xlabel(r'X, km$\times 10^3$')
    axes.set_ylabel(r'Y, km$\times 10^3$')
    axes.set_zlabel(r'Z, km$\times 10^3$')

    vec_r, vec_v = np.hsplit(rv_vecs, 2)

    draw_bias(axes, (0, 0, 0), (r'$\gamma$', r'$y_I$', r'$z_I$'), 1.4 * draw_state.xyz_lim)
    axes.quiver(0, 0, 0, *(vec_r / 1e6 ), color='black')
    axes.quiver(*(vec_r / 1e6 ), *(vec_v / 1e4), color='red')

    plt.draw()

def draw_together(axes: Axes3D, rv_vecs: np.ndarray):
    v1_r, v1_v, v2_r, v2_v = rv_vecs

    lim_l, lim_r = np.min((v1_r, v2_r), axis=0), np.max((v1_r, v2_r), axis=0)
    center = lim_l + 0.5 * (lim_r - lim_l)
    if not hasattr(draw_together, 'lim_diff'):
        draw_together.lim_diff = np.max(lim_r - lim_l)
    lim_l, lim_r = center - 10 * draw_together.lim_diff, center + 10 * draw_together.lim_diff

    # Настройка графика
    axes.set_xlim([lim_l[0], lim_r[0]])
    axes.set_ylim([lim_l[1], lim_r[1]])
    axes.set_zlim([lim_l[2], lim_r[2]])
    axes.set_box_aspect([1, 1, 1])
    axes.set_xlabel('X, m')
    axes.set_ylabel('Y, m')
    axes.set_zlabel('Z, m')

    draw_bias(axes, lim_l + 0.1 * draw_together.lim_diff, (r'$\gamma$', r'$y_I$', r'$z_I$'), 4 * draw_together.lim_diff)
    axes.scatter(*v1_r, color='black')
    axes.quiver(*v1_r, *(2 * draw_together.lim_diff * -v1_r / la.norm(v1_r)), color='black')
    axes.quiver(*v1_r, *(2 * draw_together.lim_diff * v1_v / la.norm(v1_v)), color='red')
    axes.scatter(*v2_r, color='brown')
    axes.quiver(*v2_r, *(2 * draw_together.lim_diff * -v2_r / la.norm(v2_r)), color='black')
    axes.quiver(*v2_r, *(2 * draw_together.lim_diff * v2_v / la.norm(v2_v)), color='red')

def show_anim(func_draw: Callable, func_arg: list):
    if not hasattr(show_anim, 'fig'):
        show_anim.fig = plt.figure()
    if not hasattr(show_anim, 'axes'):
        show_anim.axes = show_anim.fig.add_subplot(111, projection='3d')

    show_anim.fig.tight_layout()
    total_frames = len(func_arg)

    def anim_update(frame) -> Iterable[Artist]:
        show_anim.axes.clear()
        func_draw(show_anim.axes, func_arg[frame])
        if frame == total_frames - 1:
            anim.event_source.stop()

    anim = FuncAnimation(
        show_anim.fig, anim_update,
        frames=total_frames,
        interval=50,
        blit=False,
        repeat=False
    )
    plt.show()
