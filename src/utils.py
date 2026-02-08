import csv
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D

def read_initial_conditions(filename) -> dict:
    with open(filename, mode='r', newline='') as file:
        ic_dict: dict[str, float] = {}
        csv_reader = csv.reader(file, delimiter=';')
        for row in csv_reader:
            ic_dict.update({ row[0]: float(row[1]) })

    return ic_dict


def get_with_length(vector: np.ndarray, length: float) -> np.ndarray:
    return vector / la.norm(vector) * length


# Определение пределов и геометрического центра области траекторий объектов
def get_geom_anchors(vectors: np.ndarray) -> tuple:
    v_max, v_min = np.max(vectors, axis=1), np.min(vectors, axis=1)
    d_lims = v_max - v_min

    return v_max, v_min, d_lims / 1.9, max(d_lims), v_min + d_lims / 2.


def plot_vector(time: np.ndarray, vector_data,
                plot_name: str, titles: tuple, y_labels: tuple,
                subplot_order=None, single_scale_y=True, add_plotting: Callable=None, args=None):

    fig = plt.figure(num=plot_name)
    fig.suptitle(plot_name)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    if not isinstance(vector_data, list):
        vector_data = [vector_data]
    if subplot_order is None:
        subplot_order = (len(vector_data), 3)

    for i, vector in enumerate(vector_data):
        diff_lim, cntr = 0, 0
        if single_scale_y:
            diff_lim = max(np.max(vector, axis=1) - np.min(vector, axis=1)) / 1.95
            cntr = np.mean(vector, axis=1)

        for j, clr in enumerate(("r", "g", "b")):
            axes = fig.add_subplot(*subplot_order, 3 * i + j + 1)
            axes.plot(time, vector[j, :], color=clr, linewidth=2)
            axes.grid(True, which='both', linestyle='--')
            axes.set(xlabel='t, сек', ylabel=y_labels[3 * i + j], title=titles[3 * i + j])

            if single_scale_y:
                axes.set_ylim((cntr[j] - diff_lim, cntr[j] + diff_lim))
            if add_plotting is not None:
                add_plotting(axes, i, j, *args)


def plot_vector_with_ref_line(time: np.ndarray, vector_data: list[np.ndarray], ref_t: np.ndarray, ref_y,
                              plot_name: str, titles: tuple, y_labels: tuple):

    def plot_ref_line(axes, i, j, t, y):
        axes.plot(t, y[i][j, :], ls='--', color='black', linewidth=1, zorder=0)
        axes.legend(('Опорная прямая', titles[3 * i + j]))

    if not isinstance(ref_y, list):
        ref_y = [ref_y]

    plot_vector(time, vector_data, plot_name, titles, y_labels, add_plotting=plot_ref_line, args=(ref_t, ref_y))


def draw_bias(axes, point, labels, mult = 1.):
    bias_3d = np.identity(3) * mult
    bias_3d += point

    for i, (clr, lbl) in enumerate(zip(('r', 'g', 'b'), labels)):
        axes.plot(*np.column_stack((point, bias_3d[i, :])), color=clr, linewidth=2)
        axes.text(*bias_3d[i, :], lbl, fontsize=12)


def plot_vehicles_trajectory(figure: Figure, t: np.ndarray, vehicles_vectors: np.ndarray, frame=None):
    v1_r, v1_v, v2_r, v2_v = np.vsplit(vehicles_vectors, 4)

    if frame is None:
        frame = v1_r.shape[1] - 1

    if not hasattr(plot_vehicles_trajectory, 'figures_gnrl'):
        v_max, v_min, d_lims, d_lim_max, cntr = get_geom_anchors(np.hstack((v1_r, v2_r)))

        # Создание и настройка графика общего вида траектории
        axes_gnrl = figure.add_subplot(1, 2, 1, projection='3d')
        axes_gnrl.set(
            title='Общий вид траектории',
            xlim=(cntr[0] - d_lims[0], cntr[0] + d_lims[0]),
            ylim=(cntr[1] - d_lims[1], cntr[1] + d_lims[1]),
            zlim=(cntr[2] - d_lims[2], cntr[2] + d_lims[2]),
            box_aspect=(v_max - v_min) / d_lim_max, xlabel='X, метры', ylabel='Y, метры', zlabel='Z, метры')
        draw_bias(axes_gnrl, v_min + 1e-3 * d_lim_max, (r'$x_I$', r'$y_I$', r'$z_I$'), 0.05 * d_lim_max)

        plot_vehicles_trajectory.figures_gnrl = [
            axes_gnrl.plot(*(v1_r[:, 0]), color='k', linestyle='--', label='Траектория РН')[0],
            axes_gnrl.plot(*(v2_r[:, 0]), color='r', linestyle=':', label='Траектория КА')[0],
            axes_gnrl.scatter(*(v1_r[:, 0]), color='k'), axes_gnrl.scatter(*(v2_r[:, 0]), color='r')]
        axes_gnrl.legend(handles=plot_vehicles_trajectory.figures_gnrl[:2])

        # Создание графика приближенного вида траектории
        plot_vehicles_trajectory.axes_clsup = figure.add_subplot(1, 2, 2, projection='3d')
        plot_vehicles_trajectory.d_lim = np.max(abs(v1_r - v2_r)) / 1.9

    # Отрисовываем график общего вида
    traj1, traj2, p1, p2 = plot_vehicles_trajectory.figures_gnrl
    traj1.set_data_3d(*(v1_r[:, :frame + 1]))
    traj2.set_data_3d(*(v2_r[:, :frame + 1]))
    p1._offsets3d = tuple(np.split(v1_r[:, frame], 3))
    p2._offsets3d = tuple(np.split(v2_r[:, frame], 3))

    # Отрисовываем график приближенного вида
    d_lim = plot_vehicles_trajectory.d_lim
    v1_r, v1_v, v2_r, v2_v = np.split(vehicles_vectors[:, frame], 4)
    _, v_min, _, _, cntr = get_geom_anchors(np.vstack((v1_r, v2_r)).T)

    axes = plot_vehicles_trajectory.axes_clsup
    axes.clear()
    axes.set(
        title=rf'Приближенный вид траектории, t={t[frame]:.2f} сек.',
        xlim=(cntr[0] - d_lim, cntr[0] + d_lim),
        ylim=(cntr[1] - d_lim, cntr[1] + d_lim),
        zlim=(cntr[2] - d_lim, cntr[2] + d_lim),
        box_aspect=(1, 1, 1), xlabel='X, метры', ylabel='Y, метры', zlabel='Z, метры')

    draw_bias(axes, cntr - 0.95 * d_lim, (r'$x_I$', r'$y_I$', r'$z_I$'), 0.2 * d_lim)
    axes.scatter(*v1_r, color='k', marker='*', s=50)
    axes.text(*(v1_r + 1e-2 * d_lim), s='РН')
    axes.scatter(*v2_r, color='r', marker='*', s=50)
    axes.text(*(v2_r + 1e-2 * d_lim), s='КА')
    axes.quiver(*v1_r, *get_with_length(v1_v, 0.2 * d_lim), color='b')
    axes.text(*(v1_r + get_with_length(v1_v, 0.2 * d_lim) + 1e-2 * d_lim), s=rf'$\bf |v|$={la.norm(v1_v):.2f} м/с')
    axes.quiver(*v2_r, *get_with_length(v2_v, 0.2 * d_lim), color='b')
    axes.text(*(v2_r + get_with_length(v2_v, 0.2 * d_lim) + 1e-2 * d_lim), s=rf'$\bf |v|$={la.norm(v2_v):.2f} м/с')


def show_anim(figure: Figure, func_draw: Callable, frame_hz, t, y):
    anim = FuncAnimation
    anim_paused = True
    cur_frame = 0
    total_frames = int((t[-1] - t[0]) * frame_hz) + 1
    rel_hz = int(1. / (t[1] - t[0]) / frame_hz)

    def anim_update(frame) -> Iterable[Artist]:
        nonlocal anim_paused, cur_frame

        func_draw(figure, t, y, rel_hz * cur_frame)
        if cur_frame == total_frames - 1:
            cur_frame, anim_paused = 0, True
            anim.event_source.stop()
        else:
            cur_frame += 1

    def stop_and_redraw(new_cur_frame):
        nonlocal anim_paused, cur_frame

        anim_paused = True
        anim.event_source.stop()
        cur_frame = np.clip(new_cur_frame, 0, total_frames - 1)
        func_draw(figure, t, y, rel_hz * cur_frame)
        buttons[2].label.set_text(buttons_labels[2])
        plt.draw()

    def on_button_play_clicked(event):
        nonlocal anim, anim_paused, cur_frame, rel_hz

        if anim_paused:
            anim_paused = False
            if cur_frame == total_frames - 1:
                cur_frame = 0

            anim = FuncAnimation(
                figure, anim_update,
                frames=total_frames - cur_frame,
                interval=50,
                blit=False,
                repeat=False
            )
            buttons[2].label.set_text(buttons_labels[-1])
            plt.draw()
        else:
            stop_and_redraw(cur_frame)

    buttons_h_pos = (0.450, 0.475, 0.500, 0.525, 0.550)
    buttons_labels = ['⏮', '', '▶', '', '⏭', '']
    buttons_labels_v_align = ['center', 'center_baseline', 'center_baseline', 'center_baseline', 'center']
    buttons: list[Button] = []

    for h_pos, label, v_align in zip(buttons_h_pos, buttons_labels, buttons_labels_v_align):
        buttons.append(Button(figure.add_axes((h_pos, 0.01, 0.02, 0.04)), label))
        buttons[-1].label.set(fontfamily="Segoe UI Symbol", fontsize=16, verticalalignment=v_align)

    buttons[0].on_clicked(lambda e: stop_and_redraw(0))
    buttons[1].on_clicked(lambda e: stop_and_redraw(cur_frame - 1))
    buttons[2].on_clicked(on_button_play_clicked)
    buttons[3].on_clicked(lambda e: stop_and_redraw(cur_frame + 1))
    buttons[4].on_clicked(lambda e: stop_and_redraw(total_frames - 1))

    func_draw(figure, t, y, rel_hz * cur_frame)
    plt.show()
