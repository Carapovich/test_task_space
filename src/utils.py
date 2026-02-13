"""
utils.py

Набор утилит чтения входных данных
и печати выходных данных программы моделирования
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from typing import Callable
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D


def read_input(filename: str) -> dict[str, float]:
    """
    Считывает входные данные из csv-файла по указанному пути ``filename``
    и возвращает словарь, соответствующий таблице исходных данных::

        {имя параметра: значение}

    ----------------

    `Прим.: В качестве разделителя в csv-файле используется точка с запятой`

    ----------------

    :param filename: Путь до csv-файла c входными данными
    :return: Словарь, где ключ – имя входного параметра, а значение – его величина
    """
    with open(filename, mode='r', newline='') as file:
        ic_dict: dict[str, float] = {}
        csv_reader = csv.reader(file, delimiter=';')
        for row in csv_reader:
            ic_dict.update({row[0]: float(row[1])})

    return ic_dict


def print_results(filename: str, results: list[dict]):
    """
    Печатает результаты моделирования, собранные в списке словарей
    ``results`` в csv-файл по пути, указанному в ``filename``

    ----------------

    В качестве шапки таблицы используется множество ключей словаря.
    Для каждого элемента списка ``results`` набор ключей одинаков.
    Каждый элемент списка ``results`` – это отдельная строка csv-таблицы.

    `Прим.: В качестве разделителя в csv-файле используется точка с запятой`

    ----------------

    :param filename: Путь до печатаемого csv-файла c результатами моделирования
    :param results: Список словарей с результатами моделирования
    """

    with open(filename, mode='w', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=list(results[0].keys()), delimiter=';')
        csv_writer.writeheader()
        csv_writer.writerows(results)


def plot_vector(time: np.ndarray, vector_data, plot_name: str, titles: list[str], y_labels: list[str],
                subplot_order: tuple[int, int] = None, single_scale_y=True, add_plotting: Callable = None, args=None):
    """
    Выводит окно названия ``plot_name`` с графиками функций
    проекций векторов ``vector_data`` от времени ``time``

    ----------------

    Параметр ``vector_data`` принимает как матрицу 3xN, где N - кол-во моментов времени, для которых
    строится график проекций (каждый столбец матрицы соответствует значению вектора в момент времени),
    так и список таких матриц 3xN. В случае списка матриц, каждый элемент этого списка строится в
    окне графиков в отдельном столбце/строке.

    Каждая проекция вектора строится на отдельном подграфике. Параметр ``subplot_order`` определяет
    количество строк и столбцов подграфиков в окне. По умолчанию ``subplot_order`` задан так, чтобы
    каждый элемент списка ``vector_data`` выводился тремя проекциями в одной строке.
    Заголовки подграфиков указываются в параметре ``titles``. Подписи осей ординат подграфиков задаются
    параметром ``y_labels``. Размерность, определяемая ``subplot_order`` должна совпадать с кол-вом
    элементов в ``titles`` и ``y_labels``.

    Режим единого масштаба осей для проекций одного вектора определяется булевым параметром ``single_scale_y``
    (по умолчанию включен). Помимо единого масштаба, центр оси ординат задается так, чтобы графики проекций
    были отцентрованы по горизонтали.

    Параметр ``add_plotting`` принимает функцию вида::

        add_plotting(axes: Axes, i: int, j: int, *args)

    которая выполняется после построения каждого подграфика проекции ``axes`` и используется для
    дополнительных операций отрисовки на подграфике ``j``-ой проекции ``i``-го вектора.

    ----------------

    :param time: Массив моментов времени функций проекций векторов
    :param vector_data: Список матриц 3xN или одна матрица 3xN со значениями проекций вектора
    :param plot_name: Название графика виде строки, выводится как заголовок окна
    :param titles: Список строк заголовков подграфиков
    :param y_labels: Список строк подписей осей ординат подграфиков
    :param subplot_order: Количество строк и столбцов подграфиков в окне
    :param single_scale_y: (по умолчанию True) Флаг включения режима единого масштаба осей
    :param add_plotting: (по умолчанию None) Функция дополнительных построений на подграфике
    :param args: (по умолчанию None) Варьируемые аргументы функции дополнительных построений
    """
    fig = plt.figure(num=plot_name)
    fig.suptitle(plot_name)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # Задаем список, если передана матрица и выставляем порядок отрисовки
    if not isinstance(vector_data, list):
        vector_data = [vector_data]
    if subplot_order is None:
        subplot_order = (len(vector_data), 3)

    # Перебор векторов
    for i, vector in enumerate(vector_data):
        diff_lim, cntr = 0, 0
        if single_scale_y:
            diff_lim = max(np.max(vector, axis=1) - np.min(vector, axis=1)) / 1.95
            cntr = np.mean(vector, axis=1)

        # Перебор проекций
        for j, clr in enumerate(("r", "g", "b")):
            axes = fig.add_subplot(*subplot_order, 3 * i + j + 1)
            axes.plot(time, vector[j, :], color=clr, linewidth=2)
            axes.grid(True, which='both', linestyle='--')
            axes.set(xlabel='t, сек', ylabel=y_labels[3 * i + j], title=titles[3 * i + j])

            # Центровка по горизонтали
            if single_scale_y:
                axes.set_ylim((cntr[j] - diff_lim, cntr[j] + diff_lim))

            if add_plotting is not None:
                add_plotting(axes, i, j, *args)


def plot_vector_with_ref_line(time: np.ndarray, vector_data, ref_t: np.ndarray, ref_y,
                              plot_name: str, titles: list[str], y_labels: list[str]):
    """
    Вариация функции::

        plot_vector(time, vector_data, plot_name, titles, y_labels, subplot_order, single_scale_y, add_plotting, args)

    которая дополнительно для каждого подграфика строит опорную прямую для явной
    демонстрации, что выводимая на подграфике функция не является линеной

    ----------------

    Параметры ``ref_t`` и ``ref_y`` определяют график функции опорной прямой для каждого подграфика.
    Координаты опорной прямой для одной проекции определяются двумя значениями.

    ----------------

    :param time: Массив моментов времени функций проекций векторов
    :param vector_data: Список матриц 3xN или одна матрица 3xN со значениями проекций вектора
    :param ref_t: Массив из двух элементов моментов времени функций опорных прямых
    :param ref_y: Список матриц 3x2 или матрица 3x2, где каждая строка – две точки, определяющие опорную прямую
    :param plot_name: Название графика виде строки, выводится как заголовок окна
    :param titles: Список строк заголовков подграфиков
    :param y_labels: Список строк подписей осей ординат подграфиков
    """

    # Функция дополнительного построения. Строит опорную прямую и добавляет легенду на подграфик
    def plot_ref_line(axes, i, j, t, y):
        axes.plot(t, y[i][j, :], ls='--', color='black', linewidth=1, zorder=0)
        axes.legend((titles[3 * i + j], 'Опорная прямая'))

    if not isinstance(ref_y, list):
        ref_y = [ref_y]

    plot_vector(time, vector_data, plot_name, titles, y_labels, add_plotting=plot_ref_line, args=(ref_t, ref_y))


def plot_bias(axes: Axes3D, point: np.ndarray,
              labels: tuple[str, str, str] = (r'$x_I$', r'$y_I$', r'$z_I$'), mult: float = 1.):
    """
    Выводит графическое представление ортонормированного базиса на пространственном
    графике ``axes`` с началом координат в точке ``point``, подписями на концах осей
    базиса ``labels`` и длиной осей ``mult``

    ----------------

    :param axes: Объект пространственного графика
    :param point: Координата начала координат базиса на пространственном графике
    :param labels: (по умолчанию задано) Подписи концов осей базиса
    :param mult: (по умолчанию 1) Длина осей базиса
    """
    bias_3d = np.identity(3) * mult
    bias_3d += point

    for i, (clr, lbl) in enumerate(zip(('r', 'g', 'b'), labels)):
        axes.plot(*np.column_stack((point, bias_3d[i, :])), color=clr, linewidth=2)
        axes.text(*bias_3d[i, :], lbl, fontsize=12)


def plot_vehicles_trajectory(figure: Figure, t: np.ndarray, vehicles_vectors: np.ndarray, frame: int = None):
    """
    Строит пространственный график траектории движения РН и КА в окне
    ``figure`` по массиву координат аппаратов ``vehicles_vectors``

    ----------------

    График траектории строится в виде двух подграфиков. Масштабы всех осей на
    каждом подграфике одинаковы для лучше демонстрации физического положения
    аппаратов друг относительно друга.

    На левом подграфике строится общий вид траекторий движения двух тел. На
    него выводятся траектории тел в виде пространственных кривых и точки,
    соответствующие текущему местоположению аппаратов на траекториях. Этот
    график имеет статичные пределы осей, не меняющиеся от местоположения РН и КА.

    На правом подграфике строится увеличенный вид траектории. Он используется
    для демонстрации относительного расстояния между РН и КА и ориентации их
    векторов скоростей в СК. Дополнительно на этом графике приводится абсолютная
    скорость каждого аппарата в применяемой СК. Пределы осей на этом графике
    меняются согласно текущему местоположению РН и КА. Пределы подобраны таким
    образом, чтобы для любых координат аппаратов в один момент времени их
    точки оставались в пределах подграфика.

    Параметры ``t`` и ``frame`` по большей части нужны для анимации графика.
    Массив моментов времени ``t`` используется для печати в заголовок на правом
    подграфике времени, соответствующего местоположению РН и КА.
    Параметр ``frame`` используется для перестроения текущего вида траектории,
    согласно порядковому номеру кадра в анимации.

    ----------------

    :param figure: Объект окна для вывода графиков
    :param t: Массив из N моментов времени
    :param vehicles_vectors: Матрица 12xN, где первые 6 строк – радиус-вектор и вектор скорости РН,
                             а вторые – то же самое для КА
    :param frame: (по умолчанию не используется) Порядковый номер кадра анимации
    """
    # Рассчитывает геометрические параметры графика по набору выводимых на него координат vectors
    def get_geom_anchors(vectors: np.ndarray) -> tuple:
        max_v, min_v = np.max(vectors, axis=1), np.min(vectors, axis=1)
        lims = max_v - min_v

        # Максимум, минимум координат, величина отступа от центра до края графика,
        # макс. разность координат, координаты центра графика
        return max_v, min_v, lims / 1.9, max(lims), min_v + lims / 2.

    # Возвращает сонаправленный вектор для vector, но с указанной длиной length
    def get_with_length(vector: np.ndarray, length: float) -> np.ndarray:
        return vector / la.norm(vector) * length


    v1_r, v1_v, v2_r, v2_v = np.vsplit(vehicles_vectors, 4)

    # Если не проигрываем анимацию, то выводим график траектории целиком
    if frame is None:
        frame = v1_r.shape[1] - 1

    # Первый вызов функции
    if not hasattr(plot_vehicles_trajectory, 'figures_gnrl'):
        # Создание и настройка графика общего вида траектории
        v_max, v_min, d_lims, d_lim_max, cntr = get_geom_anchors(np.hstack((v1_r, v2_r)))

        axes_general = figure.add_subplot(1, 2, 1, projection='3d')
        axes_general.set(
            title='Общий вид траектории',
            xlim=(cntr[0] - d_lims[0], cntr[0] + d_lims[0]),
            ylim=(cntr[1] - d_lims[1], cntr[1] + d_lims[1]),
            zlim=(cntr[2] - d_lims[2], cntr[2] + d_lims[2]),
            box_aspect=(v_max - v_min) / d_lim_max, xlabel='X, метры', ylabel='Y, метры', zlabel='Z, метры')
        plot_bias(axes_general, v_min + 1e-3 * d_lim_max, mult=0.05 * d_lim_max)

        # Добавление фигур на график общего вида
        plot_vehicles_trajectory.figures_gnrl = [
            axes_general.plot(*(v1_r[:, 0]), color='k', linestyle='--', label='Траектория РН')[0],
            axes_general.plot(*(v2_r[:, 0]), color='r', linestyle=':', label='Траектория КА')[0],
            axes_general.scatter(*(v1_r[:, 0]), color='k'),
            axes_general.scatter(*(v2_r[:, 0]), color='r')
        ]
        axes_general.legend(handles=plot_vehicles_trajectory.figures_gnrl[:2])

        # Создание графика увеличенного вида траектории
        plot_vehicles_trajectory.axes_closeup = figure.add_subplot(1, 2, 2, projection='3d')
        plot_vehicles_trajectory.d_lim = np.max(abs(v1_r - v2_r)) / 1.9

    # Отрисовываем график общего вида (две траектории, две точки)
    trajectory1, trajectory2, point1, point2 = plot_vehicles_trajectory.figures_gnrl
    trajectory1.set_data_3d(*(v1_r[:, :frame + 1]))
    trajectory2.set_data_3d(*(v2_r[:, :frame + 1]))
    point1._offsets3d = tuple(np.split(v1_r[:, frame], 3))
    point2._offsets3d = tuple(np.split(v2_r[:, frame], 3))

    # Для увеличенного графика берем только координаты для одного момента времени
    v1_r, v1_v, v2_r, v2_v = np.split(vehicles_vectors[:, frame], 4)
    _, v_min, _, _, cntr = get_geom_anchors(np.vstack((v1_r, v2_r)).T)

    # Настраиваем и отрисовываем график приближенного вида
    d_lim = plot_vehicles_trajectory.d_lim
    axes = plot_vehicles_trajectory.axes_closeup
    axes.clear()
    axes.set(
        title=rf'Увеличенный вид траектории, t={t[frame]:.2f} сек.',
        xlim=(cntr[0] - d_lim, cntr[0] + d_lim),
        ylim=(cntr[1] - d_lim, cntr[1] + d_lim),
        zlim=(cntr[2] - d_lim, cntr[2] + d_lim),
        box_aspect=(1, 1, 1), xlabel='X, метры', ylabel='Y, метры', zlabel='Z, метры')

    plot_bias(axes, cntr - 0.95 * d_lim, mult=0.2 * d_lim)
    # Точки РН и КА
    axes.scatter(*v1_r, color='k', marker='*', s=50)
    axes.text(*(v1_r + 1e-2 * d_lim), s='РН')
    axes.scatter(*v2_r, color='r', marker='*', s=50)
    axes.text(*(v2_r + 1e-2 * d_lim), s='КА')
    # Приведенные векторы скорости РН и КА
    axes.quiver(*v1_r, *get_with_length(v1_v, 0.2 * d_lim), color='b')
    axes.text(*(v1_r + get_with_length(v1_v, 0.2 * d_lim) + 1e-2 * d_lim), s=rf'$\bf |v|$={la.norm(v1_v):.2f} м/с')
    axes.quiver(*v2_r, *get_with_length(v2_v, 0.2 * d_lim), color='b')
    axes.text(*(v2_r + get_with_length(v2_v, 0.2 * d_lim) + 1e-2 * d_lim), s=rf'$\bf |v|$={la.norm(v2_v):.2f} м/с')


def show_anim(figure: Figure, func_draw: Callable, t: np.ndarray, *args):
    """
    Настраивает анимацию и проигрыватель для нее в окне графиков ``figure``,
    используя для отрисовки кадра анимации функцию ``func_draw`` вида::

        func_draw(figure, t, *args, frame)

    ----------------

    Функция создает проигрыватель анимации, который предоставлеят возможность в произвольный момент времени анимации:
        - Запускать/останавливать анимацию
        - Переходить к ближайшему предыдущему/следующему кадру анимации
        - Переходить в начало конец анимации

    ----------------

    :param figure: Объект окна для вывода графиков
    :param func_draw: Функция отрисовки кадра анимации
    :param t: Массив моментов времени (временная шкала)
    :param args: Варьируемые аргументы функции отрисовки
    """
    # Функция-обработчки события анимации
    def anim_update(frame):
        nonlocal anim_paused, current_frame

        func_draw(figure, t, *args, current_frame)
        if current_frame == total_frames - 1:
            current_frame = 0
            anim_paused = True
            anim.event_source.stop()
        else:
            current_frame += 1

    # Останавливает анимацию и принудительно переходит к кадру new_cur_frame
    def stop_and_redraw(new_cur_frame):
        nonlocal anim_paused, current_frame

        anim_paused = True
        anim.event_source.stop()
        current_frame = np.clip(new_cur_frame, 0, total_frames - 1)

        func_draw(figure, t, *args, current_frame)
        # Замена значка на "воспроизведение"
        buttons[2].label.set_text(buttons_labels[2])
        plt.draw()

    # Функция-обработчик события нажатия кнопки воспроизведения/паузы
    def on_button_play_clicked(event):
        nonlocal anim, anim_paused, current_frame

        for btn in buttons:
            btn.set_active(True)

        if anim_paused:
            anim_paused = False
            if current_frame == total_frames - 1:
                current_frame = 0

            anim = FuncAnimation(
                figure, anim_update,
                frames=total_frames - current_frame,
                interval=50,
                blit=False,
                repeat=False
            )
            # Замена значка на "паузу"
            buttons[2].label.set_text(buttons_labels[-1])
            plt.draw()
        else:
            stop_and_redraw(current_frame)

    anim = FuncAnimation
    anim_paused = True

    total_frames = t.size
    current_frame = total_frames - 1

    # Настройка проигрывателя в окне графиков
    buttons_h_pos = (0.450, 0.475, 0.500, 0.525, 0.550)
    buttons_labels = ['⏮', '', '▶', '', '⏭', '']
    buttons_labels_v_align = ['center', 'center_baseline', 'center_baseline', 'center_baseline', 'center']
    buttons: list[Button] = []

    for h_pos, label, v_align in zip(buttons_h_pos, buttons_labels, buttons_labels_v_align):
        buttons.append(Button(figure.add_axes((h_pos, 0.01, 0.02, 0.04)), label))
        buttons[-1].label.set(fontfamily="Segoe UI Symbol", fontsize=16, verticalalignment=v_align)

    buttons[0].on_clicked(lambda e: stop_and_redraw(0))
    buttons[1].on_clicked(lambda e: stop_and_redraw(current_frame - 1))
    buttons[2].on_clicked(on_button_play_clicked)
    buttons[3].on_clicked(lambda e: stop_and_redraw(current_frame + 1))
    buttons[4].on_clicked(lambda e: stop_and_redraw(total_frames - 1))

    # Деактивация всех кнопок, кроме кнопки воспроизведения
    for button in buttons:
        button.set_active(False)
    buttons[2].set_active(True)

    # Первоначальная отрисовка (т.к. анимация на паузе)
    func_draw(figure, t, *args, current_frame)
    plt.show()
