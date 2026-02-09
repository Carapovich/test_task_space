"""
Тестовое задание по космосу

Моделирование динамики и кинематики поступательного движения
двух материальных точек на околоземной орбите
"""

import argparse

import src.simulation as sim
import src.utils as utils


def parse_arguments():
    """
    Парсер аргументов командной строки
    """
    parser = argparse.ArgumentParser(description='Моделирование движения материальных точек')
    parser.add_argument('--input',
                        help='Путь до .csv файла с начальными условиями',
                        default='examples/input.csv')
    parser.add_argument('--output',
                        help='Путь до .csv файла с результатами моделирования',
                        default='examples/output.csv')

    return parser.parse_args()


def main():
    """
    Точка входа в программу
    """
    args = parse_arguments()

    # Входные данные
    input_dict = utils.read_input(args.input)
    sim_input = sim.SimulationInput.from_dict(input_dict)

    # Расчет
    sim_result = sim.run_simulation(sim_input)

    # Вывод результата
    sim.process_result((args.output, *sim_result))

    return 0


main()
