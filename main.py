"""
Тестовое задание по космосу

Симуляция динамики и кинематики движения двух материальных точек на околоземной орбите
"""

import argparse

from .src.utils import read_initial_conditions
# from .src.model_motion import ModelMotion

def parse_arguments():
    parser = argparse.ArgumentParser(description='Моделирование движения материальных точек')
    parser.add_argument(name='-i', help='Путь до .csv файла с начальными условиями')
    parser.add_argument(name='-o', help='Путь до .csv файла с результатами моделирования')

    return parser.parse_args()

def main():
    args = parse_arguments()
    sim_input = read_initial_conditions(args)

    return 0

if __name__ == '__main__':
    main()
