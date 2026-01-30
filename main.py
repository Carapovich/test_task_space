"""
Тестовое задание по космосу

Моделирование динамики и кинематики движения двух материальных точек на околоземной орбите
"""

import argparse

import src.utils as utils
import src.core as core
# from .src.model_motion import ModelMotion

def parse_arguments():
    parser = argparse.ArgumentParser(description='Моделирование движения материальных точек')
    parser.add_argument('--input',
                        help='Путь до .csv файла с начальными условиями',
                        default='examples/input.csv')
    parser.add_argument('--output',
                        help='Путь до .csv файла с результатами моделирования',
                        default='examples/output.csv')

    return parser.parse_args()

def main():
    args = parse_arguments()
    sim_input = read_initial_conditions(args.input)
    input_dict = utils.read_initial_conditions(args.input)

    sim_input = core.SimulationInput.from_dict(input_dict)

    return 0

main()