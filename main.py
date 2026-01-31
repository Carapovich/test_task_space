"""
Тестовое задание по космосу

Моделирование динамики и кинематики движения двух материальных точек на околоземной орбите
"""

import argparse

import src.simulation as core
import src.utils as utils

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
    input_dict = utils.read_initial_conditions(args.input)

    sim_input = core.SimulationInput.from_dict(input_dict)
    core.run_simulation(sim_input)

    return 0

main()