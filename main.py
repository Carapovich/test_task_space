"""
Тестовое задание по космосу

Моделирование динамики и кинематики движения двух материальных точек на околоземной орбите
"""

import argparse

import src.simulation as sim
import src.utils as utils


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

    sim_input = sim.SimulationInput.from_dict(input_dict)
    sim_result = sim.run_simulation(sim_input)
    sim.process_result((args.output, sim_result))

    return 0


main()
