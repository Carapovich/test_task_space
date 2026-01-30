import csv

def read_initial_conditions(filename) -> dict:
    with open(filename, mode='r', newline='') as file:
        ic_dict: dict[str, float] = {}
        csv_reader = csv.reader(file, delimiter=';')
        for row in csv_reader:
            ic_dict.update({ row[0]: float(row[1]) })

    return ic_dict


