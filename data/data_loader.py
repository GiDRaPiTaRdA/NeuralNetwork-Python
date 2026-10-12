import csv
import io
import zipfile
import numpy as np
from neural_network import TrainCase


class DataLoader:
    def __init__(self, path="data/10x10_digit_training_dataset.zip"):
        self.path = path

    def load(self):
        cases = []

        with zipfile.ZipFile(self.path, "r") as archive:
            with archive.open("labels.csv") as file:
                reader = csv.DictReader(io.TextIOWrapper(file))
                # print(reader.fieldnames)

                for row in reader:
                    pixels = np.array(
                        [float(row[f"p{i}"]) for i in range(100)]
                    )

                    expected = np.zeros(10)
                    expected[int(row["label"])] = 1.0

                    cases.append(TrainCase(pixels, expected))

        return cases