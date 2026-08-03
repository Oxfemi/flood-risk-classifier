import json


def load_metrics():

    with open(
        "models/artifacts/metrics.json",
        "r",
    ) as file:

        return json.load(file)


