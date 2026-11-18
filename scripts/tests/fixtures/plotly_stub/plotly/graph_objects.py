import json


class Scatter(dict):
    pass


class Figure:
    def __init__(self):
        self.traces = []

    def add_trace(self, trace):
        self.traces.append(trace)

    def update_layout(self, **layout):
        pass

    def write_html(self, path):
        with open(path, "w") as f:
            json.dump(self.traces, f)
