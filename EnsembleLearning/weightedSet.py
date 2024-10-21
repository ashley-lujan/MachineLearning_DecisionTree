class WeightedX: 
    def __init__(self, weight, data):
        self.weight = weight
        self.data = data
    def __repr__(self):
        return "Weight: " + str(self.weight) + " Data: " + str(self.data) + "\n"
    # def __str__(self):
    #     return "member of Test"

class Classifiers: 
    def __init__(self, d3, at):
        self.d3 = d3
        self.at = at


