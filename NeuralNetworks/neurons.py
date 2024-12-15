import math

class Neuron: 
    def __init__(self, activator, index, layer, init_value = 0):
        self.activator = activator
        self.outgoing = [] #list of weights
        self.incoming = [] #list of weights
        self.index = index
        self.layer = layer
        self.value = init_value
        self.derivative = 0
    
    def attachTowards(self, other_neuron): 
        edge = Weight(self, other_neuron, other_neuron.layer)
        self.outgoing.append(edge)
        other_neuron.receiveIncoming(edge)
        #return a new weight
        return edge
    
    def receiveIncoming(self, edge):
        self.incoming.append(edge)
    
    def updateValue(self): 
        val = 0
        for edge in self.incoming:
            val += (edge.value) * (edge.from_neuron.value)
        self.value = self.activator(val)


    def __repr__(self):
        return 'Node{}^{} with {} incoming {} outgoing'.format(self.index, self.layer, len(self.incoming), len(self.outgoing))
    
    

class Weight:
    def __init__(self, from_neuron, to_neuron, target_layer, init_value = 0): 
        self.from_neuron = from_neuron
        self.towards = to_neuron
        self.target_layer = target_layer
        self.value = init_value
        self.id = '{}.{}.{}'.format(self.from_neuron.index, self.towards.index, self.target_layer)
    
    def updateValue(self, val):
        self.value = val
    
    def __repr__(self):
        return 'Weight{}: Value {}'.format(self.id, self.value)

class Network:
    def __init__(self, d, layers, activator):
        self.d = d + 1 #add one for bias
        self.neurons = [] #layer 0 should be input
        self.weights = []
        self.activator = activator
        self.layers = layers
        self.edges = []

        self.create_neurons()
        self.attach_edges()

        
        
        print(len(self.neurons))

    def create_neurons(self):
        hidden_size = self.layers - 1
        #technically this creates the input size too
        for layer in range(hidden_size):
            # add bias
            current_layer = []
            biasNeuron = Neuron(self.activator, 0, layer, 1)
            current_layer.append(biasNeuron)            
            for i in range(1, self.d):
                neur = Neuron(self.activator, i, layer)
                current_layer.append(neur)
            self.neurons.append(current_layer)
        #add final y
        final_layer = []
        final_layer.append(Neuron(linear, 0, self.layers))
        self.neurons.append(final_layer)

    
    def attach_edges(self):
        #attach every node at level i to level i + 1
        weights = []
        for i in range(self.layers - 1):
            layer_i = self.neurons[i]
            next_layer_j = self.neurons[i + 1]

            current_layer = []
            for neuron_i in layer_i:
                #attach to a node in the next layer
                for neuron_j in next_layer_j:
                    edge = neuron_i.attachTowards(neuron_j)
                    current_layer.append(edge)
            weights.append(current_layer)
        self.weights = weights
    
    def getY(self):
        return self.neurons[self.layers - 1][0]
    
    def getNeuron(self, index, layer):
        return self.neurons[layer][index]
    
    def getEdge(self, index, layer):
        return self.weights[layer][index]
    
    def predict(self, x): 
        self.forward_pass(x)
        return self.getY().value
    
    #does a prediction for y , do not want to update any biases
    def forward_pass(self, x):
        if len(x) != (self.d - 1):
            return -math.inf
        self.updateInputs(x)
        for i in range(1, self.layers - 1): 
            self.updateInnerLayerAtLevel(i)
        
    def updateInputs(self, x):
        input_layer = self.neurons[0]
        size = len(input_layer)

        for i in range(1, size - 1): 
            input = input_layer[i]
            input.value = x[i] #update value
    
    def updateInnerLayerAtLevel(self, level):
        input_layer = self.neurons[level]
        size = len(input_layer)

        for i in range(1, size - 1): 
            neuron = input_layer[i]
            neuron.updateValue()
            #grab
            # input.value = x[i] #update value # update it to be the sum ... hmmm
    


def sigmoid(x):
    return (1 / (1 + math.e ** (-x)))

def linear(x):
    return x


if __name__ == '__main__':
    network = Network(3, 3, sigmoid)
    print(network.predict([1, 1, 1]))

    # print(sigmoid(1))