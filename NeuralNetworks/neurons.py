import math

class Neuron: 
    def __init__(self, activator, index, layer, init_value = 0, is_y = False):
        self.activator = activator
        self.outgoing = [] #list of weights
        self.incoming = [] #list of weights
        self.index = index
        self.layer = layer
        self.value = init_value
        self.der = 0
        self.is_y = is_y
    
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
    
    def updateDeriv(self):
        if self.is_y:
            derivative = 0 
            for w in self.outgoing:
                derivative +=( w.value * w.towards.der * self.value * (1 - self.value))
            self.der = derivative
        # return self.der


    def __repr__(self):
        return 'Node{}^{} with {} incoming {} outgoing. Value: {}'.format(self.index, self.layer, len(self.incoming), len(self.outgoing), self.value)
    
    

class Weight:
    def __init__(self, from_neuron, to_neuron, target_layer, init_value = 0): 
        self.from_neuron = from_neuron
        self.towards = to_neuron
        self.target_layer = target_layer
        self.value = init_value
        self.id = '{}.{}.{}'.format(self.from_neuron.index, self.towards.index, self.target_layer)
    
    def updateValue(self, val):
        self.value = val

    def getGradient(self): 
        #update outgoing to be the derivative of outlayer going to it? 
        return self.towards.der * (self.from_neuron.value)
    
        # return 
    
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
        hidden_size = self.layers
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
        final_layer.append(Neuron(linear, 0, self.layers, 100, True))
        self.neurons.append(final_layer)

    
    def attach_edges(self):
        #attach every node at level i to level i + 1
        weights = []
        layer_size = len(self.neurons)
        print("num of lyaers", layer_size)
        for i in range(layer_size - 1):
            layer_i = self.neurons[i]
            next_layer_j = self.neurons[i + 1]

            current_layer = []
            for neuron_i in layer_i:
                #attach to a all the non bias layers in the network
                init_index = 1
                if next_layer_j[0].is_y:
                    init_index = 0
                for j in range(init_index, len(next_layer_j)):
                    neuron_j = next_layer_j[j]
                    edge = neuron_i.attachTowards(neuron_j)
                    current_layer.append(edge)
            weights.append(current_layer)
        self.weights = weights
    
    def getY(self):
        return self.neurons[self.layers][0]
    
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
        for i in range(1, self.layers): 
            self.updateInnerLayerAtLevel(i)
        self.getY().updateValue()
        
        
    def updateInputs(self, x):
        input_layer = self.neurons[0]
        size = len(input_layer)

        for i in range(1, size): 
            input = input_layer[i]
            input.value = x[i - 1] #update value
    
    def updateInnerLayerAtLevel(self, level):
        input_layer = self.neurons[level]
        size = len(input_layer)

        for i in range(1, size): 
            neuron = input_layer[i]
            neuron.updateValue()
            #grab
            # input.value = x[i] #update value # update it to be the sum ... hmmm


    def getGradient(self, x, y_):
        #update inputs and get prediction for y
        y_pred = self.predict(x) 
        #update gradient of last node

        self.getY().der = (y_pred - y_)
        grad = []
        weight_size = len(self.weights)
        for index in range(weight_size - 1, -1, -1):
            weight_layer = self.weights[index]
            grad_layer = []
            for edge in weight_layer:
                grad_layer.append(edge.getGradient())
            #update the gradients of the nodes on the next layer
            queue_to_update = self.neurons[index]
            for neuron in queue_to_update:
                neuron.updateDeriv()
            grad.append(grad_layer) #as long as I loop over it the same to update, it should be fine? 
        flattened_grad = []
        for gradient in grad:
            flattened_grad = gradient + flattened_grad #done so that its actually in order
        return flattened_grad #todo: flatten
    
    def flattenW(self): 
        flattened = []
        for gradient in self.weights:
            flattened += gradient
        return flattened 
    
    def getW(self): 
        edges = self.flattenW()
        return [weight.value for weight in edges]


    
    def updateW(self, next_w): 
        current_w = self.flattenW()
        for w, edge in zip(next_w, current_w):
            edge.value = w

    def printInnerLayers(self):
        for neuron in self.neurons:
            print(neuron)



def sigmoid(x):
    return (1 / (1 + math.e ** (-x)))

def linear(x):
    return x


if __name__ == '__main__':
    network = Network(2, 3, sigmoid)
    x = [1, 1]
    y = 1
    init_w = [-1, 1, -2, 2, -3, 3, -1, 1, -2, 2, -3, 3, -1, 2, -1.5]
    network.updateW(init_w)
    print(network.getW())
    print(network.getGradient(x, y))

    # print(sigmoid(1))