import random
class Neuron:

    # weight
    # bias

    # current_cost
    # current_activation

    # activation_function
    # activation_derivative_function
   
    def __init__(self) -> None:
        
        self.activation_function = sigmoid
        self.activation_derivative_function = sigmoid

        self.weight = float(random.randrange(-1000,1000))/2000.0

        self.bias = float(random.randrange(-1000,1000))/2000.0

        self.current_cost = 0.0

        self.current_activation = 0.0

        self.learning_rate = 0.2

    
        pass

    def FeedForward(self,value = 0.0):

        #calculate N(x) = w*x+b
        self.current_activation = self.activation_function(value * self.weight + self.bias) 

        return self.current_activation
    
    def BackPropagate(self,cost = 0.0):
        
        #first calculate the cost, actually provided to this neuron

        # we need dCost(x)/dActivation(q)

        self.current_cost = cost * self.activation_derivative_function(self.current_activation)

        self.bias = self.bias - 1 * self.learning_rate * self.current_cost


    def __str__(self) -> str:
        
        s  = "Neuron: \n"
        s += "Weight: " + ("%.2f" % self.weight) + "\n"
        s += "Bias  : " + ("%.2f" % self.bias)+ "\n"
        s += "Cost  : " + ("%.2f" % self.current_cost) + "\n"
        s += "Activ : " + ("%.2f" % self.current_activation) + "\n"
        
        return s

#
#   sig(x)
#   d_sig(x)/d_x
# 


import math
def sigmoid(value = 0.0):
    return 1 / (1 + math.exp(-value))


import math
def sigmoid_dx(value = 0.0):
    s = sigmoid(value)
    return s*(1-s)






TrainingData = [0.2,0.3,0.1,0.7,0.6,0,0.4,0.2,0.5,0.2,0.7,0.1,0.8,0.1,1,0.9,0.3]
TrainingLabel =[0  ,0  ,0  ,1  ,1  ,0,0  ,0  ,1  ,0  ,1  ,0  ,1  ,0 ,1 ,1  ,0  ]


#start demo here

neuron = Neuron()

print("Neuron :", neuron)


def train():
    print(">>>>>>>>>> Number Rounding Test <<<<<<<<<")
    print("Training Data :", TrainingData)
    print("Training Label:", TrainingLabel)

    outputs = []        

    #create a output set
    for i in range(len(TrainingData)):

        outputs.append(neuron.FeedForward(TrainingData[i]))

    printable_outputs = ["%.2f" % a for a in outputs]

    print("FF Outputs   :", printable_outputs)



    #the cost function
    # C = sum( (label - output)^2 )
    the_output_cost = []
    for i in range(len(TrainingLabel)):

        the_output_cost.append((TrainingLabel[i] - outputs[i])) #training cost


    #print("Output Cost: ", the_output_cost)

    printable_cost = ["%.2f" % a for a in the_output_cost]

    print("FF Cost   :", printable_cost)


    learning_rate = 0.02

    neuron.learning_rate = learning_rate

    for i in range(len(the_output_cost)):

        #train the bias
        #neuron.BackPropagate(the_output_cost[i])

        neuron.current_cost = the_output_cost[i] * neuron.activation_derivative_function(neuron.current_activation)

        neuron.bias = neuron.bias - 1 * neuron.learning_rate * neuron.current_cost

        #train the weight, corresponding for the training data value
        weight_delta = neuron.current_cost * TrainingData[i]
        

        neuron.weight = neuron.weight - weight_delta


    print("Neuron :", neuron)




for i in range(100):
    train()