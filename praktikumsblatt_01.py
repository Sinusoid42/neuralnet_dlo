
TESTS = 100



import random


#normalize weights [0,1]
weight_0 = float(random.randrange(-1000, 1000))/2000.0
weight_1 = float(random.randrange(-1000, 1000))/2000.0

# bias_0 = float(random.randrange(-1000, 1000))/2000.0
# bias_1 = float(random.randrange(-1000, 1000))/2000.0

learning_rate = 0.002
weights_0 = []
weights_1 = []
costs = []


for test in range(TESTS):

    #generiere zufällige werte für die addition





    random0 = random.uniform(-10,10)
    random1 = random.uniform(-10,10) 

    training_label = random0 + random1
    if (test%20==0):
        print("########## TRAIN ##########")
        print(test)
        print("X0:" ,random0, "\nX1:", random1,"\nA:",training_label)


        print("########## SUM");
    #berechne neuronen summe
    #feed forward
    neuronOutput = weight_0 * random0 + weight_1 * random1
    if (test%20==0):
        print("########## OUTPUT: ",neuronOutput)
    
    #berechne die kosten für das neuron
    cost = neuronOutput - training_label
    if (test%20==0):
        print("The Cost: ", cost)

    #kostenpropagation durch das neuron
    #costapated = cost * activation_derivative

    #update the weight
    weight_0_delta = -(cost * learning_rate * random0)

    weight_0 = weight_0 + weight_0_delta

    #
    weight_1_delta = -(cost * learning_rate * random1)

    weight_1 = weight_1 + weight_1_delta
    
    if (test%20==0):
        print("W1: ", weight_0, "\n")
        print("W2: ", weight_1, "\n")

    weights_0.append(weight_0)
    weights_1.append(weight_1)

    costs.append(cost)


x = [i for i in range(len(weights_0))]



import matplotlib.pyplot as plt

plt.plot(x, weights_0)
#plt.show()
plt.plot(x, weights_1)
plt.plot(x, costs)
plt.show()



















































