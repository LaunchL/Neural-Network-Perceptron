'''
        NN Perceptron Dinner - простой Исскуственный интелект на основе перцептрона.
                        Dinner NN Perceptron
                    Созданно в Laboratory Lab.
            Discord - https://discord.gg/ngMUeFEgQa
            GitHub - https://github.com/LaunchL
'''
#Impots
import numpy as np
import numpy.random as rand
import random

#Code
#Neural Network - Неиронная сеть
def sigmoid(x, der=False):
    if der:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


x = np.array([[1, 0, 1],
            [1, 0, 1],
            [0, 1, 0],
            [0, 1, 0]])

y = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)

syn0 = 2 * np.random.random((3, 1)) - 1

l1 = []

#Train - Обучение
#Внимание, если у вас слабый ПК снизьте показатель 10000 до 1000! Не поднимайте!
for iter in range(10000):
    l0 = x
    l1 = sigmoid(np.dot(l0, syn0))

    l1_error = y - l1

    l1_delta = l1_error * sigmoid(l1, True)

    syn0 += np.dot(l0.T, l1_delta)

print(l1)

#Test
r1 = random.randint(0, 1)
r2 = random.randint(0, 1)
r3 = random.randint(0, 1)

z = np.array([[r1,r2,r3]])
l2 = z
l3 = sigmoid(np.dot(l2, syn0))

print("Test: ")
print(l3)