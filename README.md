# log
just separate repository for thinking and writing down ideas and new things throughout journey

first commitment on 11.11.2025 - buckle up

# day 43 or so
yeah i forgot about all these days. but ive learned a lot - first of all - ive completed 2 projects on my own where i debunked softmax, graphs etc. for superficial use. however then ive faced a new problem - not understanding what they meant. and ive dived into it:

import math
dataset = [
    [0.1, 0.8, 1.0],
    [0.9, 0.2, 0.1],
    [0.2, 0.9, 0.7]
]
weights = [[0.5, 0.9, 0.1], [0.9, 0.2, 0.1]]
target = [
    [1, 0],
    [0, 1],
    [1, 0]
]
lr = 0.1

def inference(dataset, matrix):
    logits = []
    for row in matrix:
        score = sum(v * w for v, w in zip(dataset, row))
        logits.append(score)
    return logits

def softmax(logits):
    exps = [math.exp(l) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def cross_entropy(probs, real):
    loss = 0
    for p, r in zip(probs, real):
        if r == 1:
            loss = -math.log(p)
    return loss

for epoch in range(200):
    epoch_loss = 0
    for i in range(len(dataset)): 
        game = dataset[i]
        targetin = target[i]   
        logits = inference(game, weights)
        probs = softmax(logits)
        loss = cross_entropy(probs, targetin)
        epoch_loss += loss
        for row in range(len(weights)):
            for col in range(len(weights[row])):
                grad = (probs[row] - targetin[row]) * game[col]
                weights[row][col] -= lr * grad
        if epoch % 50 == 0:
            print(f"step: {epoch}, game = {i}, avgloss = {epoch_loss / len(dataset)}, loss = {loss}, probs = {probs}, weights = {weights}")
            new_game = [0.1, 0.9, 0.8]
            prediction = softmax(inference(new_game, weights))
            print(f"Новая игра: Хит: {prediction[0]:.2%}, Провал: {prediction[1]:.2%}")

here ive done what all other models do - matrix calculations, softmax, cross-entropy and finally weight changing by gradient. but lets talk about details: 1. dataset obviously can contain any number of vectors but they must be the same as weights 2. they we are taking weights and vectors and multiplying them (in this case without bias) but by its nature it is linear equation (kx + b), we append. we take those logits, exp them (we blow them bigger for our model to be more confident and be able to choose one or another) and take the ratio (e / sum(exps)). we take the ratio (probabilities lets say) and -log them (this is loss - difference between our probs and target (but reversed as we need to "go down the hill" and lower errors rather than make them bigger). taking into account all of these, we go through every weight (in this case every instance of weights and weight in it) and calculate grad ((P - T) * X, tho i do know that this formula is much more complex and involves ratios of L / P, P / z, z / w but its better to keep it simple). than we - learning rate (essentially, how much of the gradient we believe) * gradient itself (AND HERE AGAIN - AS GRADIENT IS LINEAR FUNCTION, BY ITS NATURE IT IS A LINE, LINE SEPARATING HIT AND FAILURE, THAT IS BY * LEARNING RATE WE SAY: "I BELIEVE 10% OF THIS LINE" AND FROM ALREADY EXISTING LINE (WEIGHTS) WE - GRADIENT, MAKING MUCH MORE STABLE LINE (abstract example)). well thats it for today. we'll see what is next.

# day 43 (again):
well ive made today more of ai so lets see:

import math
import random
dataset = [[0.1, 0.8, 1.0, 1.0], [0.9, 0.2, 0.1, 1.0], [0.2, 0.9, 0.7, 1.0]]
target = [[1, 0], [0, 1], [1, 0]]
lr = 0.05
neurons = [[random.uniform(-0.1, 0.1) for _ in range(4)] for _ in range(4)]
finals = [[random.uniform(-0.1, 0.1) for _ in range(4)] for _ in range(2)]
def relu(x): return max(0, x)
def relu_deriv(x): return 1 if x > 0 else 0 

for epoch in range(500):
    total_loss = 0
    for i in range(len(dataset)):
        vectors = dataset[i]
        neurons_logits = [relu(sum(v * w for v, w in zip(vectors, row))) for row in neurons]
        logits = [sum(v * w for v, w in zip(neurons_logits, row)) for row in finals]
        probs = [math.exp(l) / sum(math.exp(x) for x in logits) for l in logits]
        weights_loss = [probs[j] - target[i][j] for j in range(2)]
        neurons_loss = [0.0] * 4
        for j in range(4):
            error = sum(weights_loss[k] * finals[k][j] for k in range(2))
            neurons_loss[j] = error * relu_deriv(neurons_logits[j])
        for j in range(2):
            for k in range(4):
                finals[j][k] -= lr * weights_loss[j] * neurons_logits[k]      
        for j in range(4):
            for k in range(4):
                neurons[j][k] -= lr * neurons_loss[j] * vectors[k]
    if epoch % 100 == 0:
        print(f"Epoch {epoch} завершена")

test_game = [0.1, 0.9, 0.8, 1.0]
l1 = [relu(sum(v * w for v, w in zip(test_game, row))) for row in neurons]
l2 = [sum(v * w for v, w in zip(l1, row)) for row in finals]
final_probs = [math.exp(l) / sum(math.exp(x) for x in l2) for l in l2]
print(f"Новая игра: Хит {final_probs[0]:.2%}, Провал {final_probs[1]:.2%}")

here ive added "secret" layer (16 kinda neurons) aside from final 8 weights. obviously i had to change the way how final layers (and their probabilities) went back to vectors (ive made that last layers changed their weights the same way (weights_loss) but * not on vectors but neurons output (logits) and neurons by modified gradient of final layer * relu * vectors). well thats it for right now. this is not tree, idk how to call it, but i would call: expanded linear function with nonlinearty.

# day 44 
today ive made a little changes with bias as independent variable and added momentum:

import math, random
dataset = [[0.1, 0.8, 1.0], [0.9, 0.2, 0.1], [0.2, 0.9, 0.7]]
target = [[1, 0], [0, 1], [1, 0]]
lr = 0.02
mucoef = 0.9
neurons = [[random.uniform(-0.1, 0.1) for _ in range(3)] for _ in range(4)]
n_bias = [0.0] * 4
weights = [[random.uniform(-0.1, 0.1) for _ in range(4)] for _ in range(2)]
w_bias = [0.0] * 2
mu_n = [[0.0]*3 for _ in range(4)]
mu_w = [[0.0]*4 for _ in range(2)]

for epoch in range(500):
    for i in range(len(dataset)):
        vectors = dataset[i]
        n_logits = [max(0, sum(v * w for v, w in zip(vectors, row)) + n_bias[idx]) for idx, row in enumerate(neurons)]
        logits = [sum(v * w for v, w in zip(n_logits, row)) + w_bias[idx] for idx, row in enumerate(weights)]
        probs = [math.exp(l) / sum(math.exp(x) for x in logits) for l in logits]
        w_loss = [probs[j] - target[i][j] for j in range(2)]
        n_loss = [sum(w_loss[k] * weights[k][j] for k in range(2)) * (1 if n_logits[j] > 0 else 0) for j in range(4)]
        for j in range(2):
            for k in range(4):
                grad = w_loss[j] * n_logits[k]
                mu_w[j][k] = mucoef * mu_w[j][k] + grad
                weights[j][k] -= lr * mu_w[j][k]
            w_bias[j] -= lr * w_loss[j] 
        for j in range(4):
            for k in range(3):
                grad = n_loss[j] * vectors[k]
                mu_n[j][k] = mucoef * mu_n[j][k] + grad
                neurons[j][k] -= lr * mu_n[j][k]
            n_bias[j] -= lr * n_loss[j]
print("Обучение с моментумом завершено!")

here well almost everythings the same i just took bias out of dataset and well made that weights -= not gradient * learning rate but momentum (first of weights then neurons with formula: mucoef (0.9 in this case) * list of momentums + gradient). thats it.

# day 46
I've pushed the boundaries of my models, making it a "hardcore" optimizer by combining almost everything used in production models. I also compared it with traditional ML methods like KNN, Decision Trees, and SVM to see the difference between "probabilistic thinking" and "distance thinking":

import math, random
dataset = [[0.1, 0.8, 1.0], [0.9, 0.2, 0.1], [0.2, 0.9, 0.7]]
target = [[1, 0], [0, 1], [1, 0]]
lr = 0.02
mucoef = 0.9
l1_param = 0.005
l2_param = 0.01
neurons = [[random.uniform(-0.1, 0.1) for _ in range(3)] for _ in range(4)]
n_bias = [0.0] * 4
weights = [[random.uniform(-0.1, 0.1) for _ in range(4)] for _ in range(2)]
w_bias = [0.0] * 2
m = 2.0
mu_n = [[0.0]*3 for _ in range(4)]
mu_w = [[0.0]*4 for _ in range(2)]

for epoch in range(500):
    for i in range(len(dataset)):
        vectors = dataset[i]
        n_logits = [max(0, sum(v * w for v, w in zip(vectors, row)) + n_bias[idx]) for idx, row in enumerate(neurons)]
        logits = [sum(v * w for v, w in zip(n_logits, row)) + w_bias[idx] for idx, row in enumerate(weights)]
        logits = logits[target[i].index(1)] = logits[target[i].index(1)] / m
        probs = [math.exp(l) / sum(math.exp(x) for x in logits) for l in logits]
        w_loss = [probs[j] - target[i][j] for j in range(2)]
        n_loss = [sum(w_loss[k] * weights[k][j] for k in range(2)) * (1 if n_logits[j] > 0 else 0) for j in range(4)]
        for j in range(2):
            for k in range(4):
                grad = w_loss[j] * n_logits[k]
                l2 = l2_param * weights[j][k] + l1_param * (1 if weights[j][k] > 0 else -1)
                mu_w[j][k] = mucoef * mu_w[j][k] + grad + l2
                weights[j][k] -= lr * mu_w[j][k]
            w_bias[j] -= lr * w_loss[j] 
        for j in range(4):
            for k in range(3):
                grad = n_loss[j] * vectors[k]
                mu_n[j][k] = mucoef * mu_n[j][k] + grad
                neurons[j][k] -= lr * mu_n[j][k]
            n_bias[j] -= lr * n_loss[j]
print("Обучение с моментумом завершено!")

test_game = [0.1, 0.9, 0.8]
l1 = [max(0, sum(v * w for v, w in zip(test_game, row)) + n_bias[i]) for i, row in enumerate(neurons)]
l2 = [sum(v * w for v, w in zip(l1, row)) + w_bias[i] for i, row in enumerate(weights)]
final_probs = [math.exp(l) / sum(math.exp(x) for x in l2) for l in l2]
print(f"Новая игра: Хит {final_probs[0]:.2%}, Провал {final_probs[1]:.2%}")

-----------------------------------------------------------------------------------------

old_games = [
    [0.1, 0.8, 0.1],
    [0.9, 0.9, 0.9], 
    [0.8, 0.2, 0.5],
]
results = [0, 0, 1]

def knn_predict(new_data, dataset, targets, k=3):
    distances = []
    for i in range(len(dataset)):
        dist = sum((new_data[j] - dataset[i][j])**2 for j in range(len(new_data)))**0.5
        distances.append((dist, targets[i]))
    distances.sort(key=lambda x: x[0])
    nearest = [d[1] for d in distances[:k]]
    return "Хит" if max(set(nearest), key=nearest.count) == 0 else "Провал"

test_game = [0.15, 0.85, 0.12]
print(f"Вердикт KNN: {knn_predict(test_game, old_games, results, k=1)}")

-----------------------------------------------------------------------------------------

def tree_predict(game):
    if game[1] > 0.8:
        if game[0] < 0.3: return "Инди-хит"
        else: return "Блокбастер"
    else: return "Провал"

new_game = [0.1, 0.9, 0.5]
print(f"Вердикт дерева: {tree_predict(new_game)}")

-----------------------------------------------------------------------------------------

import random
dataset = [[0.1, 0.8, 0.1], [0.9, 0.9, 0.9], [0.8, 0.2, 0.5]]
targets = [1, 1, -1] 
weights = [random.uniform(-0.1, 0.1) for _ in range(3)]
bias = 0.0
lr = 0.02
C = 1.0
epochs = 1000
l1_param = 0.005
l2_param = 0.01

for epoch in range(epochs):
    for i, x in enumerate(dataset):
        condition = targets[i] * (sum(x[j] * weights[j] for j in range(3)) + bias)
        if condition >= 1:
            for j in range(3):
                weights[j] -= lr * (l2_param * weights[j] + l1_param * (1 if weights[j] > 0 else -1))
        else:
            for j in range(3):
                weights[j] -= lr * (l2_param * weights[j] - C * x[j] * targets[i])
            bias += lr * C * targets[i]

test_game = [0.15, 0.85, 0.12]
result = sum(test_game[j] * weights[j] for j in range(3)) + bias
print(f"SVM вердикт: {'Хит' if result > 0 else 'Провал'} (Счет: {result:.2f})")

def sigmoid(z):
    return 1 / (1 + math.exp(-z))
prob_hit = sigmoid(result)

-----------------------------------------------------------------------------------------

OVERVIEW: first i will explain my linear model (that ive changed third time already): except for momentum and bias i also added regularization (like in cvm: they essenentially are just a way to control (diminish) weights through lambda * weights - l2 is more mild (granual control) while l1 is constant and cuts off unnecessary entirely), and of course large margin (i was inspired by it again by svm) - during the logits phase (before probs) i substracted right logits thus deceiving the model to work twice as harder. Aside from linear model: i tried to make knn (but it was relatively easy as simple knn only requires dict), decision tree (again quite easy cuz i didnt make complex one yet) and svm (the hardest among them - the principle behind svm is geometric distances - that is it always use hinge lose (simplified version of large margin with if else loop whether the distance between targets and logits is more than one and they are separated (if they are aligned or distance < 1 - we add loss (targets * x * C - counteract of lambda). and also: as svm gives a 'score' not probs - we have to manually turn it into probs by sigmoid (the same formula of P = ez / sum(ez) but modified into P = 1 / z + e-z. Also i want to what i found out in theory: 1. log and exp and defined through first better version of gradient descent (where we not only * gradient but also instead of using lr / modified w - thus gaining better control "over the hill") and second through iterations where 1 e2 /4 e3 / 6 etc. 2. e itself works like a powerer stronger and stronger as we go up from 0, log works like a diminisher for big numbers but powerer for small ones (the closer we are to 0 the bigger the number becomes and vice versa), epsilon - just not to / 0, lambda - always pulls down to 0 but the closer to 0 the weaker the force. 3. gradient itself is like a selfreflection: loss (how much is this pain) * activation (what will i do with it / am i ready to do smth) * X (who gave me this pain). 4. condition in svm is like "are you on my side or not" while loss is like "did i (model) understood the assignment right". for today that's it.

