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

