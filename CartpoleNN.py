import gym
import random
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from statistics import mean, median
from collections import Counter

learning_rate = 1e-3
num_epochs = 3
env = gym.make('CartPole-v0')
env.reset()
target_steps = 500
min_steps = 50
initial_games = 10000

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = [] #Store movements etc for current game
        prev_observation = []
        for _ in range(target_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break

        if score >= min_steps:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 0: #Moved Left, one hot encode
                    output = [1,0]
                elif data[1] == 1: #Moved Right, one hot encode
                    output = [0,1]
                training_data.append([data[0], output])
        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    print('Average accepted score: {}'.format(mean(accepted_scores)))
    print('Median accepted score: {}'.format(median(accepted_scores)))
    print(Counter(accepted_scores))

    return training_data

def define_model(input_size):
    #Layers
    model = Sequential()

    model.add(Dense(128, input_dim=4, activation='relu',name='layer_1'))
    model.add(Dropout(0.2))

    model.add(Dense(256, activation='relu', name='layer_2'))
    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu', name='layer_3'))
    model.add(Dropout(0.2))

    model.add(Dense(256, activation='relu', name='layer_4'))
    model.add(Dropout(0.2))

    model.add(Dense(128, activation='relu', name='layer_5'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax', name='output_layer'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model
#Log to Tensorboard
RUN_NAME = 'Run1'
logger = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(RUN_NAME),write_graph=True)

def train_model(training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))  # Transpose + Add dimension
    y = np.asarray([i[1] for i in training_data])

    model = define_model(input_size=len(X[0]))

    model.fit(
        X,
        y,
        epochs=num_epochs,
        shuffle=True,
        verbose=2,
        callbacks=[logger]
    )

    return model

retrain = False
if retrain:
    training_data = initial_population()
    model = train_model(training_data)
    model.save("trained_model.h5")

else:
    model = load_model('trained_model.h5')

scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    previous_observation = []
    env.reset()
    for _ in range(target_steps):
        env.render()
        if len(previous_observation) == 0 :
            action = random.randrange(0,2)
        else:
            X = previous_observation.reshape(-1, len(previous_observation))
            prediction = model.predict(X)
            action = np.argmax(prediction)
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        previous_observation = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
    scores.append(score)

print('Average Score: {}'.format(sum(scores)/len(scores)))
print('Choice 1: {}, Choice 2 : {}'.format(choices.count(0)/len(choices),choices.count(1)/len(choices)))


