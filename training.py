import keras
from keras.models import Sequential, Model
from keras.layers import Dense
import numpy as np
import multiprocessing as mp

import yaml

import random

from collections import deque

from environment import Environment

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_model():
    model = Sequential()
    model.add(Dense(64, input_shape=(STATE_SIZE, ), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(ACTION_SIZE, activation = 'linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    return model

def learn(model_input, memory, memory_win, memory_lose):
    state_list = []
    target_list = []
    sample_batch = []

    batch_size_std = min(BATCH_SIZE, len(memory))
    batch_size_win = min(BATCH_SIZE, len(memory_win))
    batch_size_lose = min(BATCH_SIZE, len(memory_lose))

    sample_batch.append(random.sample(memory, batch_size_std))
    sample_batch.append(random.sample(memory_win, batch_size_win))
    sample_batch.append(random.sample(memory_lose, batch_size_lose))

    for i in range(3):
        for state, action, reward, next_state, done in sample_batch[i]:
            target = reward
            if not done:
                target = reward + DISCOUNT_FACTOR * np.amax(model_input.predict(np.array([next_state])))
            target_f = model_input.predict(np.array([state]))[0]
            target_f[action] = target
            state_list.append(state)
            target_list.append(target_f)

    model_input.fit(np.array(state_list), np.array(target_list), epochs=1, verbose=0)

def single_episode(model_input, eps, memory, memory_win, memory_lose):
    state = ENV.reset_random()
    done = False
    while not done:
        if np.random.random() < eps:
            action = np.random.randint(0, ACTION_SIZE)
        else:
            action = np.argmax(model_input.predict(np.array([state])))

        new_state, reward, done, _ = ENV.perform_action(action)

        to_append = [state, action, reward, new_state, done]

        if(reward == 1):
            memory_win.append(to_append)
        elif(reward == -1):
            memory_lose.append(to_append)
        else:
            memory.append(to_append)

        state = new_state

    return reward, memory, memory_win, memory_lose

def train():
    eps = 1.0

    memory = deque(maxlen=10000)
    memory_win = deque(maxlen=10000)
    memory_lose = deque(maxlen=10000)
    model = create_model()

    reward_queue = deque(maxlen=100)
    reward_list = []
    success_list = []

    for episode in range(EPISODES):

        reward, memory, memory_win, memory_lose = single_episode(model, eps, memory, memory_win, memory_lose)
        learn(model, memory, memory_win, memory_lose)

        eps = max(EPS_MIN, eps * DECAY_FACTOR)

        reward_list.append(reward)
        reward_queue.append(reward)
        success = int(reward_queue.count(1)/(len(reward_queue)+0.0)*100)
        success_list.append(success)

        print("Episode: {:7.0f}, Eps: {:0.2f}, Success: {:3.0f}".format(episode, eps, success))

        if(episode % 100 == 0):
            np.savetxt("results/reward_list.txt", reward_list, fmt='%3i')
            np.savetxt("results/success_list.txt", success_list, fmt='%3i')

        if(success > 90):
            model.save("models/backup_" + str(success ) + ".h5")


if __name__ == '__main__':

    with open("configuration/config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    TIMEOUT = cfg['environment']['timeout']
    STEP = cfg['environment']['step']
    ERROR = cfg['environment']['error']

    STATE_SIZE = cfg['network_size']['state_size']
    ACTION_SIZE = cfg['network_size']['action_size']
    BATCH_SIZE = cfg['network_size']['batch_size']

    EPS_MIN = cfg['hyperparameters']['epsilon_min']
    DECAY_FACTOR = cfg['hyperparameters']['epsilon_decay']
    DISCOUNT_FACTOR = cfg['hyperparameters']['discount_factor']

    EPISODES = cfg['training']['episodes']

    ENV = Environment(TIMEOUT, STEP, ERROR)

    train()