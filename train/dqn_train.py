import collections
import os
from statistics import mean

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from game.gymgame_multiplayer import GameEnv

class DQNGameRunner:
    def __init__(self, env):
        self.env = env

    def run(self, agent, target):
        obs = self.env.reset()
        done = False
        rewards = 0
        step = 0
        while done is not True:
            action = agent.get_action(obs)
            old_obs = obs
            obs, reward, done, info = self.env.step(action)
            agent.add_experience({'state': old_obs, 'action': action,
                                  'reward': reward, 'new_state': obs, 'done': done})

            agent.train(target)
            step += 1
            if step % 25 == 0:
                agent.copy_weights(target)
            rewards += reward
        return rewards


class DQN:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.batch_size = 64
        self.gamma = 0.99
        self.explore_chance = 0.99
        self.explore_decay = 0.996
        self.explore_min_chance = 0.1
        self.memory_maxsize = 1000000

        self.model = self.create_model(state_space, action_space)
        self.memory = {"state": [], "action": [],
                       "reward": [], "new_state": [], "done": []}

    def add_experience(self, exp):
        if len(self.memory['state']) >= self.memory_maxsize:
            for key in self.memory.keys():
                self.memory[key].pop(0)
        for key, value in exp.items():
            self.memory[key].append(value)

    def get_action(self, state):
        if np.random.random() > max(self.explore_chance, self.explore_min_chance):
            q_values = self.model(
                np.atleast_2d(state.astype('float32')))
            # print(q_values)
            return np.argmax(q_values)

        return np.random.randint(self.action_space)

    def create_model(self, input_shape, output_shape):
        model = Sequential()
        model.add(Dense(64, input_shape=(input_shape,),
                  kernel_initializer='RandomNormal', activation='tanh'))
        model.add(Dense(64, kernel_initializer='RandomNormal', activation='tanh'))
        model.add(
            Dense(output_shape, kernel_initializer='RandomNormal', activation='linear'))

        model.compile(loss='mse', optimizer=Adam(0.01))
        return model

    def train(self, target_network):
        if len(self.memory["state"]) < 100:
            return 0

        ids = np.random.randint(low=0, high=len(
            self.memory["state"]), size=self.batch_size)

        states = np.asarray([self.memory["state"][i] for i in ids])
        actions = np.asarray([self.memory["action"][i] for i in ids])
        rewards = np.asarray([self.memory["reward"][i] for i in ids])
        new_states = np.asarray([self.memory["new_state"][i] for i in ids])
        dones = np.asarray([self.memory["done"][i] for i in ids])

        next_qvalue = np.max(
            target_network.model(new_states), axis=1)
        updated_qvalues = np.where(
            dones, rewards, rewards+self.gamma*next_qvalue)

        new_current_qvalues = np.array(self.model(states))

        for i in range(self.batch_size):
            new_current_qvalues[i][actions[i]] = updated_qvalues[i]

        loss = self.model.fit(states, new_current_qvalues, verbose=0)
        return loss

    def copy_weights(self, target_network):
        target_network.model.set_weights(self.model.get_weights())

    def save_model(self, path, epoch):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(path + f'/dqnmodel_{epoch}.h5')

    def load_model(self, path):
        self.model.load_weights(path)


class DQNManager():
    def __init__(self, map_scale=0.4, epochs=1000000):
        self.state_space = 8
        self.action_space = 18
        self.epochs = epochs
        self.env = GameEnv(map_scale=map_scale,
                           p1_obs_space=self.state_space, singleplayer=True)

    def run_training(self):
        game_runner = DQNGameRunner(self.env)

        train_net = DQN(self.state_space, self.action_space)
        target_net = DQN(self.state_space, self.action_space)

        rewards = collections.deque(maxlen=100)

        summary_dir1 = os.path.join("logs", "DQN")
        summary_writer1 = tf.summary.create_file_writer(summary_dir1)

        for i in range(self.epochs):
            reward = game_runner.run(train_net, target_net)
            rewards.append(reward)
            train_net.explore_chance *= train_net.explore_decay

            if i % 10 == 0:
                print(i, mean(rewards))

            with summary_writer1.as_default():
                tf.summary.scalar(name="rollout/ep_rew_mean",
                                  data=mean(rewards), step=i)
                summary_writer1.flush()

            if i % 100 == 0:
                train_net.save_model(".", i)
