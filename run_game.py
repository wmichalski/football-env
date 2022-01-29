import gym
from game.gymgame_multiplayer import GameEnv
from stable_baselines3 import PPO
from utils.model_container import ModelContainer
import argparse
import pygame
import numpy as np
import time
import sys

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (119, 221, 119)

move_matrix = np.array([[0, 0, 0.7, 0.7, 0.7, 0, -0.7, -0.7, -0.7, 0, 0, 0.7, 0.7, 0.7, 0, -0.7, -0.7, -0.7],
                        [0, 0.7, 0.7, 0, -0.7, -0.7, -0.7, 0, 0.7, 0,
                            0.7, 0.7, 0, -0.7, -0.7, -0.7, 0, 0.7],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


def most_common(lst):
    return max(set(lst), key=lst.count)


class GameVisualiser():
    def __init__(self, env, player_model):
        pygame.init()
        pygame.font.init()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self.fps = 60
        self.env = env
        self.gameDisplay = pygame.display.set_mode(
            (self.env.display_width, self.env.display_height))
        self.clock = pygame.time.Clock()
        self.env.reset()
        
        if player_model.path == 'dummy':
            self.human_interaction = True
        else:
            self.model = player_model.model
            self.human_interaction = False

    def draw_player(self, player):
        pygame.draw.circle(self.gameDisplay, black, (int(player.coords[0]),
                                                     int(player.coords[1])), player.kick_radius, 1)
        pygame.draw.circle(self.gameDisplay, black, (int(
            player.coords[0]), int(player.coords[1])), player.radius+1)
        pygame.draw.circle(self.gameDisplay, white, (int(player.coords[0]),
                                                     int(player.coords[1])), int(player.radius*0.75))

    def draw_ball(self, ball):
        pygame.draw.circle(self.gameDisplay, black,
                           (int(ball.coords[0]), int(ball.coords[1])), ball.radius+1)
        pygame.draw.circle(self.gameDisplay, white,
                           (int(ball.coords[0]), int(ball.coords[1])), ball.radius-2)

    def draw_map(self, env):
        # left
        pygame.draw.line(self.gameDisplay, black, (self.env.display_width/2 - self.env.map_width/2, self.env.display_height /
                                                   2 - self.env.map_height/2), (self.env.display_width/2 - self.env.map_width/2, self.env.display_height/2 + self.env.map_height/2), 1)
        pygame.draw.line(self.gameDisplay, green, (self.env.display_width/2 - self.env.map_width/2, self.env.display_height/2 -
                                                   self.env.goal_height/2), (self.env.display_width/2 - self.env.map_width/2, self.env.display_height/2 + self.env.goal_height/2), 1)
        # right
        pygame.draw.line(self.gameDisplay, black, (self.env.display_width/2 + self.env.map_width/2, self.env.display_height /
                                                   2 - self.env.map_height/2), (self.env.display_width/2 + self.env.map_width/2, self.env.display_height/2 + self.env.map_height/2), 1)
        pygame.draw.line(self.gameDisplay, green, (self.env.display_width/2 + self.env.map_width/2, self.env.display_height/2 -
                                                   self.env.goal_height/2), (self.env.display_width/2 + self.env.map_width/2, self.env.display_height/2 + self.env.goal_height/2), 1)
        # top
        pygame.draw.line(self.gameDisplay, black, (self.env.display_width/2 - self.env.map_width/2, self.env.display_height /
                                                   2 - self.env.map_height/2), (self.env.display_width/2 + self.env.map_width/2, self.env.display_height/2 - self.env.map_height/2), 1)
        # bottom
        pygame.draw.line(self.gameDisplay, black, (self.env.display_width/2 - self.env.map_width/2, self.env.display_height /
                                                   2 + self.env.map_height/2), (self.env.display_width/2 + self.env.map_width/2, self.env.display_height/2 + self.env.map_height/2), 1)

    def draw_text(self, frame, score):
        frame_text = self.myfont.render(str(frame), False, (0, 0, 0))
        score_text = self.myfont.render(str(score), False, (0, 0, 0))
        self.gameDisplay.blit(frame_text, (0, 0))
        self.gameDisplay.blit(score_text, (0, 15))

    def draw_pygame(self, frame, score):
        self.gameDisplay.fill(green)
        self.draw_text(frame, score)
        self.draw_map(self.env)
        for player in self.env.players:
            self.draw_player(player)
        self.draw_ball(self.env.ball)
        pygame.display.update()

    def get_human_input(self):
        keys = pygame.key.get_pressed()

        x_change = 0
        y_change = 0
        kick = 0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            x_change += -0.7
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            x_change += 0.7
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            y_change += -0.7
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            y_change += 0.7
        if keys[pygame.K_SPACE]:
            kick = 1

        action = np.where(
            np.all(move_matrix.T == [x_change, y_change, kick], axis=1))[0][0]

        return action

    def run_once(self):
        obs = env.reset()
        reward_sum = 0
        end_frame = -1
        for frame in range(1, 99999):
            self.draw_pygame(frame, reward_sum)
            if not self.human_interaction:
                actions = []
                for _ in range(5):
                    action, _states = self.model.predict(obs)
                    actions.append(action)
                action = most_common(actions)
            else:
                action = self.get_human_input()

            obs, reward, done, info = self.env.step(action)
            reward_sum += reward
            if done and end_frame < 0:
                end_frame = frame + 60
            if frame == end_frame:
                return info["result"]

            self.clock.tick(self.fps)
            pygame.event.pump()

    def run_n_times(self, n):
        results = {'won': 0, 'draw': 0, 'lost': 0}
        for _ in range(n):
            reward = self.run_once()
            results[reward] += 1
            print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--enemy-path', help= 'path to enemy model (.zip file)', required=True)
    parser.add_argument('-p', '--player-path', help= '(optional) path to player model')
    args = vars(parser.parse_args())

    enemy_model = ModelContainer(args['enemy_path'])
    if args['player_path'] is not None:
        player_model = ModelContainer(args['player_path'])
    else:
        player_model = ModelContainer('dummy')

    env = GameEnv(enemy_model, 1, player_model.obs_space, enemy_model.obs_space, 1, balanced_positioning=True, game_length=1200)
    env.step_size = 1
    env.goal_height *= 3/4 # smaller goal to make it easier for humans to defend ;)

    game_visualiser = GameVisualiser(env, player_model)
    game_visualiser.run_n_times(100)
