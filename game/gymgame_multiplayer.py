import math

import gym
import numpy as np
from gym import spaces

from game.ball import Ball
from game.player import Player

move_matrix = np.array(
    [[0, 0, 0.7, 0.7, 0.7, 0, -0.7, -0.7, -0.7, 0, 0, 0.7, 0.7, 0.7, 0, -0.7, -0.7, -0.7],
     [0, 0.7, 0.7, 0, -0.7, -0.7, -0.7, 0, 0.7, 0, 0.7, 0.7, 0, -0.7, -0.7, -0.7, 0, 0.7],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class GameEnv(gym.Env):
    def __init__(self, enemy_model=None, enemy_slow=1, p1_obs_space=None, p2_obs_space=None,
                 map_scale=1, balanced_positioning=False, game_length=600, singleplayer=False):
        super(GameEnv, self).__init__()
        self.step_size = 5
        self.game_length_cnt = 0
        self.game_length = game_length
        self.done = False

        self.p1_obs_space = p1_obs_space
        self.p2_obs_space = p2_obs_space

        self.map_scale = map_scale
        self.balanced_positioning = balanced_positioning
        self.lock_enemy = singleplayer  # used for single player trainings

        self.goal_height = 200 * self.map_scale
        self.map_height = 400 * self.map_scale
        self.map_width = 600 * self.map_scale
        self.display_width = 750 * self.map_scale
        self.display_height = 666 * self.map_scale

        # helper variables for simplyfing physics code
        self.half_mh = self.map_height/2
        self.half_mw = self.map_width/2
        self.half_dw = self.display_width/2
        self.half_dh = self.display_height/2

        self.normalisation_array = np.array([self.half_dw, self.half_dh])

        self.action_space = spaces.Discrete(18)
        if p1_obs_space is not None:
            if p1_obs_space == 8:
                self.observation_space = spaces.Box(
                    low=np.array([-1, -1, -2, -2, -1, -1, -2, -2]),
                    high=np.array([1, 1, 2, 2, 1, 1, 2, 2]),
                    shape=(8,),
                    dtype=np.float32)
            elif p1_obs_space == 12:
                self.observation_space = spaces.Box(
                    low=np.array([-1, -1, -2, -2, -1, -1, -2, -2, -1, -1, -2, -2]),
                    high=np.array([1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]),
                    shape=(12,),
                    dtype=np.float32)

        # ugly temporary hardcode, to be fixed:
        self.player1 = Player(int(self.display_width*0.75),
                              int(self.display_height*0.5), radius=20)
        self.player2 = Player(int(self.display_width*0.25),
                              int(self.display_height*0.5), radius=20)
        self.players = [self.player1, self.player2]

        self.ball = Ball(int(self.display_width*0.5),
                         int(self.display_height*0.5), radius=12)

        if enemy_model is not None:
            self.enemy_model = enemy_model.model
        else:
            self.enemy_model = None
        self.enemy_slow = enemy_slow

        self.reset()

    def reset(self):
        if self.balanced_positioning:
            rand_player_height = int(self.display_height*np.random.random())

            self.player1.reset(int(0.75*self.display_width),
                               rand_player_height)
            self.player2.reset(int(0.25*self.display_width),
                               rand_player_height)

            self.ball.reset(int(0.5 * self.display_width),
                            int(self.half_dh - self.half_mh + np.random.random()*self.map_height))
        else:
            self.player1.reset(int(np.random.random()*self.display_width),
                               int(self.display_height*np.random.random()))
            self.player2.reset(int(np.random.random()*self.display_width),
                               int(self.display_height*np.random.random()))

            self.ball.reset(int(self.half_dw - self.half_mw + np.random.random()*self.map_width),
                            int(self.half_dh - self.half_mh + np.random.random()*self.map_height))

        self.game_length_cnt = 0
        self.done = False

        if self.lock_enemy:
            self.player2.reset(1000, 1000)

        return self.get_game_state_p1()

    def distance_between_two_points(self, player, ball):
        return math.sqrt(np.sum((ball.coords-player.coords)**2))

    def kick(self, player, ball):
        dist = self.distance_between_two_points(player, ball)
        if dist < 0.01:
            dist = 0.01
        if dist <= player.kick_radius + ball.radius:
            vector = player.coords - ball.coords
            ball.velocity = ball.velocity - vector/dist * 10

    def is_ball_close_to_walls(self, ball):
        if ball.coords[0] <= self.half_dw - self.half_mw + 2*ball.radius:
            return True

        if ball.coords[0] >= self.half_dw + self.half_mw - 2*ball.radius:
            return True

        if ball.coords[1] >= self.half_dh + self.half_mh - 2*ball.radius:
            return True

        if ball.coords[1] <= self.half_dh - self.half_mh + 2*ball.radius:
            return True

        return False

    def check_collisions(self, player, ball):
        dist = self.distance_between_two_points(player, ball)
        if dist < player.radius + ball.radius:
            ray = ball.velocity

            # norm vector connecting player's and ball's centers
            v = player.coords - ball.coords
            nrm = v / np.sqrt(np.sum(v**2))

            # managing specific case of ball being close to walls
            if self.is_ball_close_to_walls(self.ball):
                middle = (ball.coords + player.coords)/2
                ball.coords = middle-0.5*v*(player.radius+ball.radius)/dist
                player.coords = middle+0.5*v*(player.radius+ball.radius)/dist
            else:
                ball.coords = player.coords-v*(player.radius+ball.radius)/dist

            # if the ball hits the player, then it should bounce
            if not(ball.velocity[0] == 0 and ball.velocity[0] == 0):
                if angle_between(ray, nrm) < 1.57:
                    reflection = ray - (2 * (np.dot(ray, nrm)) * nrm)
                    ball.velocity = reflection

            # objects' positions should also impact the ball's final velocity
            player_velocity = np.sqrt(np.sum(player.velocity**2))
            nrm *= player_velocity

            ball.velocity = ball.velocity + \
                (player.velocity*7-nrm)*0.075  # empirically obtained values
            player.velocity = player.velocity * 0.9

    def check_borders_ball(self, ball):
        # left border
        if ball.coords[0] <= self.half_dw - self.half_mw + ball.radius:
            if ball.coords[1] > self.half_dh - self.goal_height/2 and ball.coords[1] < self.half_dh + self.goal_height/2:
                # if the ball is fully inside of the goal:
                if ball.coords[0] <= self.half_dw - self.half_mw - ball.radius:
                    ball.in_goal = True
            else:
                ball.coords[0] = self.half_dw - self.half_mw + ball.radius
                ball.velocity[0] *= -1

        # right border
        if ball.coords[0] >= self.half_dw + self.half_mw - ball.radius:
            if ball.coords[1] > self.half_dh - self.goal_height/2 and ball.coords[1] < self.half_dh + self.goal_height/2:
                # if the ball is fully inside of the goal:
                if ball.coords[0] >= self.half_dw + self.half_mw + ball.radius:
                    ball.in_own_goal = True
            else:
                ball.coords[0] = self.half_dw + self.half_mw - ball.radius
                ball.velocity[0] *= -1

        # bottom border
        if ball.coords[1] >= self.half_dh + self.half_mh - ball.radius:
            ball.coords[1] = self.half_dh + self.half_mh - ball.radius
            ball.velocity[1] *= -1

        # top border
        if ball.coords[1] <= self.half_dh - self.half_mh + ball.radius:
            ball.coords[1] = self.half_dh - self.half_mh + ball.radius
            ball.velocity[1] *= -1

    def check_borders_player(self, player):
        was_outside = 0
        # left border
        if player.coords[0] <= 0:
            player.coords[0] = 0
            player.velocity[0] = 0
            was_outside = 1

        # right border
        if player.coords[0] >= self.display_width:
            player.coords[0] = self.display_width
            player.velocity[0] = 0
            was_outside = 1

        # bottom border
        if player.coords[1] >= self.display_height:
            player.coords[1] = self.display_height
            player.velocity[1] = 0
            was_outside = 1

        # top border
        if player.coords[1] <= 0:
            player.coords[1] = 0
            player.velocity[1] = 0
            was_outside = 1

        return was_outside

    def get_enemy_action(self):
        if self.enemy_model is None:
            return [0, 0], 0
        obs = self.get_game_state_p2()
        obs *= -1
        action = self.enemy_model.predict(obs)[0]

        if np.random.random() > self.enemy_slow:
            action = np.random.randint(0, 18)

        move, kick = self.make_action(action)

        return move*-1*self.enemy_slow, kick

    def step(self, action):
        vel_change1, isKick1 = self.make_action(action)
        vel_change2, isKick2 = self.get_enemy_action()
        reward = 0

        self.game_length_cnt += self.step_size

        for _ in range(self.step_size):
            if isKick1:
                self.kick(self.player1, self.ball)

            if isKick2:
                self.kick(self.player2, self.ball)

            self.player1.velocity = self.player1.velocity + vel_change1
            self.player2.velocity = self.player2.velocity + vel_change2

            self.player1.update_velocity()
            self.player1.apply_velocity()

            self.player2.update_velocity()
            self.player2.apply_velocity()

            self.ball.update_velocity()
            self.ball.apply_velocity()

            for player in self.players:
                self.check_collisions(player, self.ball)
            # so interactive games dont end too soon
            if not (self.done and (self.ball.in_goal or self.ball.in_own_goal)):
                self.check_borders_ball(self.ball)
            self.check_borders_player(self.player1)
            self.check_borders_player(self.player2)

            reward += (1-self.ball.coords[0]/self.half_mw) * 0.5
            reward -= 1

            if self.ball.in_goal:
                self.done = True
                reward += 2000
                break

            if self.ball.in_own_goal:
                self.done = True
                reward -= 500
                break

            if self.game_length_cnt > self.game_length:
                self.done = True
                break

        observation = self.get_game_state_p1()

        result = ""
        if self.done:
            if self.ball.in_goal:
                result = "won"
            elif self.ball.in_own_goal:
                result = "lost"
            else:
                result = "draw"

        return observation, reward, self.done, {"result": result}

    def make_action(self, action):
        selected_move = move_matrix[:, action]
        return selected_move[0:2], selected_move[2]

    # ugly temporary hardcode, to be fixed:
    def get_game_state_p2(self):
        if self.p2_obs_space == 8:
            data = np.concatenate(
                ((self.player2.coords - self.normalisation_array) / self.normalisation_array, self.player2.velocity / 20,
                 (self.ball.coords - self.normalisation_array) / self.normalisation_array, self.ball.velocity / 20))
            data.reshape((1, 8))
        elif self.p2_obs_space == 12:
            data = np.concatenate(
                ((self.player2.coords - self.normalisation_array) / self.normalisation_array, self.player2.velocity / 20,
                 (self.player1.coords - self.normalisation_array) / self.normalisation_array, self.player1.velocity / 20,
                 (self.ball.coords - self.normalisation_array) / self.normalisation_array, self.ball.velocity / 20))
            data.reshape((1, 12))
        return data

    def get_game_state_p1(self):
        if self.p1_obs_space == 12:
            data = np.concatenate(
                ((self.player1.coords - self.normalisation_array) / self.normalisation_array, self.player1.velocity / 20,
                 (self.player2.coords - self.normalisation_array) / self.normalisation_array, self.player2.velocity / 20,
                 (self.ball.coords - self.normalisation_array) / self.normalisation_array, self.ball.velocity / 20))
            data.reshape((1, 12))
        elif self.p1_obs_space == 8:
            data = np.concatenate(
                ((self.player1.coords - self.normalisation_array) / self.normalisation_array, self.player1.velocity / 20,
                 (self.ball.coords - self.normalisation_array) / self.normalisation_array, self.ball.velocity / 20))
            data.reshape((1, 8))
        return data
