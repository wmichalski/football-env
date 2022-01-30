import math

import numpy as np


class Ball:
    def __init__(self, x, y, radius):
        self.coords = np.array([x, y])
        self.radius = radius
        # avoiding a 0 in velocity vector to simplify math
        self.velocity = np.array([0.001, 0.001])
        self.in_goal = False
        self.in_own_goal = False
        self.slow_param = 0.975

    def reset(self, x, y):
        self.coords = np.array([x, y])
        self.velocity = np.array([0.001, 0.001])
        self.in_goal = False
        self.in_own_goal = False

    def update_velocity(self):
        self.velocity = self.velocity * self.slow_param

        # slow down if the ball got too fast
        speed = math.sqrt(np.sum(self.velocity**2))
        if speed > 20:
            self.velocity = self.velocity * 20/speed

        # avoiding a 0 in velocity vector to simplify math
        self.velocity = np.where(self.velocity == 0, 0.001, self.velocity)

    def apply_velocity(self):
        speed = math.sqrt(np.sum(self.velocity**2))

        # buggy colisions fix
        if speed > 10:
            self.coords = self.coords + self.velocity * 10/speed
        else:
            self.coords = self.coords + self.velocity
