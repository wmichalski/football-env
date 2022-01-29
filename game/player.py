import numpy as np

class Player:
    def __init__(self, x, y, radius):
        self.coords = np.array([x, y])
        self.radius = radius
        self.kick_radius = radius+10
        self.velocity = np.array([0.0001, 0.0001])
        self.slow_param = 0.9

    def reset(self, x, y):
        self.coords = np.array([x, y])
        self.velocity = np.array([0.0001, 0.0001])

    def update_velocity(self):
        self.velocity = self.velocity * self.slow_param

    def apply_velocity(self):
        self.coords = self.coords + self.velocity