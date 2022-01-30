import sys

from stable_baselines3 import PPO

class ModelContainer:
    def __init__(self, path):
        custom_objects = self.python_version_fix()
        self.path = path
        self.obs_space = 12
        if path != 'dummy':
            self.model = PPO.load(path, custom_objects=custom_objects)
            self.obs_space = self.get_obs_space()
        self.elo = 1500

    def get_obs_space(self):
        return self.model.policy.observation_space.shape[0]

    def python_version_fix(self):
        newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
        custom_objects = {}
        if newer_python_version:
            custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            }

        return custom_objects
