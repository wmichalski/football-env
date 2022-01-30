import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from game.gymgame_multiplayer import GameEnv
from utils.model_container import ModelContainer


class PPOManager():
    def __init__(self, is_mode_predefined, map_scale=0.4, steps=1000000):
        self.mapsize = map_scale
        self.steps = steps
        self.predefined = is_mode_predefined
        if self.predefined:
            print("running predefined multistage variant, overriding user's input")

    def custom_training(self):
        env = make_vec_env(GameEnv, n_envs=32, env_kwargs={'p1_obs_space': 8, 'map_scale': self.mapsize, 'singleplayer': True})
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='logs/')
        model.learn(total_timesteps=self.steps)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        model.save(f"ppo-{timestr}")

    def predefined_training(self):
        enemy_model = ModelContainer('bots/egoistic.zip')

        # STAGE 1
        env = make_vec_env(
            GameEnv, n_envs=32,
            env_kwargs={'enemy_model': enemy_model, 'enemy_slow': 0.8 * 0.4, 'p1_obs_space': 12,
                        'p2_obs_space': enemy_model.obs_space, 'map_scale': 0.4})
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='logs/')
        model.set_env(env)
        model.learn(total_timesteps=2000000, reset_num_timesteps=False)
        model.save("selfplay2-1")

        # STAGE 2
        env = make_vec_env(
            GameEnv, n_envs=32,
            env_kwargs={'enemy_model': enemy_model, 'enemy_slow': 0.8 * 0.6, 'p1_obs_space': 12,
                        'p2_obs_space': enemy_model.obs_space, 'map_scale': 0.6})
        model.set_env(env)
        model.learn(total_timesteps=4000000, reset_num_timesteps=False)
        model.save("selfplay2-2")

        # STAGE 3
        env = make_vec_env(
            GameEnv, n_envs=32,
            env_kwargs={'enemy_model': enemy_model, 'enemy_slow': 0.8 * 0.8, 'p1_obs_space': 12,
                        'p2_obs_space': enemy_model.obs_space, 'map_scale': 0.8})
        model.set_env(env)
        model.learn(total_timesteps=8000000, reset_num_timesteps=False)
        model.save("selfplay2-3")

        # STAGE 4
        env = make_vec_env(GameEnv, n_envs=32,
                           env_kwargs={'enemy_model': enemy_model, 'enemy_slow': 0.8, 'p1_obs_space': 12,
                                       'p2_obs_space': enemy_model.obs_space, 'map_scale': 1})
        model.set_env(env)
        model.learn(total_timesteps=16000000, reset_num_timesteps=False)
        model.save("selfplay2-4")

        # STAGE 5
        env = make_vec_env(GameEnv, n_envs=32,
                           env_kwargs={'enemy_model': enemy_model, 'enemy_slow': 0.85, 'p1_obs_space': 12,
                                       'p2_obs_space': enemy_model.obs_space, 'map_scale': 1})
        model.set_env(env)
        model.learn(total_timesteps=16000000, reset_num_timesteps=False)
        model.save("selfplay2-5")

        # STAGE 6
        env = make_vec_env(GameEnv, n_envs=32,
                           env_kwargs={'enemy_model': enemy_model, 'enemy_slow': 0.9, 'p1_obs_space': 12,
                                       'p2_obs_space': enemy_model.obs_space, 'map_scale': 1})
        model.set_env(env)
        model.learn(total_timesteps=16000000, reset_num_timesteps=False)
        model.save("selfplay2-6")

        env.close()
