import argparse

# had some CUDA conflict when tensorflow (DQN) was loaded first
from train.ppo_train import PPOManager
from train.dqn_train import DQNManager

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm',
                        help='algorithm to be used: DQN/PPO', required=True)
    parser.add_argument('-m', '--mapsize',
                        help='mapsize used during the training', default=0.4, type=float)
    parser.add_argument('-e', '--epochs',
                        help='epochs for DQN/timesteps for PPO', default=1000000, type=int)
    parser.add_argument('--predefined',
                        help='run advanced predefined training for PPO', default=False, action='store_true')

    args = vars(parser.parse_args())

    if args['algorithm'] == "PPO":
        ppo_manager = PPOManager(args['predefined'], args['mapsize'], args['epochs'])
        if args["predefined"]:
            ppo_manager.predefined_training()
        else:
            ppo_manager.custom_training()
    elif args['algorithm'] == "DQN":
        if args["predefined"]:
            print("running DQN algorithm, ignoring PPO predefined flag")
        dqn_manager = DQNManager(map_scale=args['mapsize'])
        dqn_manager.run_training()
    else:
        raise argparse.ArgumentTypeError(f"unexpected algorithm name: {args['algorithm']}")
