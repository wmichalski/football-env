from game.gymgame_multiplayer import GameEnv
from utils.elo_module import EloModule
from utils.model_container import ModelContainer
import argparse
import matplotlib.pyplot as plt
import random
import json
import os
import time

def most_common(lst):
    return max(set(lst), key=lst.count)

class RankingSystem():
    def __init__(self):
        self.players = []
        self.elo_module = EloModule()
        self.history = []

    def get_rand_pair(self):
        id1 = random.randrange(0, len(self.players))
        id2 = random.randrange(0, len(self.players))
        while id1 == id2:
            id2 = random.randrange(0, len(self.players))

        return self.players[id1], self.players[id2]

    def populate(self):
        # more models can be added
        self.players.append(ModelContainer(
            "bots/sp2-4.zip"))
        self.players.append(ModelContainer(
            "bots/sp2-5.zip"))
        self.players.append(ModelContainer(
            "bots/sp2-6.zip"))

    def print_scoreboard(self, winner, loser, diff):
        self.players.sort(key=lambda x: x.elo, reverse=False)
        print("=================")
        for player in self.players:
            if player == winner:
                print(
                    "\x1b[6;30;42m{:.2f}   +{:.2f}   {}\x1b[0m".format(player.elo, diff, player.path))
            elif player == loser:
                print("\x1b[6;37;41m{:.2f}   {:.2f}   {}\x1b[0m".format(
                    player.elo, -diff, player.path))
            else:
                print("{:.2f}   -----   {}".format(player.elo, player.path))

    def play_match(self, p1, p2):
        env = GameEnv(p2, 1, p1.obs_space, p2.obs_space,
                      map_scale=1, balanced_positioning=True, game_length=900)
        env.step_size = 3
        wins, losses, draws = 0, 0, 0
        for _ in range(10):
            done = False
            obs = env.reset()
            while not done:
                action, _ = p1.model.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)

            if info["result"] == "won":
                wins += 1
            elif info["result"] == "draw":
                draws += 1
            else:
                losses += 1

        print(wins, losses, draws)
        return wins, losses, draws

    def add_history(self, p1, p2, wins, draws, losses):
        self.history.append({'p1': p1.path, 'p2': p2.path,
                            'wins': wins, 'draws': draws, 'losses': losses})

    def save_history(self, path):
        with open(path, 'w') as f:
            json.dump(self.history, f)

    def load_and_process_json_history(self, path):
        with open(path, 'r') as f:
            data = json.load(f)

        # loading players from the jsonfile
        player_paths = []
        for match in data:
            if match['p1'] not in player_paths:
                player_paths.append(match['p1'])
            if match['p2'] not in player_paths:
                player_paths.append(match['p2'])

        players = []
        for player_path in player_paths:
            players.append(ModelContainer(player_path))

        elo_history = dict()
        for player in players:
            elo_history[player.path] = []

        for match in data:
            # mapping paths from json to player structures
            for player in players:
                if match['p1'] == player.path:
                    p1 = player
                if match['p2'] == player.path:
                    p2 = player

            p1.elo, p2.elo, _, _ = self.elo_module.process_match(
                p1, p2, match['wins'], match['draws'], match['losses'])
            for player in players:
                elo_history[player.path].append(player.elo)

        # sort by final elo scores
        elo_history = dict(
            sorted(elo_history.items(), key=lambda item: item[1][-1], reverse=True))
        for key, values in elo_history.items():
            x = [i for i in range(len(values))]
            y = values
            plt.plot(x, y, label=os.path.split(key)[1])

        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                   mode="expand", borderaxespad=0, ncol=3)

        plt.legend().set_draggable(True)
        plt.show()

    def run(self):
        self.populate()

        for i in range(1000):
            p1, p2 = self.get_rand_pair()
            wins, losses, draws = self.play_match(p1, p2)

            p1.elo, p2.elo, has_p1_won, diff = self.elo_module.process_match(
                p1, p2, wins, draws, losses)

            if has_p1_won:
                self.print_scoreboard(p1, p2, diff)
            else:
                self.print_scoreboard(p2, p1, diff)

            self.add_history(p1, p2, wins, draws, losses)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        path = 'match-history-' + timestr + '.json'
        self.save_history(path)
        self.load_and_process_json_history(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to .json ranking file')
    args = parser.parse_args()

    rank_system = RankingSystem()

    if args.path is not None:
        rank_system.load_and_process_json_history(args.path)
    else:
        rank_system.run()
