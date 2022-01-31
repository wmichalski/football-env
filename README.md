# Football environment

It is an implementation of a custom football-like environment which can be used for RL-related research. OpenAI Gym's convention is followed with appropriate ```env.step()``` and ```env.reset()``` functions.

## Installation

```pip install -r requirements.txt```

Run the command above in order to install required dependencies. Using Python 3.8 is recommended.

## Bots

Pre-trained models can be found in the ```bots``` directory with short descriptions in the ```bots/README.md```.

## Playing the game

The game can be played either in interactive mode (human vs bot) or in non-interactive testing mode (bot vs bot). 

``` python run_game.py -e path_to_enemy_model [-p path_to_another_model] [-m mapsize] [-t time]```

If the path to another model is not specified, the game runs in interactive mode. Map size can be defined (float scalar, with 1.0 being default) as well as length of a single game (time in seconds, with 20 being default).

## Ranking module

``` python generate_ranking.py [-p path_to_match_history]```

Tool used for simulating 1000 matches between models, saving the match history to a file and creating Elo history graph. The match simulation process can be skipped by providing a path to a pre-generated json match history.

## Training models

``` python run_training.py -a (PPO/DQN) [-m mapsize] [-e number of epochs/steps] [--predefined]```

There are two modes available:
- Training a model for the two goals single-player variant. Mapsize and training length can be defined by the user.
- Training a model for the enhanced self-play variant. Full process of learning is performed.
  
For both cases appropriate data is saved to logs which can be previewed using tensorboard. 

Appropriate changes in code can be made for managing more sophisticated scenarios due to the code being open source. Also, please note that the DQN training is available just for the time performance analysis and DQN models cannot be used in a simulation.
