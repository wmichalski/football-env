# Football environment

It is an implementation of a custom football-like environment which can be used for RL-related research. OpenAI Gym's convention is followed with appropriate ```env.step()``` and ```env.reset()``` functions.

## Installation

```pip install -r requirements.txt```

Run the command above in order to install required dependencies.

## Bots

Pre-trained models can be found in the ```bots``` directory with short descriptions in the ```bots/README.md```.

## Playing the game

The game can be played either in interactive mode (human vs bot) or in non-interactive testing mode (bot vs bot). 

``` python run_game.py -e path_to_enemy_model [-p path_to_another_model]```

If the path to another model is not specified, the game runs in interactive mode.

## Ranking module

``` python generate_ranking.py [-p path_to_match_history]```

Tool used for simulating 1000 matches between models, saving the match history to a file and creating Elo history graph. The match simulation process can be skipped by providing a path to a pre-generated match history.

## Training models

- ``` python dqn_train.py ```   
  Script which trains a model for the two goals small-pitch single-player variant. Left as a proof that it does work (slower than expected though).
- ``` python ppo_train.py ```  
  Script which trains a model for the enhanced self-play variant. Full process of learning is performed.

For both cases appropriate data is saved to logs which can be previewed using tensorboard.



