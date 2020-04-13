import numpy as np
import logging

from grid_world import GridWorld
from learning_agents import *

# configure universal params
STORE_PATH = '/Users/djordje/ML/personal/RL/rl_projects/grid_world_playground/' \
                 'experimentation/learning_algorithms/'
MAX_REWARD = 1
MIN_REWARD = -10
DEFAULT_REWARD = -1

# set which grid world to use among options below,
# 0 for 'SIMPLE', 1 for 'DESTINY', 2 for 'CHAOTIC'
WORLD_OPTION = 2

# Planning Algorithms config
GAMMA = 0.9
ERROR = 0.00001

# touch if you want to modify or add new Grid World configurations
WORLD_OPTIONS = ['SIMPLE', 'DESTINY', 'CHAOTIC', 'TEST']
SIMPLE_GRID_WORLD = {
    'transition_probs': [1, 0, 0],
    'grid_dims': (6, 6),
    'start_pos': (5, 0),
    'special_nodes': [
        {
            'position': (2, 5),
            'reward': MAX_REWARD,
            'terminal': True
        },
        {
            'position': (4, 1),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (3, 1),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (2, 1),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (4, 3),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (5, 3),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (0, 4),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (1, 4),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (2, 4),
            'reward': np.nan,
            'terminal': False
        },
    ]
}

MULTIPLE_DESTINY_GRID_WORLD = {
    'transition_probs': [1, 0, 0],
    'grid_dims': (6, 6),
    'start_pos': (5, 0),
    'special_nodes': [
        {
            'position': (2, 5),
            'reward': MAX_REWARD,
            'terminal': True
        },
        {
            'position': (4, 4),
            'reward': MIN_REWARD,
            'terminal': True
        },
        {
            'position': (4, 1),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (3, 1),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (2, 1),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (4, 3),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (5, 3),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (0, 4),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (1, 4),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (2, 4),
            'reward': np.nan,
            'terminal': False
        },
    ]
}

CHAOTIC_GRID_WORLD = {
    'transition_probs': [0.5, 0.25, 0.25],
    'grid_dims': (6, 6),
    'start_pos': (5, 0),
    'special_nodes': [
        {
            'position': (2, 5),
            'reward': MAX_REWARD,
            'terminal': True
        },
        {
            'position': (4, 4),
            'reward': MIN_REWARD,
            'terminal': True
        },
        {
            'position': (4, 1),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (3, 1),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (2, 1),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (4, 3),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (5, 3),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (0, 4),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (1, 4),
            'reward': np.nan,
            'terminal': False
        },
        {
            'position': (2, 4),
            'reward': np.nan,
            'terminal': False
        },
    ]
}

TEST_GRID_WORLD = {
    'transition_probs': [0.8, 0.1, 0.1],
    'grid_dims': (3, 4),
    'start_pos': (2, 0),
    'special_nodes': [
        {
            'position': (0, 3),
            'reward': MAX_REWARD,
            'terminal': True
        },
        {
            'position': (1, 3),
            'reward': MIN_REWARD,
            'terminal': True
        },
        {
            'position': (1, 1),
            'reward': np.nan,
            'terminal': False
        }
    ]
}


def configure_world_options():
    if WORLD_OPTION == 0:
        action_probs = SIMPLE_GRID_WORLD['transition_probs']
        special_nodes = SIMPLE_GRID_WORLD['special_nodes']
        grid_dims = SIMPLE_GRID_WORLD['grid_dims']
        start_pos = SIMPLE_GRID_WORLD['start_pos']
    elif WORLD_OPTION == 1:
        action_probs = MULTIPLE_DESTINY_GRID_WORLD['transition_probs']
        special_nodes = MULTIPLE_DESTINY_GRID_WORLD['special_nodes']
        grid_dims = MULTIPLE_DESTINY_GRID_WORLD['grid_dims']
        start_pos = MULTIPLE_DESTINY_GRID_WORLD['start_pos']
    elif WORLD_OPTION == 2:
        action_probs = CHAOTIC_GRID_WORLD['transition_probs']
        special_nodes = CHAOTIC_GRID_WORLD['special_nodes']
        grid_dims = CHAOTIC_GRID_WORLD['grid_dims']
        start_pos = CHAOTIC_GRID_WORLD['start_pos']
    else:
        action_probs = TEST_GRID_WORLD['transition_probs']
        special_nodes = TEST_GRID_WORLD['special_nodes']
        grid_dims = TEST_GRID_WORLD['grid_dims']
        start_pos = TEST_GRID_WORLD['start_pos']

    return action_probs, special_nodes, grid_dims, start_pos


def main():
    logging.basicConfig(level=logging.INFO)
    action_probs, special_nodes, grid_dims, start_pos = configure_world_options()

    grid_world = GridWorld(
        start_pos=start_pos,
        action_probs=action_probs,
        grid_dims=grid_dims,
        default_reward=DEFAULT_REWARD,
        special_nodes=special_nodes,
        heuristic=False
    )

    # epsilon 0.3
    # exp_start_mc = OnMonteCarlo(
    #     world=grid_world,
    #     debug=False,
    #     gamma=0.9,
    #     epsilon=0.3,
    #     num_episodes=10000,
    #     explore_starts=False,
    #     num_steps=100
    # )
    # exp_start_mc.solve()
    # exp_start_mc.visualize(store_path=STORE_PATH)

    exp_start_mc = OffMonteCarlo(
        world=grid_world,
        debug=False,
        gamma=0.9,
        epsilon=0.5,
        num_episodes=100,
        num_steps=100
    )
    exp_start_mc.solve()
    exp_start_mc.visualize(store_path=STORE_PATH)


if __name__ == '__main__':
    main()

