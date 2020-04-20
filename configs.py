import numpy as np

MAX_REWARD = 1
MIN_REWARD = -100

# touch if you want to modify or add new Grid World configurations
WORLD_OPTIONS = ['SIMPLE', 'DESTINY', 'CHAOTIC', 'TEST']
SIMPLE_GRID_WORLD = {
    'transition_probs': [1, 0, 0],
    'grid_dims': (6, 6),
    'start_pos': (5, 0),
    'default_reward': -1,
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
    'default_reward': -1,
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
    'default_reward': -1,
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
    'default_reward': -1,
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


def configure_world_options(option):
    if option == 0:
        action_probs = SIMPLE_GRID_WORLD['transition_probs']
        special_nodes = SIMPLE_GRID_WORLD['special_nodes']
        grid_dims = SIMPLE_GRID_WORLD['grid_dims']
        start_pos = SIMPLE_GRID_WORLD['start_pos']
        default = SIMPLE_GRID_WORLD['default_reward']
    elif option == 1:
        action_probs = MULTIPLE_DESTINY_GRID_WORLD['transition_probs']
        special_nodes = MULTIPLE_DESTINY_GRID_WORLD['special_nodes']
        grid_dims = MULTIPLE_DESTINY_GRID_WORLD['grid_dims']
        start_pos = MULTIPLE_DESTINY_GRID_WORLD['start_pos']
        default = MULTIPLE_DESTINY_GRID_WORLD['default_reward']
    elif option == 2:
        action_probs = CHAOTIC_GRID_WORLD['transition_probs']
        special_nodes = CHAOTIC_GRID_WORLD['special_nodes']
        grid_dims = CHAOTIC_GRID_WORLD['grid_dims']
        start_pos = CHAOTIC_GRID_WORLD['start_pos']
        default = CHAOTIC_GRID_WORLD['default_reward']
    else:
        action_probs = TEST_GRID_WORLD['transition_probs']
        special_nodes = TEST_GRID_WORLD['special_nodes']
        grid_dims = TEST_GRID_WORLD['grid_dims']
        start_pos = TEST_GRID_WORLD['start_pos']
        default = TEST_GRID_WORLD['default_reward']

    return action_probs, special_nodes, grid_dims, start_pos, default