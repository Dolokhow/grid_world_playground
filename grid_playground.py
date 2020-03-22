from world import GridWorld
from ai_agents import BreadthlyCooper, JohnnyDeppth, AStarIsClimbing
import numpy as np
import logging

# configure universal params
STORE_PATH = '/Users/djordje/ML/personal/RL/rl_projects/block_world_q_learning_scratch/' \
                 'experimentation/search_algorithms/1_SIMPLE'
MAX_REWARD = 1
MIN_REWARD = -10
START_POS = (5, 0)
DEFAULT_REWARD = -1

# set which grid world to use among options below,
# 0 for 'SIMPLE', 1 for 'DESTINY', 2 for 'CHAOTIC'
WORLD_OPTION = 0

# touch if you want to modify or add new Grid World configurations
WORLD_OPTIONS = ['SIMPLE', 'DESTINY', 'CHAOTIC']
SIMPLE_GRID_WORLD = {
    'transition_probs': [1, 0, 0],
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


def main():
    logging.basicConfig(level=logging.INFO)

    if WORLD_OPTION == 0:
        action_probs = SIMPLE_GRID_WORLD['transition_probs']
        special_nodes = SIMPLE_GRID_WORLD['special_nodes']
    elif WORLD_OPTION == 1:
        action_probs = MULTIPLE_DESTINY_GRID_WORLD['transition_probs']
        special_nodes = MULTIPLE_DESTINY_GRID_WORLD['special_nodes']
    else:
        action_probs = CHAOTIC_GRID_WORLD['transition_probs']
        special_nodes = CHAOTIC_GRID_WORLD['special_nodes']

    grid_world = GridWorld(
        start_pos=START_POS,
        action_probs=action_probs,
        grid_dims=(6, 6),
        default_reward=DEFAULT_REWARD,
        special_nodes=special_nodes,
        heuristic=True
    )
    grid_world.print_info()
    grid_world.visualize_heuristic(store_path=STORE_PATH)

    # UNINFORMED SEARCH

    # Breadth-First Search
    breadth_first_agent = BreadthlyCooper(world=grid_world, debug=True)
    breadth_first_agent.search()
    breadth_first_agent.visualize(store_path=STORE_PATH)

    # Branch-And-Bound Search
    branch_and_bound_agent = BreadthlyCooper(world=grid_world, debug=True, b_bound=True)
    branch_and_bound_agent.search()
    branch_and_bound_agent.visualize(store_path=STORE_PATH)

    # Depth-First Search
    depth_first_agent = JohnnyDeppth(world=grid_world, debug=True)
    depth_first_agent.search()
    depth_first_agent.visualize(store_path=STORE_PATH)

    # INFORMED SEARCH

    # Greedy-Best-First Search
    best_first_agent = AStarIsClimbing(world=grid_world, debug=True, alg=1)
    best_first_agent.search()
    best_first_agent.visualize(store_path=STORE_PATH)

    # Hill-Climbing Search
    hill_climbing_agent = AStarIsClimbing(world=grid_world, debug=True, alg=2)
    hill_climbing_agent.search()
    hill_climbing_agent.visualize(store_path=STORE_PATH)

    # A* Search
    a_star_agent = AStarIsClimbing(world=grid_world, debug=True)
    a_star_agent.search()
    a_star_agent.visualize(store_path=STORE_PATH)


if __name__ == '__main__':
    main()
