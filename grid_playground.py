import numpy as np
import logging

from grid_world import GridWorld
from search_agents import BreadthlyCooper, JohnnyDeppth, AStarIsClimbing
from planning_agents import QIteration, ValueIteration, PolicyIteration

# configure universal params
STORE_PATH = '/Users/djordje/ML/personal/RL/rl_projects/grid_world_playground/' \
                 'experimentation/planning_algorithms/3_CHAOTIC'
MAX_REWARD = 1
MIN_REWARD = -100
START_POS = (5, 0)
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
    elif WORLD_OPTION == 1:
        action_probs = MULTIPLE_DESTINY_GRID_WORLD['transition_probs']
        special_nodes = MULTIPLE_DESTINY_GRID_WORLD['special_nodes']
        grid_dims = MULTIPLE_DESTINY_GRID_WORLD['grid_dims']
    elif WORLD_OPTION == 2:
        action_probs = CHAOTIC_GRID_WORLD['transition_probs']
        special_nodes = CHAOTIC_GRID_WORLD['special_nodes']
        grid_dims = CHAOTIC_GRID_WORLD['grid_dims']
    else:
        action_probs = TEST_GRID_WORLD['transition_probs']
        special_nodes = TEST_GRID_WORLD['special_nodes']
        grid_dims = TEST_GRID_WORLD['grid_dims']

    return action_probs, special_nodes, grid_dims


def main():
    logging.basicConfig(level=logging.INFO)
    action_probs, special_nodes, grid_dims = configure_world_options()

    grid_world = GridWorld(
        start_pos=START_POS,
        action_probs=action_probs,
        grid_dims=grid_dims,
        default_reward=DEFAULT_REWARD,
        special_nodes=special_nodes,
        heuristic=True
    )
    # grid_world.print_info()
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

    # ITERATIVE PLANNING ALGORITHMS

    q_iter_agent = QIteration(
        world=grid_world,
        debug=True,
        gamma=GAMMA,
        error=ERROR
    )
    q_iter_agent.iteration()
    q_iter_agent.visualize(store_path=STORE_PATH)

    v_iter_agent = ValueIteration(
        world=grid_world,
        debug=True,
        gamma=GAMMA,
        error=ERROR
    )
    v_iter_agent.iteration()
    v_iter_agent.visualize(store_path=STORE_PATH)

    p_iter_agent = PolicyIteration(
        world=grid_world,
        debug=True,
        gamma=GAMMA,
        error=ERROR,
        policy_shift_max=10
    )
    p_iter_agent.iteration()
    p_iter_agent.visualize(store_path=STORE_PATH)


if __name__ == '__main__':
    main()
