import numpy as np
import logging

from grid_world import GridWorld
from planning_agents import QIteration
from learning_agents import OnMonteCarlo, QLearning, SARSA
from visualization import visualize_utilities, visualize_returns

from configs import configure_world_options


def obtain_optimal_values(world, error, gamma, store_path, states_of_interest=None):
    q_planner = QIteration(
        world=world,
        debug=True,
        error=error,
        gamma=gamma
    )
    q_planner.solve()
    q_values = q_planner.get_debug_info()
    q_planner.visualize(store_path=store_path)
    rets = []
    for i in range(0, 100):
        returns = q_planner.play_optimally()
        rets.append(returns)
    returns = np.sum(rets) / len(rets)

    utilities = {}

    for percept, qs in q_values.items():

        if states_of_interest is None or percept in states_of_interest:
            utilities[percept] = np.max(qs).flatten()[0]

    return utilities, returns


def plot_ready_utils(optimal_utilities, debug_freq, num_episodes):

    plot_rdy_utils = {}
    for percept, utility in optimal_utilities.items():
        axis = range(0, num_episodes + debug_freq, debug_freq)
        plot_rdy_utils[percept] = [(np.ones(num_episodes // debug_freq + 1)) * utility, axis]

    return plot_rdy_utils


def plot_ready_returns(optimal_returns, debug_freq, num_episodes):
    axis = range(0, num_episodes + debug_freq, debug_freq)
    return [np.ones(num_episodes // debug_freq+1) * optimal_returns, axis]


def get_agent(tag, world, gamma, max_steps, num_ep, epsilon, alpha, decay, debug_freq):

    if tag == 'Q':
        agent = QLearning(
            world=world,
            debug=True,
            gamma=gamma,
            max_steps=max_steps,
            num_episodes=num_ep,
            epsilon=epsilon,
            step=alpha,
            decay=decay,
            debug_freq=debug_freq
        )
    elif tag == 'ESARSA':
        agent = SARSA(
            world=world,
            debug=True,
            gamma=gamma,
            max_steps=max_steps,
            num_episodes=num_ep,
            epsilon=epsilon,
            step=alpha,
            decay=decay,
            expected=True,
            debug_freq=debug_freq
        )
    elif tag == 'SARSA':
        agent = SARSA(
            world=world,
            debug=True,
            gamma=gamma,
            max_steps=max_steps,
            num_episodes=num_ep,
            epsilon=epsilon,
            step=alpha,
            decay=decay,
            expected=False,
            debug_freq=debug_freq
        )
    elif tag == 'MC-Soft':
        agent = OnMonteCarlo(
            world=world,
            debug=True,
            gamma=gamma,
            max_steps=max_steps,
            num_episodes=num_ep,
            epsilon=epsilon,
            explore_starts=False,
            debug_freq=debug_freq
        )
    elif tag == 'MC-ES':
        agent = OnMonteCarlo(
            world=world,
            debug=True,
            gamma=gamma,
            max_steps=max_steps,
            num_episodes=num_ep,
            epsilon=None,
            explore_starts=True,
            debug_freq=debug_freq
        )
    else:
        agent = OnMonteCarlo(
            world=world,
            debug=True,
            gamma=gamma,
            max_steps=max_steps,
            num_episodes=num_ep,
            epsilon=epsilon,
            explore_starts=True,
            debug_freq=debug_freq
        )

    return agent


def main():
    # Test world
    WORLD_OPTION = 2
    USE_DYNAMIC_DEFAULT = True

    STORE_PATH = '/Users/djordje/ML/personal/RL/rl_projects/grid_world_playground/experimentation/test/'

    # Planning ground truth params
    ERROR = 0.00001

    # Learning Agent Params
    GAMMA = 0.9
    EPSILON = 0.3
    ALPHA = 0.01
    DECAY = False

    NUM_EP = 10000
    MAX_STEPS = 1000

    # Debug Params
    DEBUG_FREQ = 100

    STATES_OF_INTEREST = {(5, 0), (5, 2), (3, 2), (3, 3), (3, 4), (3, 5), (4, 5), (0, 2), (1, 3)}

    # Agent tags: ['Q', 'SARSA', 'ESARSA', 'MC-ES', 'MC-Soft', 'MC-ES-Soft']
    AGENTS_TO_TEST = ['Q', 'ESARSA', 'MC-ES', 'MC-ES-Soft']

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    action_probs, special_nodes, grid_dims, start_pos, default_r = configure_world_options(option=WORLD_OPTION)
    if USE_DYNAMIC_DEFAULT is True:
        default_r = - 1/MAX_STEPS

    grid_world = GridWorld(
        start_pos=start_pos,
        action_probs=action_probs,
        grid_dims=grid_dims,
        default_reward=default_r,
        special_nodes=special_nodes,
        heuristic=True
    )

    ground_truth_u, ground_truth_ret = obtain_optimal_values(
        world=grid_world,
        error=ERROR,
        gamma=GAMMA,
        store_path=STORE_PATH,
        states_of_interest=STATES_OF_INTEREST
    )

    ground_truth_u_plt = plot_ready_utils(
        optimal_utilities=ground_truth_u,
        debug_freq=DEBUG_FREQ,
        num_episodes=NUM_EP
    )

    ground_truth_ret_plt = plot_ready_returns(
        optimal_returns=ground_truth_ret,
        debug_freq=DEBUG_FREQ,
        num_episodes=NUM_EP
    )

    comparison_utilities = {}
    comparison_returns = {}

    for agent_tag in AGENTS_TO_TEST:
        utility_arg = {}

        agent = get_agent(
            tag=agent_tag,
            world=grid_world,
            gamma=GAMMA,
            max_steps=MAX_STEPS,
            num_ep=NUM_EP,
            epsilon=EPSILON,
            alpha=ALPHA,
            decay=DECAY,
            debug_freq=DEBUG_FREQ
        )

        logger.info('%s: Running.\n', agent_tag)

        agent.solve()
        if STORE_PATH is not None:
            agent.visualize(store_path=STORE_PATH)

        utilities = agent.export_utility_history(state_subset=STATES_OF_INTEREST)
        returns = agent.export_returns_history()

        detailed_tag = agent.get_model_name()
        comparison_utilities[agent_tag] = utilities
        comparison_returns[agent_tag] = returns

        utility_arg[detailed_tag] = utilities

        logger.info('%s Visualizing utility convergence.\n', agent_tag)
        visualize_utilities(
            ground_truth_dict=ground_truth_u_plt,
            utility_history_dicts=utility_arg,
            store_path=STORE_PATH,
            tag=detailed_tag + '_state_plt'
        )

    logger.info('Visualizing utility convergence comparison.\n')
    visualize_utilities(
        ground_truth_dict=ground_truth_u_plt,
        utility_history_dicts=comparison_utilities,
        store_path=STORE_PATH,
        tag='agent_utility_comparison'
    )

    logger.info('Visualizing returns convergence comparison.\n')
    visualize_returns(
        ground_truth=ground_truth_ret_plt,
        returns_dict=comparison_returns,
        store_path=STORE_PATH,
        tag='agent_returns_comparison'

    )


if __name__ == '__main__':
    main()



