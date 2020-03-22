from abc import ABC, abstractmethod
from collections import deque
from world import World, GridWorld
from utils import PerceptViz
import logging
import os
import numpy as np


class SearchTreeNode:
    def __init__(self, percept, parent=None, action=None, path_cost=0, depth=0, heuristic=None):
        self.percept = percept
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = depth
        self.heuristic = heuristic


class AI(ABC):
    _world: World

    def __init__(self, world, debug=False, tag='AI'):
        """
        Root class for all AI algorithms that are implemented.
        :param world: World in which agent lives.
        :param debug: Whether to store additional info useful for debugging.
        All examples can handle any world as long as it subclasses abstract World.
        Available actions are requested from the world.
        """
        self._world = world
        self._actions = world.get_available_actions()
        self._start_pos = world.get_default_start()
        self._debug = debug

        self._tag = tag
        self._logger = logging.getLogger(os.path.join(__name__, self._tag))
        # Delete line below, reminder for TD Agents!
        # self._uniform_percepts = world.get_available_percepts()

    def _take_action(self, percept, action_id):
        """
        Implements interaction with the world. Calls World.next_position().
        :return: next state, next percept, a Python list of values
        """
        next_state = self._world.next_position(percept=percept, action_id=action_id)
        return next_state

    @abstractmethod
    def _select_next_action(self):
        """
        Implements decision making logic of an AI agent.
        For example, in case of searching algorithms it discerns breadth first from depth first by changing FIFO/LIFO
        logic, or in case of Q learning it can implement a stochastic epsilon greedy policy.
        :return: next action
        """
        pass

    @abstractmethod
    def get_debug_info(self):
        """
        Returns additional info useful for debugging.
        :return: Depends on subclass.
        """
        pass

    @abstractmethod
    def visualize(self, store_path):
        """
        Passes relevant info to the world to visualize. May utilize self.get_debug_info. Passed value must be in
        dictionary {(percept tuple): PerceptViz object}. Calls world.visualize_solution.
        :return: None.
        """


class TreeSearch(AI, ABC):
    """
    Actually a graph search. All search algorithms implemented from
    Artificial Intelligence, A Modern Approach by Russel & Norvig.
    Variable names kept as close to original as possible.
    """
    def __init__(self, world, use_heuristic=False, debug=False, tag='Search-Graph'):
        """

        :param world: World in which agent lives. See AI class.
        :param debug: Whether to store additional info useful for debugging and visualization
        :param use_heuristic: Whether to use heuristic function from the world or not.
        :var self._fringe Python deque (double sided queue) object containing SearchTree nodes to be expanded.
        :var self._closed Python dictionary {percept: SearchTreeNode}. Used when determining when to close certain path.
        See Russel & Norwig for more details.
        :var self._next_node Immediate next SearchTreeNode node to be expanded by self._expand_next_states function.
        :var self._expanded_nodes List of SearchTreeNode elements in the current expansion, to be added to self._fringe
        by subclassed self._add_next_nodes.
        :var self._final_path To be set when method search() is completed.
        :var self._expansion_order Order of node visits, python list of percepts. Used only when self._debug is True.
        A list of percepts from start_state to terminal_state.
        """
        super().__init__(world=world, debug=debug, tag=tag)
        start = SearchTreeNode(percept=self._world.get_default_start())
        self._fringe = deque([start])
        self._closed = {start.percept}
        self._next_node: SearchTreeNode = start
        self._expanded_nodes = []
        self._use_heuristic = use_heuristic

        self._final_path = None
        self._visitation_order = []

        self._search_called = False

    @staticmethod
    def _solution(node):
        """
        Backtracks solution from terminal node.
        :param node: Terminal node.
        :return: Path of percepts / states from root node to terminal node.
        """
        solution = [node.percept]
        while node.parent is not None:
            solution.append(node.parent.percept)
            node = node.parent
        solution.reverse()
        return solution

    @abstractmethod
    def _add_next_nodes(self):
        """
        Implements the logic of adding new nodes to self._fringe. If FIFO is used then we have breadth first search,
        if LIFO is used we have depth first search.
        :return: None.
        """
        pass

    @abstractmethod
    def _order_expanded_nodes(self):
        """
        Orders currently expanded nodes to be later added to self._fringe by self._add_next_nodes. Used with informed
        searches where we have some heuristic which can tell us which currently expanded nodes are best.
        :return: None. Modifies self._expanded_nodes order.
        """
        pass

    @abstractmethod
    def _reorder_fringe(self):
        """
        Reorders self._fringe. Potentially useful if one wants to change FIFO / LIFO order in light of new information.
        Breadth first and depth first algorithms never use this function, ie, it does nothing.
        :return: None. Reorders self._fringe.
        """
        pass

    def get_debug_info(self):
        return self._visitation_order

    def _expand_next_states(self):

        for action_id in self._actions:
            percept = self._world.next_position(
                percept=self._next_node.percept,
                action_id=action_id
            )

            _, cost = self._world.get_cost(percept=percept)

            # to be used by heuristic aware agents (informed search algorithms)
            if self._use_heuristic:
                heuristic = self._world.get_heuristic(percept=percept, action_id=action_id)
            else:
                heuristic = None

            path_cost = self._next_node.path_cost + cost
            path_length = self._next_node.depth + 1

            new_node = SearchTreeNode(
                percept=percept,
                action=action_id,
                parent=self._next_node,
                path_cost=path_cost,
                depth=path_length,
                heuristic=heuristic
            )
            self._expanded_nodes.append(new_node)

        self._order_expanded_nodes()
        self._add_next_nodes()
        self._reorder_fringe()

    def search(self):
        """
        Search method general for all TreeSearch algorithms.
        :return: State / percept list, from start_state to terminal state.
        """
        self._search_called = True
        while True:
            if bool(self._fringe) is False:
                return -1
            else:
                node: SearchTreeNode = self._select_next_action()
                if self._debug is True:
                    self._visitation_order.append(node.percept)
                is_terminal, _ = self._world.get_cost(percept=node.percept)
                if is_terminal is True:
                    solution = self._solution(node=node)
                    self._final_path = solution
                    return solution
                else:
                    self._next_node = node
                    self._expand_next_states()

    def visualize(self, store_path):
        percept_viz = {}

        self._world.visualize_solution(
            percept_viz=percept_viz,
            store_path=store_path,
            tag=self._tag
        )

        if self._search_called is False:
            self._logger.warning('Nothing to visualize. Search method never called. Will visualize empty World.')
            return

        # if there is debug info available
        if len(self._visitation_order) != 0:
            for index in range(len(self._visitation_order)):
                percept = self._visitation_order[index]
                if self._final_path is not None and percept in self._final_path:
                    color = (160, 85, 129)
                else:
                    color = (159, 85, 160)

                percept_viz[percept] = PerceptViz(
                    percept=percept,
                    single_value=index,
                    color=color
                )
        else:
            if self._final_path is not None:
                for index in range(len(self._final_path)):
                    percept = self._visitation_order[index]
                    percept_viz[percept] = PerceptViz(
                        percept=percept,
                        single_value=index,
                        color=(160, 85, 129)
                    )

        for unvisited_but_expanded in self._fringe:
            if unvisited_but_expanded.percept in percept_viz:
                raise ValueError('Oops')
            percept_viz[unvisited_but_expanded.percept] = PerceptViz(
                percept=unvisited_but_expanded.percept,
                color=(151, 124, 234)
            )

        self._world.visualize_solution(
            percept_viz=percept_viz,
            store_path=store_path,
            tag=self._tag
        )


# Either Best-First or Branch-and-Bound
class BreadthlyCooper(TreeSearch):
    def __init__(self, world, b_bound=False, debug=False):
        """
        :param world: World parameter from abstract AI.
        :param b_bound: whether Breadth First should be used, or Branch-and-Bound method. Branch-and-Bound reorders
        self._fringe according to least node.path_cost.
        :param debug: Debug parameter from abstract AI.
        """
        self._b_bound = b_bound
        tag = 'Branch-And-Bound' if self._b_bound is True else 'Breadth-First'
        super().__init__(world=world, debug=debug, use_heuristic=False, tag=tag)

    def _select_next_action(self):
        return self._fringe.pop()

    def _add_next_nodes(self):
        for node in self._expanded_nodes:
            if node.percept not in self._closed:
                # Does not check if newly discovered path is better than the previous one.
                self._fringe.appendleft(node)
                self._closed.add(node.percept)
        self._expanded_nodes = []

    def _order_expanded_nodes(self):
        if self._debug is False:
            self._expanded_nodes = sorted(self._expanded_nodes, key=lambda node: node.path_cost)

    def _reorder_fringe(self):
        if self._b_bound is True:
            self._fringe = deque(sorted(self._fringe, key=lambda node: node.path_cost, reverse=True))


# Depth-First or Depth-Limited search
class JohnnyDeppth(TreeSearch):
    def __init__(self, world, debug=False, limit=None):
        self._limit = limit
        tag = 'Dept-Limited' if self._limit is not None else 'Depth-First'
        super().__init__(world=world, debug=debug, use_heuristic=False, tag=tag)

    def _select_next_action(self):
        return self._fringe.pop()

    def _add_next_nodes(self):
        for node in self._expanded_nodes:
            if node.percept not in self._closed:
                if self._limit is None or node.depth < self._limit:
                    # Does not check if newly discovered path is better than the previous one for now.
                    self._fringe.append(node)
                    self._closed.add(node.percept)
        self._expanded_nodes = []

    def _order_expanded_nodes(self):
        if self._debug is False:
            self._expanded_nodes = sorted(self._expanded_nodes, key=lambda node: node.path_cost, reverse=True)
        else:
            # not important, just adjusts the priority of newly expanded nodes to force taking actions
            # in action order. Depth first logic is held even if line below is commented, however actions
            # will take reverse priority
            self._expanded_nodes.reverse()

    def _reorder_fringe(self):
        pass


# Greedy-Best-First or Hill-Climbing or A*, depending on the alg param:
# 1 - Greedy-Best-First
# 2 - Hill-Climbing
# 3 - A*
class AStarIsClimbing(TreeSearch):

    def __init__(self, world, alg=3, check_closed=False, debug=False):
        """

        :param world: world from abstract AI.
        :param alg: Choice of algorithm. integer 1 for Best-First, 2 for for Hill-Climbing, 3 for A*.
        :param check_closed: Used when A* is selected. If heuristic world implements is not consistent guarantees
        optimality. If heuristic is consistent should be left false to speed up algorithm. See consistency pg. 101
        Artificial Intelligence a Modern Approach.
        :param debug: debug from abstract AI.
        """

        err_msg = None

        if alg in {1, 3}:
            self.__modify_fringe = True
            if alg == 1:
                self.__use_paths = False
                tag = 'Greedy-Best-First'
            else:
                self.__use_paths = True
                tag = 'A*'
        elif alg == 2:
            self.__modify_fringe = False
            self.__use_paths = False
            tag = 'Hill-Climbing'
        else:
            err_msg = 'Allowed values for parameter are either integers 1, 2, or 3.'
            tag = ''

        super().__init__(world=world, debug=debug, use_heuristic=True, tag=tag)
        if err_msg is not None:
            self._logger.error('Unable to create class. Invalid parameter alg.')
            raise ValueError(err_msg)

        heuristic = self._world.heuristic_available()
        if heuristic is None:
            self._logger.error('Informed algorithms require the world to have a heuristic function. World has none.')
            raise ValueError('Informed algorithms require the world to have a heuristic function. World has none.')

        self._check_closed = check_closed

    def _select_next_action(self):
        return self._fringe.pop()

    def _add_next_nodes(self):
        for node in self._expanded_nodes:
            if node.percept not in self._closed:
                if self._check_closed is False:
                    # Does not check if newly discovered path is better than the previous one.
                    self._fringe.append(node)
                    self._closed.add(node.percept)
                elif self.__use_paths is True:
                    # Means we are using A* and self._check_closed is True:
                    for index in range(len(self._fringe)):
                        previous_node = self._fringe[index]
                        if previous_node.percept == node.percept:
                            if previous_node.path_cost + previous_node.heuristic > node.path_cost + node.heuristic:
                                self._fringe[index] = node
                            break

        self._expanded_nodes = []

    def _order_expanded_nodes(self):
        if self.__modify_fringe is False:
            # Hill-Climbing
            self._expanded_nodes = sorted(self._expanded_nodes, key=lambda node: node.heuristic, reverse=True)
        else:
            # A* or Greedy-Best-First
            # serves same purpose as reverse in depth-first search
            self._expanded_nodes.reverse()

    def _reorder_fringe(self):
        if self.__modify_fringe is True:
            if self.__use_paths is True:
                # A*
                self._fringe = deque(sorted(self._fringe, key=lambda node: node.heuristic + node.path_cost, reverse=True))
            else:
                # Greedy-Best-First
                self._fringe = deque(sorted(self._fringe, key=lambda node: node.heuristic, reverse=True))


def test_ai():
    store_path = '/Users/djordje/ML/personal/RL/rl_projects/block_world_q_learning_scratch/experimentation/search_algorithms'
    max_reward = 1
    # min_reward = -10
    start_pos = (5, 0)
    # Only deterministic environments should be considered by basic search algorithms
    action_probs = [1, 0, 0]
    default_reward = -1
    # Only one state can be terminal with basic search based algorithms
    special_nodes = [
        {
            'position': (2, 5),
            'reward': max_reward,
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

    logging.basicConfig(level=logging.INFO)

    grid_world = GridWorld(
        start_pos=start_pos,
        action_probs=action_probs,
        grid_dims=(6, 6),
        default_reward=default_reward,
        special_nodes=special_nodes,
        heuristic=True
    )
    grid_world.print_info()
    grid_world.visualize_heuristic(store_path=store_path)

    # UNINFORMED SEARCH

    # Breadth-First Search
    breadth_first_agent = BreadthlyCooper(world=grid_world, debug=True)
    breadth_first_agent.search()
    breadth_first_agent.visualize(store_path=store_path)

    # Branch-And-Bound Search
    branch_and_bound_agent = BreadthlyCooper(world=grid_world, debug=True, b_bound=True)
    branch_and_bound_agent.search()
    branch_and_bound_agent.visualize(store_path=store_path)

    # Depth-First Search
    depth_first_agent = JohnnyDeppth(world=grid_world, debug=True)
    depth_first_agent.search()
    depth_first_agent.visualize(store_path=store_path)

    # INFORMED SEARCH

    # Greedy-Best-First Search
    best_first_agent = AStarIsClimbing(world=grid_world, debug=True, alg=1)
    best_first_agent.search()
    best_first_agent.visualize(store_path=store_path)

    # Hill-Climbing Search
    hill_climbing_agent = AStarIsClimbing(world=grid_world, debug=True, alg=2)
    hill_climbing_agent.search()
    hill_climbing_agent.visualize(store_path=store_path)

    # A* Search
    a_star_agent = AStarIsClimbing(world=grid_world, debug=True)
    a_star_agent.search()
    a_star_agent.visualize(store_path=store_path)

    agent = BreadthlyCooper(world=grid_world, debug=True)
    agent.visualize(store_path=store_path)

    special_nodes.append({
            'position': (4, 4),
            'reward': -1,
            'terminal': True
        })
    multiple_terminal_grid = GridWorld(
        start_pos=start_pos,
        action_probs=action_probs,
        grid_dims=(6, 6),
        default_reward=default_reward,
        special_nodes=special_nodes,
        heuristic=True,
        tag='Multiple-Grid-World'
    )
    multiple_terminal_grid.visualize_heuristic(store_path=store_path)


if __name__ == '__main__':
    test_ai()








