import numpy as np
from core import PerceptViz, TransitionModel, Action, World
from visualization import GridVisualizer


class GridActions(Action):
    def __init__(self):
        super().__init__(action_ids=['UP', 'DOWN', 'LEFT', 'RIGHT'])


class GridTransitions(TransitionModel):
    def __init__(self, actions, action_probs):
        """
        Initializes GridWorld transition model.
        :param action_probs: Probabilities associated with performing orthogonal actions as per definition
        of GridWorld problem.
        """
        super().__init__(actions=actions)
        self._action_cutoffs = [action_probs[0]]
        for index in range(1, len(action_probs)):
            self._action_cutoffs.append(self._action_cutoffs[index-1]+action_probs[index])

    @staticmethod
    def _orth_1(action_ind):

        if action_ind == 0:
            action_ind += 2
        elif action_ind == 1:
            action_ind += 2
        elif action_ind == 2:
            action_ind -= 1
        else:
            action_ind -= 3

        return action_ind

    @staticmethod
    def _orth_2(action_ind):

        if action_ind == 0:
            action_ind += 3
        elif action_ind == 1:
            action_ind += 1
        elif action_ind == 2:
            action_ind -= 2
        else:
            action_ind -= 2

        return action_ind

    def _stochastic_action(self, action_id):
        """
        See abstract method.
        :param action_id: Intended action id.
        :return: Taken action id, either desired action, or orthogonal to it defined by some probabilities
        given by action_probs in constructor.
        """
        rand = np.random.uniform()

        if rand < self._action_cutoffs[0]:
            return action_id
        else:
            # we avoid string comparison, but nothing smarter implemented
            # relies on the fact that there are 4 actions
            action_ind = self._actions.get_action_by_id(action_id=action_id)
            if rand < self._action_cutoffs[1]:
                action_ind = self._orth_1(action_ind=action_ind)
            else:
                action_ind = self._orth_2(action_ind=action_ind)
            return self._actions.get_action_by_ind(action_ind=action_ind)

    def get_alt_outcomes(self, action_id):
        possible_actions = []

        action_ind = self._actions.get_action_by_id(action_id=action_id)
        possible_actions.append([action_id, self._action_cutoffs[0]])

        alt_1_prob = self._action_cutoffs[1] - self._action_cutoffs[0]
        if alt_1_prob > 0:
            alt_1 = self._actions.get_action_by_ind(action_ind=self._orth_1(action_ind=action_ind))
            possible_actions.append([alt_1, alt_1_prob])

        alt_2_prob = self._action_cutoffs[2] - self._action_cutoffs[1]
        if alt_2_prob > 0:
            alt_2 = self._actions.get_action_by_ind(action_ind=self._orth_2(action_ind=action_ind))
            possible_actions.append([alt_2, alt_2_prob])

        return possible_actions


class GridWorld(World):

    def __init__(self, start_pos, action_probs, grid_dims, default_reward,
                 special_nodes=None, heuristic=False, tag=None, visualization=True):
        """
        :param start_pos: Starting agent position, if left None, agents are positioned at random.
        :param action_probs: Action probabilities for GridWorld. Alternative action probs need not be symmetrical.
        :param grid_dims: Dimensions of GridWorld. Python tuple.
        :param default_reward: Default reward for each state in the world.
        :param special_nodes: Python list of dictionaries reflecting each special GridWorld location (node).
        Each dictionary contains boolean 'terminal', indicating if the state is terminal, 'reward' indicating reward
        at that state, and 'position' indicating matrix position of element in GridWorld. If 'reward' is np.nan then
        that position cannot be entered and is a wall. If special_nodes is left None then grid world with set dimensions
        will be generated at random.
        :param heuristic: Whether heuristic should be generated. PLEASE NOTE: If heuristic is True then only one
        terminal node must exist in the world!!!
        """

        self.__action_probs = action_probs
        self.__grid_dims = grid_dims
        self.__special_nodes = special_nodes
        self.__walls = set([])
        self.__terminal_states = set([])
        self.__default = default_reward
        tag = 'Grid-World' if tag is None else tag

        if special_nodes is None:
            self._generate_random_world()
        else:
            self._parse_special_nodes()

        super().__init__(
            start_pos=start_pos,
            heuristic=heuristic,
            tag=tag,
            visualization=visualization
        )

        start = (3, 1)
        if self._start_percept is None:
            if start not in self.__walls and start not in self.__terminal_states:
                self._start_percept = start
            else:
                self._logger.error('Unrecoverable error: Specified starting position is either a '
                                   'wall or a terminal state according to special nodes argument. '
                                   'Change either special nodes or starting position.')
                raise ValueError('Specified starting position is either a wall or a terminal state '
                                 'according to special nodes argument. '
                                 'Change either special nodes or starting position.')

        if self._complex_vis is True:
            self._visualizer = GridVisualizer()

    # TODO: Implement this later
    def _generate_random_world(self):
        """
        Generates random world if self._special_nodes is None.
        :return: sets self._world to the generated random world. Sets self.__special_nodes.
        """
        pass

    def _parse_special_nodes(self):
        """
        Parses self.__special_nodes and stores them internally into self.__walls and self.__terminal_states var.
        :return: None
        """
        for node in self.__special_nodes:
            if node['terminal'] is True:
                self.__terminal_states.add(node['position'])
            elif np.isnan(node['reward']):
                self.__walls.add(node['position'])

    def _initialize_world(self):
        world = np.ones(self.__grid_dims) * self.__default
        for node in self.__special_nodes:
            world[node['position'][0], node['position'][1]] = node['reward']
        return world

    def _initialize_actions(self):
        return GridActions()

    def _initialize_transition_model(self):
        transition_model = GridTransitions(
            actions=self._actions,
            action_probs=self.__action_probs
        )
        return transition_model

    def _initialize_heuristic(self):
        if len(self.__terminal_states) > 1:
            self._logger.warning('Found multiple terminal states. Searching agents are not meant to solve '
                                 'these problems and implemented heuristic may be inadequate.')

        # terminal_state = list(self.__terminal_states)[0]
        # terminal_row = terminal_state[0]
        # terminal_col = terminal_state[1]

        terminal_states = list(self.__terminal_states)

        action_ids = self._actions.get_action_ids()

        heuristic_dims = list(self.__grid_dims)
        heuristic_dims.append(len(action_ids))
        heuristic_dims = tuple(heuristic_dims)

        # row, column, action
        heuristic_matrix = np.ones(heuristic_dims)
        # for each position in grid world we set the heuristic h(a, s) to be the minimal distance to goal
        # if the agent acts optimally from that point onward to the goal
        for row in range(0, heuristic_dims[0]):
            for col in range(0, heuristic_dims[1]):

                if (row, col) in self.__terminal_states:
                    _, reward = self.get_reward(percept=self._map_percepts_to_state(percept=(row, col)))
                    if reward > 0:
                        heuristic_matrix[row, col] = np.zeros(len(action_ids))
                    else:
                        heuristic_matrix[row, col] = np.ones(len(action_ids)) * \
                                                     (self.__grid_dims[0] + self.__grid_dims[1])
                elif (row, col) in self.__walls:
                    values = np.empty(len(action_ids))
                    values[:] = np.nan
                    heuristic_matrix[row, col] = values
                else:
                    for action_id in action_ids:
                        action_ind = self._actions.get_action_by_id(action_id=action_id)
                        action_row, action_col = self.next_position(percept=(row, col), action_id=action_id)

                        distances_to_terminal = []
                        for terminal_state in terminal_states:
                            _, reward = self.get_reward(percept=terminal_state)
                            distance_to_terminal = abs(terminal_state[0] - action_row) + \
                                               abs(terminal_state[1] - action_col) + 1
                            if reward > 0:
                                # enforce realistic distance to positive goal
                                distance_to_goal = distance_to_terminal
                            else:
                                # discourage action, discourages more the closer to negative terminal state agent is
                                distance_to_goal = self.__grid_dims[0] + self.__grid_dims[1] - distance_to_terminal

                            # if we've hit a wall and stayed in same position
                            if (row, col) == (action_row, action_col):
                                distance_to_goal += 1

                            distances_to_terminal.append(distance_to_goal)

                        distance_to_goal = np.min(distances_to_terminal)
                        heuristic_matrix[row, col, action_ind] = distance_to_goal

        return heuristic_matrix

    def _map_percepts_to_state(self, percept):
        return percept

    def get_reward(self, percept):
        key = self._map_percepts_to_state(percept=percept)
        if key in self.__walls:
            self._logger.error('Agent has entered the wall, this should never happen.')
            raise ValueError('Passed percept is wall. Agent should have never reached this percept.')

        if key in self.__terminal_states:
            terminal = True
        else:
            terminal = False
        reward = self._world[key]
        return terminal, reward

    def get_cost(self, percept):
        is_terminal, reward = self.get_reward(percept=percept)
        return is_terminal, -reward

    def get_heuristic(self, percept, action_id):
        action_ind = self._actions.get_action_by_id(action_id=action_id)
        row = percept[0]
        col = percept[1]
        if self._heuristic is not None:
            return self._heuristic[row, col, action_ind]
        else:
            return None

    def _next_position(self, percept, action_id):
        key = self._map_percepts_to_state(percept=percept)

        row = key[0]
        col = key[1]

        if action_id == 'UP':
            row -= 1
        elif action_id == 'RIGHT':
            col += 1
        elif action_id == 'LEFT':
            col -= 1
        else:
            row += 1

        if 0 <= row < self.__grid_dims[0] and 0 <= col < self.__grid_dims[1] and not np.isnan(self._world[row, col]):
            next_key = (row, col)
        else:
            next_key = key

        return next_key

    def get_available_percepts(self):
        percepts = []
        for row in range(self.__grid_dims[0]):
            for col in range(self.__grid_dims[1]):
                key = (row, col)
                if key not in self.__walls and key not in self.__terminal_states:
                    percepts.append(key)
        return percepts

    def get_default_start(self):
        return self._start_percept

    def print_info(self):

        print('GridWorld')

        print('Grid dimensions: ', self.__grid_dims)
        print('\n')

        print('States / percepts: \n')
        print(self.get_available_percepts())
        print('\n')

        print('Walls states: \n')
        print(self.__walls)
        print('\n')

        print('Terminal states: \n')
        print(self.__terminal_states)
        print('\n')

        print('Special nodes: \n')
        print(self.__special_nodes)
        print('\n')

        if self._heuristic is not None:
            print('Heuristic values: \n')
            print(self._heuristic)

    def _pack_world_info(self):
        world_info = {
            'rows': self.__grid_dims[0],
            'cols': self.__grid_dims[1],
            'tag': self._tag,
            'walls': self.__walls,
            'terminal_states': {},
            'start_percept': self._start_percept
        }
        for percept in list(self.__terminal_states):
            _, world_info['terminal_states'][percept] = self.get_reward(percept=percept)

        return world_info

    def visualize_solution(self, percept_viz, store_path, tag=None):

        if self._visualizer is not None:
            world_info = self._pack_world_info()

            self._visualizer.visualize_solution(
                world_info=world_info,
                percept_viz=percept_viz,
                store_path=store_path,
                tag=tag
            )
        else:
            for elem in percept_viz:
                print(elem.percept)
                print(elem.action_values)
            self.print_info()

    def visualize_heuristic(self, store_path):
        if self._visualizer is not None:
            world_info = self._pack_world_info()

            percept_viz = {}
            for row_ind in range(self.__grid_dims[0]):
                for col_ind in range(self.__grid_dims[1]):
                    percept = (row_ind, col_ind)
                    if percept not in self.__walls:
                        percept_viz_obj = PerceptViz(
                            percept=percept,
                            action_values=self._heuristic[row_ind, col_ind, :]
                        )
                        percept_viz[percept] = percept_viz_obj
            self._visualizer.visualize_heuristic(
                world_info=world_info,
                percept_viz=percept_viz,
                store_path=store_path,
                tag=self._tag
            )
        else:
            if self._heuristic is True:
                self.print_info()
            else:
                self._logger.warning('Visualization of heuristic requested, but found none.')


def test_world():
    max_reward = 1
    min_reward = -10
    start_pos = (3, 1)
    action_probs = [1, 0, 0]
    default_reward = -1
    special_nodes = [
        {
            'position': (0, 3),
            'reward': max_reward,
            'terminal': True
        },
        {
            'position': (1, 3),
            'reward': min_reward,
            'terminal': False
        },
        {
            'position': (2, 1),
            'reward': np.nan,
            'terminal': False
        }
    ]

    grid_world = GridWorld(
        start_pos=start_pos,
        action_probs=action_probs,
        grid_dims=(4, 4),
        default_reward=default_reward,
        special_nodes=special_nodes,
        heuristic=True
    )
    grid_world.print_info()


if __name__ == '__main__':
    test_world()
