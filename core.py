import os
import logging
from abc import ABC, abstractmethod
import numpy as np


class Action:
    def __init__(self, action_ids):
        """
        The purpose of this whole class is to increase readability. It maps action indexes to action ids and stores
        them in a uniform way.
        :param action_ids: List of string action ids that are available to the agent.
        """
        self.__actions_by_id = {}
        self.__actions_by_index = {}
        self.__actions_len = len(action_ids)
        self.__action_ids = action_ids

        for index in range(len(action_ids)):
            self.__actions_by_id[action_ids[index]] = index
            self.__actions_by_index[index] = action_ids[index]

    def get_action_by_ind(self, action_ind):
        return self.__actions_by_index[action_ind]

    def get_action_by_id(self, action_id):
        return self.__actions_by_id[action_id]

    def get_actions_len(self):
        return self.__actions_len

    def get_action_ids(self):
        return self.__action_ids


class TransitionModel(ABC):
    def __init__(self, actions):
        self._actions = actions
        self._actions_len = actions.get_actions_len()
        np.random.seed(123)

    def take_action(self, action_id):
        """
        Returns  stochastic action taken.
        :param action_id: Action intended to be taken.
        :return: Taken action_id.
        """
        action_id = self._stochastic_action(action_id=action_id)
        return action_id

    @abstractmethod
    def _stochastic_action(self, action_id):
        """
        Used when transition model is stochastic as in GridWorld problem. Based on some probability desired
        action is selected. In other cases some other action is selected based on some logic which this function
        implements.
        :param action_id: Desired action id.
        :return: Action id corresponding to the action to be taken.
        """
        pass

    @abstractmethod
    def get_alt_outcomes(self, action_id):
        """
        Used when transition model is stochastic as in GridWorld problem. Given the intended action_id, returns
        all possible action_ids taken, as dictated by the transition model.
        :param action_id: Desired action id.
        :return: Action ids of all potential outcomes.
        """


class PerceptViz:
    def __init__(self, percept, action_values=[], single_value=None, color=(255, 255, 255)):
        """
        The purpose of this class is to be the bridge between the AI agent and World classes, intended for visualizing
        solutions obtained with the AI agents within the World. Should be only used for visualization purposes.
        Each of these parameters may or may not be used by the world when visualizing solution.
        :param percept: Python tuple. Percept to be visualized.
        :param action_values: Python list. For each action available in the world, single float value can be passed. THE
        ORDER OF THE LIST MUST MATCH THE ORDER OF THE ACTIONS FROM THE WORLD. This should be the case by default as
        agents have the same order of actions as the world.
        :param single_value: Python float. May be useful for some algorithms.
        :param color: Color of the field in the world.
        """
        self.percept = percept
        self.action_values = action_values
        self.cv_color = color
        self.single_value = single_value


class World(ABC):
    def __init__(self, start_pos=None, heuristic=False, tag='Undefined-World', visualization=True):
        """
        :param heuristic: Boolean, whether heuristic should be generated. Heuristics are necessary for some informed
        search algorithms such as A* search. They are not useful to basic search algorithms or advanced learning
        algorithms like q learning.
        :param start_pos: Starting percept/state of the agent in the world.
        :param visualization: boolean value, whether complex visualization should be used. Complex visualization
        uses WorldVisualizer object defined in visualization.py. Implemented via command design pattern.
        :var self._world: Contains internal representation of the world, containing rewards.
        :var self._actions: Action object describing available actions. Agent will request actions from World.
        :var self._transition_model: Is of type TransitionModel, contains logic that maps actions into future states.
        :var self._start_percept: Starting percept,a python tuple,
        of agent in world, if left None then position is picked at random.
        :var self._heuristic: Maps each (state/percept, action) pair into a single value representing heuristic
        h(n=(a,s)).
        """
        self._tag = tag
        self._world = self._initialize_world()
        self._actions = self._initialize_actions()
        self._transition_model = self._initialize_transition_model()
        self._logger = logging.getLogger(os.path.join(__name__, self._tag))
        # Necessary for informed search algorithms
        if heuristic:
            self._heuristic = self._initialize_heuristic()
        else:
            self._heuristic = None

        self._start_percept = start_pos
        self._complex_vis = visualization
        # To be set by subclass
        self._visualizer: WorldVisualizer = None

    @abstractmethod
    def _initialize_actions(self):
        pass

    @abstractmethod
    def _initialize_transition_model(self):
        pass

    @abstractmethod
    def _initialize_world(self):
        pass

    @abstractmethod
    def _initialize_heuristic(self):
        pass

    @abstractmethod
    def _map_percepts_to_state(self, percept):
        """
        Maps current agent's perception, represented as percept, to a specific valid world (agent) state.
        This abstraction gives worlds ability to arbitrarily implement internal world.
        :param percept: Python tuple representing current agent's perception.
        :return: state from the world.
        """
        pass

    @abstractmethod
    def _next_position(self, percept, action_id):
        """
        Calculates agent's next position in the world. Maps action to next percept. Is called either directly by agent or self.take_action.
        :param action_id: Action intelligent agent or algorithm has decided to take.
        :param percept: Current percept, a python tuple of values.
        :return: Next percept, a python tuple of values, obtained by enacting action in the world.
        """

    @abstractmethod
    def get_reward(self, percept):
        """
        Returns reward based on current percept. Depends on internal world implementation and
        self._map_percepts_to_state implementation.
        :param percept: Python tuple corresponding to values of current percept.
        :return: (is_terminal, reward_signal) tuple corresponding to state_id.
        """
        pass

    @abstractmethod
    def get_cost(self, percept):
        """
        Returns cost based associated with current percept. Depends on internal world implementation and
        self._map_percepts_to_state implementation. Similar to reward, but opposite. Agents usually either maximize
        reword or reduce cost. In some worlds these two values may be direct opposites.
        :param percept: Python tuple corresponding to values of current percept.
        :return: (is_terminal, cost_signal) tuple corresponding to state_id.
        """

    @abstractmethod
    def get_heuristic(self, percept, action_id):
        """
        Method to be used by informed search algorithms
        :param percept: Python tuple corresponding to values of current percept.
        :param action_id:  Action intelligent agent or algorithm considers to take.
        :return: Returns heuristic value h(percept, action_id) if heuristic exists, or None if it doesn't.
        """
        pass

    @abstractmethod
    def get_available_percepts(self):
        """
        Returns all available percepts in the world.
        Should be used by agents whose implementation would greatly benefit knowing beforehand all available percepts.
        Do not confuse with rewards, the agents should not know all available rewards beforehand.
        :return: Python list of tuples, each tuple indicates percepts belonging to one internal state.
        List of tuples as a structure is mandatory so agents can be decoupled from any particular world.
        """
        pass

    @abstractmethod
    def get_default_start(self):
        """
        Utility method. Returns default starting position in the world.
        :return: Python tuple.
        """
        pass

    @abstractmethod
    def print_info(self):
        """
        Prints to console useful info about self in standardized form.
        :return: None
        """
        pass

    def next_position(self, percept, action_id):
        """
        Takes stochastic action, according to internal transition model. Intended action is given by action_id,
        actual action is determined by transition model.
        :param percept: See self._next_position()
        :param action_id: See self._next_position()
        :return: next percept, after enacting action_id obtained from transition model
        """
        action_id = self._transition_model.take_action(action_id=action_id)
        next_percept = self._next_position(percept=percept, action_id=action_id)
        return next_percept

    def take_action(self, percept, action_id):
        """
        See self.next_position. This method calculates next position and reaps reward for the agent.
        :param action_id: Action intelligent agent or algorithm has decided to take.
        :param percept: Current percept, a python tuple of values.
        :return: Reword in the state reached after enacting action_id in the world.
        """
        next_percept = self.next_position(percept=percept, action_id=action_id)
        is_terminal, reward = self.get_reward(percept=next_percept)
        return is_terminal, reward

    def get_action_outcomes(self, percept, action_id):
        possible_actions = self._transition_model.get_alt_outcomes(action_id=action_id)
        for action_prob in possible_actions:
            action_prob[0] = self._next_position(percept=percept, action_id=action_prob[0])
        return possible_actions

    def get_available_actions(self):
        """
        To be called by agents, returns available actions in the world.
        :return: Python list of action_ids.
        """
        return self._actions.get_action_ids()

    def heuristic_available(self):
        return self._heuristic

    def visualize_heuristic(self, store_path):
        """
        Visualizes heuristic if there is some.
        :return: By default just prints info about the world which includes heuristic.
        If complex visualizations are requested, WorldVisualized object visualizes the heuristic.
        Left for subclasses set appropriate visualizer object and redefine this method.
        """
        if self._heuristic is True:
            self.print_info()
        else:
            self._logger.warning('Visualization of heuristic requested, but found none.')

    def visualize_solution(self, percept_viz, store_path, tag=None):
        """
        Visualizes the solution.
        :param percept_viz: Python mapping tuple percept to PerceptViz object giving information about an agent solution.
        Can contain either all available percepts or just some subset.
        :param tag Agent's tag. String.
        :param store_path: Path where to store the visualization.
        :return: By default just prints info about the world and percept values.
        If complex visualizations are requested, WorldVisualized object visualizes solution.
        Left for subclasses set appropriate visualizer object and redefine this method.
        """

        for elem in percept_viz:
            print(elem.percept)
            print(elem.action_values)
        self.print_info()


class WorldVisualizer(ABC):
    def __init__(self, img_width, img_height, enforce_img_dims):
        """

        :param img_width: Image width in pixels to be stored as visualization.
        :param img_height: Image height in pixels to be stored as visualization.
        :param enforce_img_dims: Whether to enforce image dims or not. If image dims are enforced, image dimension
        will never exceed limits, but visualization may not be crisp. If they are not enforced, image will be sharp
        and readable, but may be too large. If one is creating a large world, it is recommended to opt for enforcing
        dims.
        """

        self._img_width = img_width
        self._img_height = img_height
        self._enforce_dims = enforce_img_dims
        self._tag = 'WorldVisualizer'
        self._logger = logging.getLogger(os.path.join(__name__, self._tag))

    @abstractmethod
    def visualize_solution(self, world_info, percept_viz, store_path, tag):
        pass

    @abstractmethod
    def visualize_heuristic(self, world_info, percept_viz, store_path, tag):
        pass


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

    @abstractmethod
    def solve(self):
        """
        Agent applies strategy to solve the given problem.
        :return: Left for subclasses to define. Either some form of a solution or sets some internal variable.
        """

