

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


class GridActions(Action):
    def __init__(self):
        super().__init__(action_ids=['UP', 'DOWN', 'LEFT', 'RIGHT'])


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
