import cv2
from abc import abstractmethod, ABC
import numpy as np
from utils import PerceptViz
import os


class WorldVisualizer(ABC):

    @abstractmethod
    def visualize_solution(self, world_info, percept_viz, store_path, tag):
        pass

    @abstractmethod
    def visualize_heuristic(self, world_info, percept_viz, store_path, tag):
        pass


class GridVisualizer(WorldVisualizer):
    def __init__(self):
        super().__init__()

    def visualize_heuristic(self, world_info, percept_viz, store_path, tag):
        self.visualize_solution(
            world_info=world_info,
            percept_viz=percept_viz,
            store_path=store_path,
            tag=tag
        )

    def visualize_solution(self, world_info, percept_viz, store_path, tag):

        # extract info from world_info
        grid_rows = world_info['rows']
        grid_cols = world_info['cols']
        world_tag = world_info['tag']
        walls = world_info['walls']
        # dictionary mapping terminal state percepts to rewards
        terminal_states = world_info['terminal_states']
        start_percept = world_info['start_percept']

        cell_width = 300
        cell_height = 300

        if tag is None:
            tag = 'unknown_model'

        # positions of action_values for each PerceptViz
        top_left = (int(cell_height * 0.01), int(cell_width * 0.01))
        bot_right = (int(cell_height * 0.99), int(cell_width * 0.99))

        up_pos = (int(0.4 * cell_width), int(0.2 * cell_height))
        down_pos = (int(0.4 * cell_width), int(0.8 * cell_height))
        left_pos = (int(0.15 * cell_width), int(0.5 * cell_height))
        right_pos = (int(0.65 * cell_width), int(0.5 * cell_height))

        arrow_up = [(int(0.5 * cell_width), int(0.6 * cell_height)),
                    (int(0.5 * cell_width), int(0.4 * cell_height))]
        arrow_down = [(int(0.5 * cell_width), int(0.4 * cell_height)),
                      (int(0.5 * cell_width), int(0.6 * cell_height))]
        arrow_left = [(int(0.6 * cell_width), int(0.47 * cell_height)),
                      (int(0.4 * cell_width), int(0.47 * cell_height))]
        arrow_right = [(int(0.4 * cell_width), int(0.47 * cell_height)),
                       (int(0.6 * cell_width), int(0.47 * cell_height))]

        # positions for single_value in perceptViz
        center_pos_negative = (int(0.3 * cell_width), int(0.5 * cell_height))
        center_pos_positive = (int(0.42 * cell_width), int(0.5 * cell_height))

        # position for final state rewards
        reward_pos = (int(0.6 * cell_width), int(0.3 * cell_height))

        arrows = [arrow_up, arrow_down, arrow_left, arrow_right]
        sequence = [up_pos, down_pos, left_pos, right_pos]

        # B, G, R
        color_base = [1, 90, 0]

        if bool(percept_viz) is not False:
            some_percept: PerceptViz = next(iter(percept_viz.values()))
            # if one action_value for single percept is not None, then all are non None
            # find max and min action_values among all percepts
            if len(some_percept.action_values) != 0:
                max_value = np.max(some_percept.action_values)
                min_value = np.min(some_percept.action_values)

                for _, percept_obj in percept_viz.items():
                    mx = np.max(percept_obj.action_values)
                    mn = np.min(percept_obj.action_values)

                    if mx > max_value:
                        max_value = mx

                    if mn < min_value:
                        min_value = mn
                color_range = abs(max_value - min_value)
            else:
                color_range = None
        else:
            # action_values not used, no need for color_range
            color_range = None

        row_concatenation = None

        for row_ind in range(grid_rows):
            column_concatenation = None

            for col_ind in range(grid_cols):

                blank_image = np.ones((cell_height, cell_width, 3), np.uint8)
                percept = (row_ind, col_ind)
                colored_field = False

                if percept in percept_viz:
                    percept_obj = percept_viz[percept]

                    # no action_values found
                    if color_range is None:

                        if percept_obj.cv_color is not None:
                            blank_image = blank_image * percept_obj.cv_color
                            blank_image = blank_image.astype(np.uint8)
                            colored_field = True

                        if percept_obj.single_value is not None:

                            if percept_obj.single_value > 0:
                                center_pos = center_pos_positive
                            else:
                                center_pos = center_pos_negative

                            if colored_field is True:
                                txt_color = (255, 255, 255)
                            else:
                                txt_color = (0, 0, 0)

                            cv2.putText(blank_image, str(round(percept_obj.single_value, 2)), center_pos,
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, txt_color, 2, cv2.LINE_AA)
                    else:
                        blank_image = blank_image * 255
                        blank_image = blank_image.astype(np.uint8)
                        # tag also indicates if we are calling this function from within visualize_heuristic func.
                        if tag == world_tag:
                            local_extrm_indexes = set(np.argwhere(
                                percept_obj.action_values == np.amin(percept_obj.action_values)).flatten())
                            # local_extrm_index = np.argmin(percept_obj.action_values)
                        else:
                            local_extrm_indexes = set(np.argwhere(
                                percept_obj.action_values == np.amax(percept_obj.action_values)).flatten())
                            # local_extrm_index = np.argmax(percept_obj.action_values)

                        for index in range(len(percept_obj.action_values)):

                            action_value = percept_obj.action_values[index]
                            action_value_text = str(round(action_value, 2))
                            pos = sequence[index]
                            color = color_base.copy()
                            color[0] = int(((action_value - min_value) / color_range) * 255)
                            cv2.putText(blank_image, action_value_text, pos,
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                            if index in local_extrm_indexes:
                                arrow_coords = arrows[index]
                                cv2.arrowedLine(blank_image, arrow_coords[0], arrow_coords[1],
                                                color=color, thickness=3, line_type=cv2.LINE_AA)
                else:
                    # walls or special states or none
                    if percept in walls:
                        blank_image = blank_image * 128
                        blank_image.astype(np.uint8)
                    else:
                        # white state by default
                        blank_image = blank_image * 255
                        blank_image = blank_image.astype(np.uint8)

                if percept == start_percept:
                    # start percept has border with special color
                    cv2.rectangle(blank_image, top_left, bot_right, (170, 255, 255), int(cell_width * 0.06))
                elif percept in terminal_states:
                    reward = terminal_states[percept]

                    if reward > 0:
                        rew_color = (86, 255, 170)
                    else:
                        rew_color = (86, 86, 255)

                    cv2.rectangle(blank_image, top_left, bot_right, rew_color, int(cell_width * 0.06))

                    blank_image = blank_image.astype(np.uint8)

                    if colored_field is True:
                        txt_color = (255, 255, 255)
                    else:
                        txt_color = rew_color

                    if abs(reward) > 9:
                        txt_size = 1
                    else:
                        txt_size = 2

                    cv2.putText(blank_image, str(int(reward)), reward_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, txt_size, txt_color, 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(blank_image, top_left, bot_right, (0, 0, 0), int(cell_width * 0.03))
                blank_image = blank_image.astype(np.uint8)

                if column_concatenation is None:
                    column_concatenation = blank_image
                else:
                    column_concatenation = np.concatenate((column_concatenation, blank_image), axis=1)

            if row_concatenation is None:
                row_concatenation = column_concatenation
            else:
                row_concatenation = np.concatenate((row_concatenation, column_concatenation), axis=0)

        cv2.imwrite(os.path.join(store_path, tag + '.png'), row_concatenation)
