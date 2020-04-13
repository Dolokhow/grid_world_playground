from abc import ABC, abstractmethod
import numpy as np
from core import AI, PerceptViz
from random import sample
from time import time


class LearningAgent(AI, ABC):

    def __init__(self, world, debug, gamma):
        super().__init__(world=world, debug=debug)
        self._tag = 'Learning-Agent'
        self._discount = gamma
        self._num_actions = len(self._actions)

        # Same as planning agent world replica, stores q values
        self._world_replica = {}

    def get_debug_info(self):
        return self._world_replica

    def visualize(self, store_path):
        percept_viz = {}

        for percept, internal_state in self._world_replica.items():
            percept_viz[percept] = PerceptViz(
                percept=percept,
                action_values=tuple(internal_state)
            )

        self._world.visualize_solution(
            percept_viz=percept_viz,
            store_path=store_path,
            tag=self._tag
        )

    @abstractmethod
    def _generate_episode(self):
        pass

    def _handle_state_discovery(self, percept):
        updated = False
        if percept not in self._world_replica:
            self._world_replica[percept] = np.zeros(self._num_actions)
            updated = True
        return updated


class MonteCarlo(LearningAgent, ABC):
    def __init__(self, world, debug, gamma, num_steps, num_episodes):
        super().__init__(world=world, debug=debug, gamma=gamma)
        self._num_steps = num_steps
        self._num_episodes = num_episodes

        self._step_normalizer = {}

        # Episode
        self._visited_sequence = []
        self._reward_sequence = []
        self._action_sequence = []

        self._step_counter = 0
        self._episode_counter = 0

        # Policy
        self._policy = {}
        np.random.seed(123)

    @abstractmethod
    def _initialize_episode(self):
        pass

    @abstractmethod
    # Different signature!
    def _select_next_action(self, percept):
        pass

    @abstractmethod
    def _update_world_replica(self):
        pass

    @abstractmethod
    def _q_update(self, percept, action_ind, returns):
        pass

    def _update_episode(self, cur_action, next_state, next_reward):
        self._action_sequence.append(cur_action)
        self._reward_sequence.append(next_reward)
        self._visited_sequence.append(next_state)
        self._step_counter += 1

    def _generate_episode(self):
        is_terminal = self._initialize_episode()

        while self._step_counter < self._num_steps and is_terminal is False:
            cur_percept = self._visited_sequence[-1]
            action_id = self._select_next_action(percept=cur_percept)
            next_state = self._take_action(percept=cur_percept, action_id=action_id)
            is_terminal, next_reward = self._world.get_reward(percept=next_state)
            self._update_episode(cur_action=action_id, next_state=next_state, next_reward=next_reward)

    def solve(self):
        while self._episode_counter < self._num_episodes:
            start_time = time() * 1000
            self._generate_episode()
            self._update_world_replica()
            end_time = time() * 1000
            self._logger.info('EP: %d time [ms]: %f',
                              self._episode_counter, end_time - start_time)


class OnMonteCarlo(MonteCarlo):
    def __init__(self, world, debug, gamma, num_steps, num_episodes, explore_starts=True, epsilon=None):
        super().__init__(world=world, debug=debug, gamma=gamma, num_steps=num_steps, num_episodes=num_episodes)

        self._explore_starts = explore_starts
        self._epsilon = epsilon

        self._tag = 'MCOnPolicy'
        if self._epsilon is not None:
            self._tag += '-Soft'
        if self._explore_starts is True:
            self._tag += '-Explore-S'

    def _select_next_action(self, percept):

        if percept not in self._world_replica:
            rand_act_ind = np.random.randint(0, len(self._actions))
            action_id = self._actions[rand_act_ind]
        else:
            if self._epsilon is not None:
                rand = np.random.uniform(0, 1)
                if rand < self._epsilon:
                    rand_act_ind = np.random.randint(0, len(self._actions))
                    action_id = self._actions[rand_act_ind]
                else:
                    action_id = self._actions[np.argmax(self._world_replica[percept])]
            else:
                action_id = self._actions[np.argmax(self._world_replica[percept])]

        self._policy[percept] = action_id
        return action_id

    def _initialize_episode(self):

        del self._visited_sequence
        del self._reward_sequence
        del self._action_sequence
        self._step_counter = 0

        if self._explore_starts is False or self._episode_counter == 0:
            start_state = self._start_pos
            is_terminal, start_reward = self._world.get_reward(percept=start_state)

            self._visited_sequence = [start_state]
            self._reward_sequence = [start_reward]
            self._action_sequence = []
            self._step_counter += 1

        else:
            start_state = sample(self._world_replica.keys(), 1)[0]
            _, start_reward = self._world.get_reward(percept=start_state)

            rand = np.random.randint(0, len(self._actions))
            start_action = self._actions[rand]

            first_state = self._take_action(percept=start_state, action_id=start_action)
            is_terminal, first_reward = self._world.get_reward(percept=first_state)

            self._visited_sequence = [start_state, first_state]
            self._reward_sequence = [start_reward, first_reward]
            self._action_sequence = [start_action]
            self._step_counter += 2

        self._episode_counter += 1

        return is_terminal

    def _handle_state_discovery(self, percept):
        updated = super()._handle_state_discovery(percept=percept)
        if percept not in self._step_normalizer:
            self._step_normalizer[percept] = np.zeros(self._num_actions)
        return updated

    def _q_update(self, percept, action_ind, returns):
        # prev_value = self._world_replica[percept][action_ind]
        # num_visits = self._num_visits[percept][action_ind]
        # self._world_replica[percept][action_ind] = (prev_value * num_visits + returns) / (num_visits + 1)
        # self._num_visits[percept][action_ind] += 1

        self._step_normalizer[percept][action_ind] += 1
        prev_value = self._world_replica[percept][action_ind]
        num_visits = self._step_normalizer[percept][action_ind]
        self._world_replica[percept][action_ind] = prev_value + (returns - prev_value)/num_visits

    def _update_world_replica(self):

        first_visits = set([])
        episode_length = len(self._visited_sequence)
        reward_arr = np.array(self._reward_sequence).reshape(1, -1)

        for t in range(len(self._action_sequence)):
            action_id = self._action_sequence[t]

            percept = self._visited_sequence[t]
            _ = self._handle_state_discovery(percept=percept)

            action_ind = self._actions.index(action_id)
            visit_key = (percept, action_ind)

            # if already_explored is True:
            #     preferred_action_ind = np.argmax(self._world_replica[percept])
            # else:
            #     preferred_action_ind = None

            if visit_key not in first_visits:
                discounts = np.power(self._discount, np.arange(episode_length - t)).reshape(episode_length - t, 1)
                cumulative_reward = (reward_arr[:, t:] @ discounts).flatten()[0]
                self._q_update(percept=percept, action_ind=action_ind, returns=cumulative_reward)

                # new_preferred_action_ind = np.argmax(self._world_replica[percept])
                # if preferred_action_ind is not None and new_preferred_action_ind != preferred_action_ind:
                #     break

            first_visits.add(visit_key)


class OffMonteCarlo(MonteCarlo):
    def __init__(self, world, debug, gamma, num_steps, num_episodes, epsilon=0.5):
        super().__init__(world=world, debug=debug, gamma=gamma, num_steps=num_steps, num_episodes=num_episodes)
        self._tag = 'MCOffPolicy'
        self._epsilon = epsilon

    def _initialize_episode(self):
        del self._visited_sequence
        del self._reward_sequence
        del self._action_sequence
        self._step_counter = 0

        start_state = self._start_pos
        is_terminal, start_reward = self._world.get_reward(percept=start_state)

        self._visited_sequence = [start_state]
        self._reward_sequence = [start_reward]
        self._action_sequence = []
        self._step_counter += 1

        self._episode_counter += 1
        return is_terminal

    def _select_next_action(self, percept):
        rand = np.random.uniform(0, 1)
        if percept not in self._world_replica or rand < self._epsilon:
            rand_act_ind = np.random.randint(0, len(self._actions))
            action_id = self._actions[rand_act_ind]
        else:
            action_id = self._actions[np.argmax(self._world_replica[percept])]
        self._policy[percept] = action_id
        return action_id

    def _handle_state_discovery(self, percept):
        updated = super()._handle_state_discovery(percept=percept)
        if percept not in self._step_normalizer:
            self._step_normalizer[percept] = np.zeros(self._num_actions)
        return updated

    def _update_world_replica(self):

        cumulative_reward = 0
        sampling_weight = 1

        first_discoveries = set([])

        for t in reversed(range(len(self._action_sequence))):
            action_id = self._action_sequence[t]
            action_ind = self._actions.index(action_id)

            percept = self._visited_sequence[t]
            reward = self._reward_sequence[t+1]
            cumulative_reward = self._discount * cumulative_reward + reward

            updated = self._handle_state_discovery(percept=percept)
            if updated is True:
                first_discoveries.add(percept)

            # if state already had associated q-value, we get the action index of the highest q-value
            if percept not in first_discoveries:
                preferred_action_ind = np.argmax(self._world_replica[percept])
            else:
                preferred_action_ind = None

            self._q_update(
                percept=percept,
                action_ind=action_ind,
                returns=cumulative_reward,
                weight=sampling_weight
            )
            new_preferred_action_ind = np.argmax(self._world_replica[percept])

            if preferred_action_ind is not None and new_preferred_action_ind != preferred_action_ind:
                break

            # update importance_sampling
            if percept in first_discoveries:
                action_prob = 1 / self._num_actions
            else:
                if action_ind == preferred_action_ind:
                    action_prob = (1 - self._epsilon) + self._epsilon * (1 / self._num_actions)
                else:
                    action_prob = self._epsilon * (1 / (self._num_actions - 1))
            sampling_weight /= action_prob

    def _q_update(self, percept, action_ind, returns, weight):
        self._step_normalizer[percept][action_ind] += weight
        prev_value = self._world_replica[percept][action_ind]
        ratio = self._step_normalizer[percept][action_ind]
        self._world_replica[percept][action_ind] += (returns - prev_value) * (weight / ratio)









