from abc import ABC, abstractmethod
import numpy as np
from core import MDP, PerceptViz
from random import sample
from time import time


class LearningAgent(MDP, ABC):

    def __init__(self, world, debug, gamma, max_steps, num_episodes, debug_freq):
        """
        Abstract Learning Agent class. All learning agents use Q values for internal world representation.
        :param world: See core.AI.
        :param debug: See core.AI.
        :param gamma: Future reward discount. 1 for non-discounted problems, < 1 for discounted.
        :param max_steps: Maximum number of steps per episode.
        :param num_episodes: Number of episodes to train the agent for.
        :param debug_freq: Frequency how often agent should store past utility values.
        Frequency calculated as 1/debug_freq.
        """
        super().__init__(world=world, debug=debug)
        self._tag = 'Learning-Agent'

        self._arg_tag = 'Y_' + str(gamma) + '_N_' + str(num_episodes)
        if max_steps is not None:
            self._arg_tag += '_n_' + str(max_steps)

        self._discount = gamma
        self._num_actions = len(self._actions)

        self._num_episodes = num_episodes
        self._max_steps = max_steps
        self._episode_counter = 0
        self._step_counter = 0

        # Similar to planning agent's world replica.
        # Does not need to know all possible states beforehand.
        self._world_replica = {}
        np.random.seed(123)

        # Debug info
        self._debug_freq = debug_freq
        self.running_time = 0
        self.utility_history = {}
        self.naturally_finished_episodes = []
        self.returns = []

    def get_debug_info(self):
        return self._world_replica

    def get_tag(self):
        return self.get_model_name()

    def visualize(self, store_path):

        if self._solve_called is False:
            self._logger.warning('Nothing to visualize. Solve method never called.')
            return

        percept_viz = {}

        for percept, internal_state in self._world_replica.items():
            percept_viz[percept] = PerceptViz(
                percept=percept,
                action_values=tuple(internal_state)
            )

        self._world.visualize_solution(
            percept_viz=percept_viz,
            store_path=store_path,
            tag=self.get_model_name()
        )

    def _alt_exit_criteria(self):
        if self._max_steps is not None and self._step_counter >= self._max_steps:
            return True
        else:
            return False

    def _reset_alt_exit_criteria(self):
        self._step_counter = 0

    def _handle_state_discovery(self, percept):
        """
        Learning agents do not need to have knowledge of all available states in the world. This function handles
        the discovery of states never before seen by the agent. Updates self._world_replica.
        :param percept: Python tuple. Potentially new state.
        :return: Boolean value indication whether percept was indeed a never before seen state.
        """
        updated = False
        if percept not in self._world_replica:
            self._world_replica[percept] = np.zeros(self._num_actions)
            updated = True
        return updated

    def _update_debug_info(self, running_time, returns):
        """
        Updates debug info after each episode. Keeps track of utility values max_a(Q, a) for each state.
        :param running_time: Running time of the finished episode.
        :return: None.
        """
        if self._episode_counter % self._debug_freq == 0:
            for percept, q_values in self._world_replica.items():
                if percept in self.utility_history:
                    self.utility_history[percept]['values'].append(np.max(q_values).flatten()[0])
                else:
                    self.utility_history[percept] = {
                        'first_ep': self._episode_counter,
                        'values': [np.max(q_values).flatten()[0]]
                    }
            self.returns.append(returns)

        self.running_time += running_time

    def _log_info(self, time_ms, error):
        self._logger.info(' %s: EP: %d time [ms]: %f | error: %f',
                          self._tag, self._episode_counter, round(time_ms, 2), error)

    def get_model_name(self):
        return self._tag + '-' + self._arg_tag

    def export_utility_history(self, state_subset=None):

        if self._debug is False:
            self._logger.warning('Utility history requested but none stored as debug flag was set to False.')
            return

        utilities = {}

        for percept, value_dict in self.utility_history.items():
            if state_subset is None or percept in state_subset:
                utilities[percept] = [value_dict['values']]

                first_occ = value_dict['first_ep']
                num_skipped = first_occ // self._debug_freq
                start = num_skipped * self._debug_freq

                axis = range(start, self._num_episodes + self._debug_freq, self._debug_freq)
                utilities[percept].append(axis)

        return utilities

    def export_returns_history(self):
        if self._debug is False:
            self._logger.warning('Returns history requested but none stored as debug flag was set to False.')
            return

        axis = range(self._debug_freq, self._num_episodes + self._debug_freq, self._debug_freq)
        returns = [self.returns, axis]
        return returns

    @abstractmethod
    def _generate_episode(self):
        """
        Generates episode and, depending on the agent type (MonteCarlo vs TD), updates q_values.
        :return: None for MonteCarlo methods, or maximal difference between q_values in TD agents.
        """
        pass

    @abstractmethod
    def _q_update(self, percept, action_ind, returns):
        """
        Updates Q value associated with percept and action_ind.
        :param percept: Python tuple or a tuple of python tuples. State for which we want to update Q value,
        and, optionally the next state. Depends on type of LearningAgent MonteCarlo vs TD approach.
        :param action_ind: Integer, action index or a tuple of integer action indexes.
        Index of the action taken in percept/percept[0], and potentially next optimal action to be taken at percept[1].
        Depends on specific agent.
        :param returns: Float or integer, or tuple of floats or integers. Reward associated with entire episode, or a
        tuple of rewards obtained in percept list.
        :return: Error, difference between previous Q value and the updated added one.
        """
        pass

    @abstractmethod
    def _select_next_action(self, percept):
        self._step_counter += 1


class MonteCarlo(LearningAgent, ABC):

    def __init__(self, world, debug, gamma=0.9, max_steps=100, num_episodes=1000, debug_freq=1):
        """
        MonteCarlo approach superclass.
        :param world: See LearningAgent.
        :param debug: See LearningAgent.
        :param gamma: See LearningAgent.
        :param max_steps: See LearningAgent.
        :param num_episodes: See LearningAgent.
        :param debug_freq: See LearningAgent.
        """
        super().__init__(
            world=world,
            debug=debug,
            gamma=gamma,
            max_steps=max_steps,
            num_episodes=num_episodes,
            debug_freq=debug_freq
        )

        # Concrete meaning depends on subclass.
        # First-Visit number of times action is taken during training for On-Policy Monte Carlo agents,
        # or Importance-Sampling cummulative weight for Off-Policy Monte Carlo agents.
        self._step_normalizer = {}

        # Episode
        self._visited_sequence = []
        self._reward_sequence = []
        self._action_sequence = []

        # Policy
        self._policy = {}

    def _update_episode(self, cur_action, next_state, next_reward):
        """
        Updates current episode.
        :param cur_action: action_id selected in current state.
        :param next_state: Python tuple. Next state following the selected action.
        :param next_reward: Float or integer. Reward obtained in next_state.
        :return: None.
        """
        self._action_sequence.append(cur_action)
        self._reward_sequence.append(next_reward)
        self._visited_sequence.append(next_state)

    def _generate_episode(self):
        """
        Generates an entire episode. Agent plays the game until either terminal state is reached, or allowed number
        of steps is exceeded.
        :return: None.
        """
        is_terminal = self._initialize_episode()

        while is_terminal is False:
            cur_percept = self._visited_sequence[-1]
            action_id = self._select_next_action(percept=cur_percept)
            next_state = self._take_action(percept=cur_percept, action_id=action_id)
            is_terminal, next_reward = self._world.get_reward(percept=next_state)
            self._update_episode(cur_action=action_id, next_state=next_state, next_reward=next_reward)

            if is_terminal is False:
                is_terminal = self._alt_exit_criteria()

        if self._debug is True:
            if self._max_steps is not None and self._step_counter < self._max_steps:
                self.naturally_finished_episodes.append(self._episode_counter)

    def solve(self):
        super().solve()
        while self._episode_counter < self._num_episodes:
            start_time = time() * 1000
            self._generate_episode()
            returns = np.sum(self._reward_sequence)
            error = self._update_world_replica()
            end_time = time() * 1000
            delta = end_time - start_time
            self._log_info(time_ms=delta, error=error)

            if self._debug is True:
                self._update_debug_info(running_time=delta, returns=returns)

    @abstractmethod
    def _initialize_episode(self):
        """
        Handles new episode initialization, keeps track of self._episode_counter, self._step_counter,
        starting states, ...
        :return: None
        """
        pass

    @abstractmethod
    def _update_world_replica(self):
        """
        Updates internal world representation, stored in self._world_replica, based on the entire episode.
        :return: Maximal difference between pairs of Q values, before and after the update.
        """
        pass


class OnMonteCarlo(MonteCarlo):

    def __init__(self, world, debug, gamma=0.9, max_steps=100, num_episodes=1000,
                 explore_starts=True, epsilon=None, debug_freq=1):
        super().__init__(
            world=world,
            debug=debug,
            gamma=gamma,
            max_steps=max_steps,
            num_episodes=num_episodes,
            debug_freq=debug_freq
        )

        self._explore_starts = explore_starts
        self._epsilon = epsilon

        self._tag = 'MC-On-Policy'

        if self._epsilon is not None:
            self._tag += '-Soft'
        if self._explore_starts is True:
            self._tag += '-Explore-S'

        arg_tag = ''
        if epsilon is not None:
            arg_tag += 'e_' + str(epsilon)
        self._arg_tag = arg_tag + '_' + self._arg_tag

    def _select_next_action(self, percept):
        super()._select_next_action(percept=percept)
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

        self._step_normalizer[percept][action_ind] += 1
        prev_value = self._world_replica[percept][action_ind]
        num_visits = self._step_normalizer[percept][action_ind]
        update = (returns - prev_value)/num_visits
        self._world_replica[percept][action_ind] += update

        return np.abs(update)

    def _update_world_replica(self):

        first_visits = set([])
        episode_length = len(self._visited_sequence)
        reward_arr = np.array(self._reward_sequence).reshape(1, -1)
        max_error = 0

        for t in range(len(self._action_sequence)):
            action_id = self._action_sequence[t]

            percept = self._visited_sequence[t]
            _ = self._handle_state_discovery(percept=percept)

            action_ind = self._actions.index(action_id)
            visit_key = (percept, action_ind)

            if visit_key not in first_visits:
                discounts = np.power(self._discount, np.arange(episode_length - t)).reshape(episode_length - t, 1)
                cumulative_reward = (reward_arr[:, t:] @ discounts).flatten()[0]
                error = self._q_update(percept=percept, action_ind=action_ind, returns=cumulative_reward)
                if error > max_error:
                    max_error = error

            first_visits.add(visit_key)

        return max_error


class OffMonteCarlo(MonteCarlo):

    def __init__(self, world, debug, gamma=0.9, max_steps=100, num_episodes=1000, epsilon=0.5, debug_freq=1):
        super().__init__(
            world=world,
            debug=debug,
            gamma=gamma,
            max_steps=max_steps,
            num_episodes=num_episodes,
            debug_freq=debug_freq
        )
        self._tag = 'MC-Off-Policy'
        self._epsilon = epsilon

        arg_tag = ''
        if epsilon is not None:
            arg_tag += 'e_' + str(epsilon)
        self._arg_tag = arg_tag + '_' + self._arg_tag

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
        super()._select_next_action(percept=percept)
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
        max_error = 0

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

            error = self._q_update(
                percept=percept,
                action_ind=action_ind,
                returns=cumulative_reward,
                weight=sampling_weight
            )
            if error > max_error:
                max_error = error

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

        return max_error

    def _q_update(self, percept, action_ind, returns, weight):

        self._step_normalizer[percept][action_ind] += weight
        prev_value = self._world_replica[percept][action_ind]
        ratio = self._step_normalizer[percept][action_ind]
        update = (returns - prev_value) * (weight / ratio)
        self._world_replica[percept][action_ind] += update

        return np.abs(update)


class TD(LearningAgent, ABC):

    def __init__(self, world, debug, gamma=0.9, max_steps=100, num_episodes=1000,
                 epsilon=0.3, step=0.1, decay=False, debug_freq=1):
        super().__init__(
            world=world,
            debug=debug,
            gamma=gamma,
            max_steps=max_steps,
            num_episodes=num_episodes,
            debug_freq=debug_freq
        )

        self._epsilon = epsilon
        self._handle_state_discovery(percept=self._start_pos)
        self._episode_counter = 0

        self._decay = decay
        self._lr = step
        self._step = step

        self._terminal_states = set([])

        self._tag = 'TD'
        arg_tag = 'e_' + str(epsilon) + '_' + 'st_' + str(step)
        if decay is True:
            arg_tag += '_exp_decay'
        self._arg_tag = arg_tag + '_' + self._arg_tag

    def _lr_episode_decay(self):
        if self._decay is True:
            self._lr = self._step * np.log(self._episode_counter + 1) / (self._episode_counter + 1)

    def _select_next_action(self, percept):
        super()._select_next_action(percept=percept)
        rand = np.random.uniform(0, 1)
        if percept not in self._world_replica or rand < self._epsilon:
            rand_act_ind = np.random.randint(0, len(self._actions))
            return self._actions[rand_act_ind]
        else:
            return self._actions[np.argmax(self._world_replica[percept])]

    def solve(self):
        super().solve()
        while self._episode_counter < self._num_episodes:
            start_time = time() * 1000
            error, returns = self._generate_episode()
            end_time = time() * 1000
            delta = end_time - start_time
            self._log_info(time_ms=delta, error=error)

            if self._debug is True:
                self._update_debug_info(running_time=delta, returns=returns)


class QLearning(TD):

    def __init__(self, world, debug, gamma=0.9, max_steps=100, num_episodes=1000,
                 epsilon=0.3, step=0.1, decay=False, debug_freq=1):
        super().__init__(
            world=world,
            debug=debug,
            gamma=gamma,
            max_steps=max_steps,
            num_episodes=num_episodes,
            epsilon=epsilon,
            step=step,
            decay=decay,
            debug_freq=debug_freq
        )
        self._tag = 'Q-Learning'

    def _generate_episode(self):
        is_terminal = False
        cur_state = self._start_pos
        max_error = 0
        self._step_counter = 0
        episode_returns = []

        while is_terminal is False:
            _, cur_reward = self._world.get_reward(percept=cur_state)
            episode_returns.append(cur_reward)
            action_id = self._select_next_action(percept=cur_state)
            action_ind = self._actions.index(action_id)

            next_state = self._world.next_position(percept=cur_state, action_id=action_id)
            is_terminal, next_reward = self._world.get_reward(percept=next_state)

            if is_terminal is True:
                self._terminal_states.add(next_state)
                episode_returns.append(next_reward)
            else:
                _ = self._handle_state_discovery(percept=next_state)

            error = self._q_update(
                percept=(cur_state, next_state),
                action_ind=action_ind,
                returns=(cur_reward, next_reward)
            )
            if error > max_error:
                max_error = error

            cur_state = next_state
            self._step_counter += 1

            if is_terminal is False:
                is_terminal = self._alt_exit_criteria()

        if self._debug is True:
            if self._max_steps is not None and self._step_counter < self._max_steps:
                self.naturally_finished_episodes.append(self._episode_counter)

        self._episode_counter += 1
        self._lr_episode_decay()
        return max_error, np.sum(episode_returns)

    def _q_update(self, percept, action_ind, returns):
        prev_q = self._world_replica[percept[0]][action_ind]

        if percept[1] in self._terminal_states:
            next_step_reward = self._discount * returns[1]
        else:
            next_step_reward = self._discount * np.max(self._world_replica[percept[1]]).flatten()[0]

        self._world_replica[percept[0]][action_ind] += self._lr * (returns[0] + next_step_reward - prev_q)
        return np.abs(self._world_replica[percept[0]][action_ind] - prev_q)


class SARSA(TD):

    def __init__(self, world, debug, gamma=0.9, max_steps=100, num_episodes=1000, epsilon=0.3, step=0.1, decay=False,
                 expected=False, debug_freq=1):
        super().__init__(
            world=world,
            debug=debug,
            gamma=gamma,
            max_steps=max_steps,
            num_episodes=num_episodes,
            epsilon=epsilon,
            step=step,
            decay=decay,
            debug_freq=debug_freq
        )
        self._expected = expected
        self._tag = 'SARSA'

        if self._expected is True:
            self._tag = 'Expected-' + self._tag

        self._alt_prob = self._epsilon * (1 / self._num_actions)

    def _generate_episode(self):

        cur_state = self._start_pos
        cur_action = self._select_next_action(percept=cur_state)
        cur_action_ind = self._actions.index(cur_action)
        is_terminal, cur_reward = self._world.get_reward(percept=cur_state)
        self._step_counter = 0
        max_error = 0
        episode_returns = [cur_reward]

        while is_terminal is False:

            next_state = self._world.next_position(percept=cur_state, action_id=cur_action)
            is_terminal, next_reward = self._world.get_reward(percept=next_state)
            episode_returns.append(next_reward)

            if is_terminal is False:
                _ = self._handle_state_discovery(percept=next_state)
                next_action = self._select_next_action(percept=next_state)
                next_action_ind = self._actions.index(next_action)
            else:
                self._terminal_states.add(next_state)
                next_action = next_action_ind = None

            error = self._q_update(
                percept=(cur_state, next_state),
                action_ind=(cur_action_ind, next_action_ind),
                returns=(cur_reward, next_reward)
            )
            if error > max_error:
                max_error = error

            cur_state = next_state
            cur_action = next_action
            cur_action_ind = next_action_ind
            cur_reward = next_reward
            self._step_counter += 1

            if is_terminal is False:
                is_terminal = self._alt_exit_criteria()

        if self._debug is True:
            if self._max_steps is not None and self._step_counter < self._max_steps:
                self.naturally_finished_episodes.append(self._episode_counter)

        self._episode_counter += 1
        self._lr_episode_decay()
        return max_error, np.sum(episode_returns)

    def _generate_action_probs(self, action_ind):
        action_probs = np.ones((self._num_actions, 1))
        action_probs *= self._alt_prob
        action_probs[action_ind] += 1 - self._epsilon
        return action_probs

    def _q_update(self, percept, action_ind, returns):
        prev_q = self._world_replica[percept[0]][action_ind[0]]

        if percept[1] in self._terminal_states:
            next_step_reward = self._discount * returns[1]
        else:
            if self._expected is False:
                next_step_reward = self._discount * self._world_replica[percept[1]][action_ind[1]]
            else:
                action_probs = self._generate_action_probs(action_ind=action_ind[1])
                expected_reward = (self._world_replica[percept[1]].reshape(1, self._num_actions) @ action_probs).flatten()[0]
                next_step_reward = self._discount * expected_reward

        self._world_replica[percept[0]][action_ind[0]] += self._lr * (returns[0] + next_step_reward - prev_q)
        return np.abs(self._world_replica[percept[0]][action_ind[0]] - prev_q)

