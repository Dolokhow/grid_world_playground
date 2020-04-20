from abc import ABC, abstractmethod
import numpy as np
from core import MDP, PerceptViz


class Iteration(MDP, ABC):

    def __init__(self, world, debug='False', tag='Iteration', error=0.001, gamma=0.9):
        super().__init__(world=world, debug=debug, tag=tag)

        self._max_error = error
        self._gamma = gamma
        all_percepts = self._world.get_available_percepts()
        self._world_replica = self._create_world_replica(all_percepts=all_percepts)

    @abstractmethod
    def _create_world_replica(self, all_percepts):
        """
        Creates self._world_replica which stores either utility values (U, v) or Q-values (q)
        which agent uses to make optimal decisions.
        :return: world_replica
        """
        pass

    @abstractmethod
    def _get_reward_from_replica(self, percept):
        """
        Gets estimated reward (utility value, or Q-value) from internal state represented by self._world_replica.
        :param percept Index in self._world_replica
        :return currently estimated reward
        """
        pass

    @abstractmethod
    def _update_world_replica(self, percept, estimated_rewards):
        pass

    def _estimate_reward(self, percept, outcomes):

        percepts = [outcome[0] for outcome in outcomes]
        probs = np.array([outcome[1] for outcome in outcomes])
        rewards = np.ones((len(probs), 1))
        _, current_state_reward = self._world.get_reward(percept=percept)

        for percept_ind in range(len(percepts)):
            outcome_percept = percepts[percept_ind]

            is_terminal, world_reward = self._world.get_reward(percept=outcome_percept)
            if is_terminal is True:
                immediate_reward = world_reward
            else:
                # either state value, or maximum q-value, to be subclassed
                immediate_reward = self._get_reward_from_replica(percept=outcome_percept)

            rewards[percept_ind, 0] *= immediate_reward

        reward_estimate = (probs @ rewards).flatten()[0]
        reward_estimate = current_state_reward + self._gamma * reward_estimate

        return reward_estimate

    def get_debug_info(self):
        return self._world_replica

    # Iteration methods do not have alternative exit criteria
    def _alt_exit_criteria(self):
        return False

    # Iteration methods do not have alternative exit criteria
    def _reset_alt_exit_criteria(self):
        pass

    def _iteration(self):
        iter_error = self._max_error
        while iter_error >= self._max_error:
            errors = []
            for percept, value in self._world_replica.items():
                # we have a list of estimated_rewards for each action one reward
                estimated_rewards = np.ones((len(self._actions)))

                for action_ind in range(len(self._actions)):
                    action_id = self._actions[action_ind]
                    # all possible outcomes based on action_id. List of new percepts with assigned probabilities.
                    possible_actions = self._world.get_action_outcomes(percept=percept, action_id=action_id)
                    estimated_reward = self._estimate_reward(
                        percept=percept,
                        outcomes=possible_actions
                    )
                    estimated_rewards[action_ind] *= estimated_reward

                error = self._update_world_replica(
                    percept=percept,
                    estimated_rewards=estimated_rewards
                )
                errors.append(error)
            iter_error = np.max(errors)

    def solve(self):
        super().solve()
        self._iteration()


class QIteration(Iteration):

    def __init__(self, world, debug='False', error=0.001, gamma=0.9):
        super().__init__(world=world, debug=debug, tag='Q-Iteration', error=error, gamma=gamma)

    def visualize(self, store_path):
        percept_viz = {}

        if self._solve_called is False:
            self._logger.warning('Nothing to visualize. Iteration method never called. Will visualize empty World.')
            self._world.visualize_solution(
                percept_viz=percept_viz,
                store_path=store_path,
                tag=self._tag
            )
            return

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

    def _select_next_action(self, percept):
        return self._actions[np.argmax(self._world_replica[percept])]

    def _create_world_replica(self, all_percepts):
        world_replica = {}
        for percept in all_percepts:
            q_values = np.zeros(len(self._actions))
            world_replica[percept] = q_values
        return world_replica

    def _get_reward_from_replica(self, percept):
        return np.max(self._world_replica[percept])

    def _update_world_replica(self, percept, estimated_rewards):
        prev_states = self._world_replica[percept]
        self._world_replica[percept] = estimated_rewards
        err = np.max(np.abs(prev_states - estimated_rewards))
        return err


class ValueIteration(Iteration):

    def __init__(self, world, debug='False', error=0.001, gamma=0.9):
        super().__init__(world=world, debug=debug, tag='Value-Iteration', error=error, gamma=gamma)

    def visualize(self, store_path):
        percept_viz = {}

        if self._solve_called is False:
            self._logger.warning('Nothing to visualize. Solve method never called. Will visualize empty World.')
            self._world.visualize_solution(
                percept_viz=percept_viz,
                store_path=store_path,
                tag=self._tag
            )
            return

        for percept, internal_state in self._world_replica.items():
            percept_viz[percept] = PerceptViz(
                percept=percept,
                single_value=internal_state
            )

        self._world.visualize_solution(
            percept_viz=percept_viz,
            store_path=store_path,
            tag=self._tag
        )

    def _select_next_action(self, percept):
        optimal_action = None
        highest_utility = self._world_replica[percept]

        for action_id in self._actions:
            next_pos = self._world.next_position(percept=percept, action_id=action_id)
            utility = self._world_replica[next_pos]

            if utility > highest_utility:
                highest_utility = utility
                optimal_action = action_id

        if optimal_action is not None:
            return optimal_action
        else:
            return self._actions[np.random.randint(0, len(self._actions))]

    def _create_world_replica(self, all_percepts):
        world_replica = {}
        for percept in all_percepts:
            world_replica[percept] = 0
        return world_replica

    def _get_reward_from_replica(self, percept):
        return self._world_replica[percept]

    def _update_world_replica(self, percept, estimated_rewards):
        prev_state = self._world_replica[percept]
        new_state = np.max(estimated_rewards).flatten()[0]
        self._world_replica[percept] = new_state
        err = np.abs(prev_state - new_state)
        return err


class PolicyIteration(Iteration):

    def __init__(self, world, debug='False', error=0.001, gamma=0.9, policy_shift_max=10):
        super().__init__(world=world, debug=debug, tag='Policy-Iteration', error=error, gamma=gamma)
        self._policy = self._initialize_policy()
        self._policy_shift_counter = {}
        self._policy_shift_limit = policy_shift_max
        np.random.seed(123)

    def _select_next_action(self, percept):
        optimal_action = None
        highest_utility = self._world_replica[percept]

        for action_id in self._actions:
            next_pos = self._world.next_position(percept=percept, action_id=action_id)
            utility = self._world_replica[next_pos]

            if utility > highest_utility:
                highest_utility = utility
                optimal_action = action_id

        if optimal_action is not None:
            return optimal_action
        else:
            return self._actions[np.random.randint(0, len(self._actions))]

    def _evaluation_step(self):
        iter_error = self._max_error
        while iter_error >= self._max_error:
            errors = []
            for percept, value in self._world_replica.items():
                action_id = self._policy[percept]
                possible_actions = self._world.get_action_outcomes(percept=percept, action_id=action_id)
                estimated_reward = self._estimate_reward(
                    percept=percept,
                    outcomes=possible_actions
                )
                error = self._update_world_replica(
                    percept=percept,
                    estimated_rewards=estimated_reward
                )
                errors.append(error)
            iter_error = np.max(errors)

    def _improvement_step(self):
        policy_stability = True
        for percept, value in self._world_replica.items():

            old_action = self._policy[percept]
            old_utility_value = self._world_replica[percept]

            new_action = old_action
            new_utility_value = old_utility_value

            for action_id in self._actions:
                possible_actions = self._world.get_action_outcomes(percept=percept, action_id=action_id)
                estimated_reward = self._estimate_reward(
                    percept=percept,
                    outcomes=possible_actions
                )
                if estimated_reward > new_utility_value:
                    new_utility_value = estimated_reward
                    new_action = action_id

            if new_action != old_action:
                policy_stable = self._update_policy(
                    percept=percept,
                    new_action_id=new_action,
                    old_action_id=old_action
                )
                if policy_stable is False:
                    policy_stability = False
                # self._policy[percept] = new_action
                # policy_stability = False

        return policy_stability

    def _create_world_replica(self, all_percepts):
        world_replica = {}
        for percept in all_percepts:
            world_replica[percept] = 0
        return world_replica

    def _initialize_policy(self):
        policy = {}
        num_actions = len(self._actions)
        for percept, _ in self._world_replica.items():
            random_action_index = np.random.randint(low=0, high=num_actions)
            policy[percept] = self._actions[random_action_index]
        return policy

    def _get_reward_from_replica(self, percept):
        return self._world_replica[percept]

    def _update_world_replica(self, percept, estimated_rewards):
        prev_state = self._world_replica[percept]
        self._world_replica[percept] = estimated_rewards
        err = np.abs(prev_state - estimated_rewards)
        return err

    def _update_policy(self, percept, new_action_id, old_action_id):

        policy_shift_key = (percept, old_action_id, new_action_id)
        policy_shift_alt_key = (percept, new_action_id, old_action_id)

        if policy_shift_key not in self._policy_shift_counter:
            self._policy_shift_counter[policy_shift_key] = 1
        else:
            self._policy_shift_counter[policy_shift_key] += 1

        if policy_shift_alt_key in self._policy_shift_counter:
            opposite_shifts = self._policy_shift_counter[policy_shift_alt_key]
            if opposite_shifts + self._policy_shift_counter[policy_shift_key] > self._policy_shift_limit:
                return True

        self._policy[percept] = new_action_id
        return False

    def visualize(self, store_path):
        percept_viz = {}

        if self._solve_called is False:
            self._logger.warning('Nothing to visualize. Solve method never called. Will visualize empty World.')
            self._world.visualize_solution(
                percept_viz=percept_viz,
                store_path=store_path,
                tag=self._tag
            )
            return

        for percept, internal_state in self._world_replica.items():
            percept_viz[percept] = PerceptViz(
                percept=percept,
                single_value=internal_state
            )

        self._world.visualize_solution(
            percept_viz=percept_viz,
            store_path=store_path,
            tag=self._tag
        )

    # redefined w.r.t Iteration
    def _iteration(self):
        policy_stable = False
        while policy_stable is False:
            self._evaluation_step()
            policy_stable = self._improvement_step()
