import numpy as np
import numpy.random as rd
import torch

"""ElegantRL (https://github.com/AI4Finance-LLC/ElegantRL)"""

# Константы для настроек PER (Prioritized Experience Replay)
MAX_PROBABILITY = 10.0
PER_ALPHA_DEFAULT = 0.6  # alpha = (Uniform:0, Greedy:1)
PER_BETA_DEFAULT = 0.4  # beta = (PER:0, NotPER:1)
PER_BETA_INCREMENT = 0.001

# Константы для печати статистик
MAX_SAMPLE_SIZE_FOR_STATS = 2 ** 14
MAX_STATE_DIM_FOR_PRINT = 64

# Константы для вычислений
NUMERICAL_STABILITY_EPS = 1e-6
CLAMP_MIN = 1e-6
CLAMP_MAX = 10.0


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_discrete, if_on_policy, if_per_or_gae):
        """Experience Replay Buffer

        Save environment transition in a continuous RAM for high performance training.
        We save trajectory in order and save state and other (action, reward, mask, ...) separately.

        Args:
            max_len: The maximum capacity of ReplayBuffer. First In First Out
            state_dim: The dimension of state
            action_dim: The dimension of action (action_dim==1 for discrete action)
            if_discrete: Whether the action space is discrete
            if_on_policy: Whether using on-policy or off-policy
            if_per_or_gae: Whether using PER (Prioritized Experience Replay) or GAE
        """
        self.current_length = 0
        self.next_index = 0
        self.is_buffer_full = False
        self.maximum_length = max_len
        self.data_type = torch.float32
        self.is_on_policy = if_on_policy
        self.action_dimension = 1 if if_discrete else action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        should_use_per = bool(if_per_or_gae and if_on_policy)
        self.priority_tree = BinarySearchTree(max_len) if should_use_per else None

        if if_on_policy:
            other_dimension = 1 + 1 + self.action_dimension + action_dim
            # other = (reward, mask, action, a_noise) for continuous action
            # other = (reward, mask, a_int, a_prob) for discrete action
            self.buffer_other = np.empty((max_len, other_dimension), dtype=np.float32)
            self.buffer_state = np.empty((max_len, state_dim), dtype=np.float32)
        else:
            other_dimension = 1 + 1 + self.action_dimension
            self.buffer_other = torch.empty((max_len, other_dimension), dtype=torch.float32, device=self.device)
            self.buffer_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)

    def append_buffer(self, state, other):
        """Append single state-other pair to buffer (CPU array to CPU array)."""
        self.buffer_state[self.next_index] = state
        self.buffer_other[self.next_index] = other

        if self.priority_tree:
            self.priority_tree.update_id(self.next_index)

        self.next_index += 1
        if self.next_index >= self.maximum_length:
            self.is_buffer_full = True
            self.next_index = 0

    def extend_buffer(self, state, other):
        """Extend buffer with multiple state-other pairs (CPU array to CPU array)."""
        batch_size = len(other)
        next_index = self.next_index + batch_size

        if self.priority_tree:
            data_indices = np.arange(self.next_index, next_index) % self.maximum_length
            self.priority_tree.update_ids(data_indices)

        if next_index > self.maximum_length:
            # Часть данных помещается в конец буфера
            end_part_length = self.maximum_length - self.next_index
            self.buffer_state[self.next_index:self.maximum_length] = state[:end_part_length]
            self.buffer_other[self.next_index:self.maximum_length] = other[:end_part_length]
            self.is_buffer_full = True

            # Остаток данных помещается в начало буфера
            remaining_length = next_index - self.maximum_length
            self.buffer_state[0:remaining_length] = state[-remaining_length:]
            self.buffer_other[0:remaining_length] = other[-remaining_length:]
            next_index = remaining_length
        else:
            # Все данные помещаются без переполнения
            self.buffer_state[self.next_index:next_index] = state
            self.buffer_other[self.next_index:next_index] = other

        self.next_index = next_index

    def extend_buffer_from_list(self, trajectory_list):
        """Extend buffer from list of trajectories."""
        if self.is_on_policy:
            state_array = np.array([item[0] for item in trajectory_list], dtype=np.float32)
            other_array = np.array([item[1] for item in trajectory_list], dtype=np.float32)
        else:
            state_array = torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float32)
            other_array = torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float32)

        self.extend_buffer(state_array, other_array)

    def sample_batch(self, batch_size) -> tuple:
        """Randomly sample a batch of data for training.

        Args:
            batch_size: The number of data in a batch for Stochastic Gradient Descent

        Returns:
            tuple containing:
                - reward: reward.shape==(now_len, 1)
                - mask: mask.shape==(now_len, 1), mask = 0.0 if done else gamma
                - action: action.shape==(now_len, action_dim)
                - state: state.shape==(now_len, state_dim)
                - next_state: state.shape==(now_len, state_dim), next state
                - (optional) is_weights: important sampling weights (for PER)
        """
        if self.priority_tree:
            begin_index = -self.maximum_length
            end_index = (self.current_length - self.maximum_length) if (
                        self.current_length < self.maximum_length) else None

            indices, importance_weights = self.priority_tree.get_indices_is_weights(batch_size, begin_index, end_index)
            reward_mask_action = self.buffer_other[indices]

            return (reward_mask_action[:, 0:1].type(torch.float32),  # reward
                    reward_mask_action[:, 1:2].type(torch.float32),  # mask
                    reward_mask_action[:, 2:].type(torch.float32),  # action
                    self.buffer_state[indices].type(torch.float32),  # state
                    self.buffer_state[indices + 1].type(torch.float32),  # next state
                    torch.as_tensor(importance_weights, dtype=torch.float32, device=self.device))  # importance weights
        else:
            indices = rd.randint(self.current_length - 1, size=batch_size)
            reward_mask_action = self.buffer_other[indices]

            return (reward_mask_action[:, 0:1],
                    reward_mask_action[:, 1:2],
                    reward_mask_action[:, 2:],
                    self.buffer_state[indices],
                    self.buffer_state[indices + 1])

    def sample_all(self) -> tuple:
        """Sample all the data in ReplayBuffer (for on-policy).

        Returns:
            tuple containing:
                - reward: reward.shape==(now_len, 1)
                - mask: mask.shape==(now_len, 1), mask = 0.0 if done else gamma
                - action: action.shape==(now_len, action_dim)
                - noise: noise.shape==(now_len, action_dim)
                - state: state.shape==(now_len, state_dim)
        """
        all_state = torch.as_tensor(self.buffer_state[:self.current_length], device=self.device)
        all_other = torch.as_tensor(self.buffer_other[:self.current_length], device=self.device)

        return (all_other[:, 0],  # reward
                all_other[:, 1],  # mask
                all_other[:, 2:2 + self.action_dimension],  # action
                all_other[:, 2 + self.action_dimension:],  # action_noise or action_prob
                all_state)  # state without last_state

    def update_current_length(self):
        """Update the pointer `current_length`, which is the current data number of ReplayBuffer."""
        self.current_length = self.maximum_length if self.is_buffer_full else self.next_index

    def empty_buffer(self):
        """Empty the buffer by setting current_length=0. On-policy needs to empty buffer before exploration."""
        self.current_length = 0
        self.next_index = 0
        self.is_buffer_full = False

    def print_state_norm(self, negative_average=None, division_std=None):
        """Print the state norm information: state_avg, state_std.

        We don't suggest to use running stat state. We directly do normalization
        on state using the historical avg and std.

        Args:
            negative_average: negative_average.shape=(state_dim)
            division_std: division_std.shape=(state_dim)
        """
        # Check if state dimension is too large to print
        state_shape = self.buffer_state.shape
        if len(state_shape) > 2 or state_shape[1] > MAX_STATE_DIM_FOR_PRINT:
            print(f"| print_state_norm(): state_dim: {state_shape} is too large to print its norm.")
            return None

        # Sample state
        indices = np.arange(self.current_length)
        rd.shuffle(indices)
        indices = indices[:MAX_SAMPLE_SIZE_FOR_STATS]

        batch_state = self.buffer_state[indices]

        # Compute state norm
        if isinstance(batch_state, torch.Tensor):
            batch_state = batch_state.cpu().data.numpy()

        assert isinstance(batch_state, np.ndarray)

        if batch_state.shape[1] > MAX_STATE_DIM_FOR_PRINT:
            print(f"| _print_norm(): state_dim: {batch_state.shape[1]:.0f} is too large to print its norm.")
            return None

        if np.isnan(batch_state).any():
            batch_state = np.nan_to_num(batch_state)

        array_average = batch_state.mean(axis=0)
        array_std = batch_state.std(axis=0)
        fixed_std = ((np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6 + array_std) / 2

        if negative_average is not None:
            array_average = array_average - negative_average / division_std
            array_std = fixed_std / division_std

        print("print_state_norm: state_avg, state_std (fixed)")
        print(f"avg = np.{repr(array_average).replace('=float32', '=np.float32')}")
        print(f"std = np.{repr(array_std).replace('=float32', '=np.float32')}")

    def td_error_update(self, td_error):
        """Update PER tree with TD error."""
        if self.priority_tree:
            self.priority_tree.td_error_update(td_error)


class ReplayBufferMP:
    def __init__(self, max_len, worker_num, state_dim, action_dim, if_discrete, if_on_policy, if_per_or_gae):
        """Experience Replay Buffer for Multiple Processing.

        Args:
            max_len: The max_len of ReplayBuffer, not the total len of ReplayBufferMP
            worker_num: The rollout workers number
        """
        self.current_length = 0
        self.maximum_length = max_len
        self.worker_number = worker_num

        buffer_max_len = max_len // worker_num
        self.buffers = [
            ReplayBuffer(buffer_max_len, state_dim, action_dim, if_discrete, if_on_policy, if_per_or_gae)
            for _ in range(worker_num)
        ]

    def sample_batch(self, batch_size) -> list:
        """Sample batch from multiple buffers."""
        batch_per_worker = batch_size // self.worker_number
        items_per_worker = [self.buffers[i].sample_batch(batch_per_worker) for i in range(self.worker_number)]

        # Transpose 2D-list
        transposed_items = list(map(list, zip(*items_per_worker)))
        return [torch.cat(item, dim=0) for item in transposed_items]

    def update_current_length(self):
        """Update total length from all buffers."""
        self.current_length = 0
        for buffer in self.buffers:
            buffer.update_current_length()
            self.current_length += buffer.current_length

    def empty_buffer(self):
        """Empty all buffers."""
        for buffer in self.buffers:
            buffer.empty_buffer()

    def print_state_norm(self, negative_average=None, division_std=None):
        """Print state norm from first buffer."""
        self.buffers[0].print_state_norm(negative_average, division_std)

    def td_error_update(self, td_error):
        """Update PER trees in all buffers with TD error."""
        td_errors = td_error.view(self.worker_number, -1, 1)
        for i in range(self.worker_number):
            self.buffers[i].priority_tree.td_error_update(td_errors[i])


class BinarySearchTree:
    """Binary Search Tree for PER (Prioritized Experience Replay)."""

    def __init__(self, memory_length):
        self.memory_length = memory_length  # replay buffer length
        self.probability_array = np.zeros((memory_length - 1) + memory_length)  # parent_nodes + leaf_nodes
        self.maximum_length = len(self.probability_array)
        self.current_length = self.memory_length - 1  # pointer
        self.indices = None
        self.tree_depth = int(np.log2(self.maximum_length))

        # PER hyperparameters
        self.per_alpha = PER_ALPHA_DEFAULT  # alpha = (Uniform:0, Greedy:1)
        self.per_beta = PER_BETA_DEFAULT  # beta = (PER:0, NotPER:1)

    def update_id(self, data_id, probability=MAX_PROBABILITY):
        """Update single data id probability."""
        tree_id = data_id + self.memory_length - 1
        if self.current_length == tree_id:
            self.current_length += 1

        delta = probability - self.probability_array[tree_id]
        self.probability_array[tree_id] = probability

        while tree_id != 0:
            tree_id = (tree_id - 1) // 2
            self.probability_array[tree_id] += delta

    def update_ids(self, data_ids, probability=MAX_PROBABILITY):
        """Update multiple data ids probability."""
        tree_ids = data_ids + self.memory_length - 1
        self.current_length += (tree_ids >= self.current_length).sum()

        steps_remaining = self.tree_depth - 1
        self.probability_array[tree_ids] = probability
        parent_ids = (tree_ids - 1) // 2

        while steps_remaining:
            left_child_ids = parent_ids * 2 + 1
            self.probability_array[parent_ids] = self.probability_array[left_child_ids] + self.probability_array[
                left_child_ids + 1]
            parent_ids = (parent_ids - 1) // 2
            steps_remaining -= 1

        self.probability_array[0] = self.probability_array[1] + self.probability_array[2]

    def get_leaf_id(self, value):
        """Get leaf id for given value."""
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.probability_array):
                leaf_index = parent_index
                break
            else:
                if value <= self.probability_array[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.probability_array[left_child_index]
                    parent_index = right_child_index

        return min(leaf_index, self.current_length - 2)

    def get_indices_is_weights(self, batch_size, begin_index, end_index):
        """Get indices and importance sampling weights."""
        self.per_beta = min(1.0, self.per_beta + PER_BETA_INCREMENT)

        random_values = (rd.rand(batch_size) + np.arange(batch_size)) * (self.probability_array[0] / batch_size)
        leaf_ids = np.array([self.get_leaf_id(value) for value in random_values])
        self.indices = leaf_ids - (self.memory_length - 1)

        min_probability = self.probability_array[begin_index:end_index].min() if end_index else self.probability_array[
            begin_index:].min()
        probabilities = self.probability_array[leaf_ids] / min_probability
        importance_weights = np.power(probabilities, -self.per_beta)

        return self.indices, importance_weights

    def td_error_update(self, td_error):
        """Update probabilities based on TD error."""
        probability = td_error.squeeze().clamp(CLAMP_MIN, CLAMP_MAX).pow(self.per_alpha)
        probability = probability.cpu().numpy()
        self.update_ids(self.indices, probability)