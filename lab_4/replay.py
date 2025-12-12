import numpy as np
import numpy.random as rd
import torch
from typing import Optional, Tuple, List, Union

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


class BufferStorage:
    """Базовый класс для хранения данных буфера."""

    def __init__(self, max_len: int, state_dim: int, other_dim: int, use_torch: bool):
        self.maximum_length = max_len
        self.state_dimension = state_dim
        self.other_dimension = other_dim
        self.use_torch_storage = use_torch

        if use_torch:
            self._initialize_torch_storage(max_len, state_dim, other_dim)
        else:
            self._initialize_numpy_storage(max_len, state_dim, other_dim)

    def _initialize_torch_storage(self, max_len: int, state_dim: int, other_dim: int):
        """Инициализация хранения с использованием PyTorch тензоров."""
        self.buffer_state = torch.empty((max_len, state_dim), dtype=torch.float32)
        self.buffer_other = torch.empty((max_len, other_dim), dtype=torch.float32)

    def _initialize_numpy_storage(self, max_len: int, state_dim: int, other_dim: int):
        """Инициализация хранения с использованием NumPy массивов."""
        self.buffer_state = np.empty((max_len, state_dim), dtype=np.float32)
        self.buffer_other = np.empty((max_len, other_dim), dtype=np.float32)

    def store(self, index: int, state, other):
        """Сохранить состояние и другие данные по указанному индексу."""
        self.buffer_state[index] = state
        self.buffer_other[index] = other

    def batch_store(self, start_idx: int, states, others):
        """Сохранить пакет состояний и других данных."""
        end_idx = start_idx + len(others)
        self.buffer_state[start_idx:end_idx] = states
        self.buffer_other[start_idx:end_idx] = others

    def get(self, indices):
        """Получить данные по указанным индексам."""
        return self.buffer_state[indices], self.buffer_other[indices]

    def get_all(self, length: int):
        """Получить все данные до указанной длины."""
        return self.buffer_state[:length], self.buffer_other[:length]


class BufferIndexManager:
    """Управление индексами циклического буфера."""

    def __init__(self, max_len: int):
        self.maximum_length = max_len
        self.current_length = 0
        self.next_index = 0
        self.is_buffer_full = False

    def update_after_append(self):
        """Обновить состояние после добавления одного элемента."""
        self.next_index += 1
        if self.next_index >= self.maximum_length:
            self.is_buffer_full = True
            self.next_index = 0

    def update_after_extend(self, batch_size: int) -> Tuple[int, int, int]:
        """Обновить состояние после добавления пакета элементов.

        Returns:
            Tuple[start_idx, end_idx, wrap_length] где:
                - start_idx: начальный индекс для записи
                - end_idx: конечный индекс для записи
                - wrap_length: длина данных, которые перезаписывают начало буфера
        """
        start_idx = self.next_index
        potential_end_idx = start_idx + batch_size

        if potential_end_idx <= self.maximum_length:
            # Данные помещаются без переполнения
            self.next_index = potential_end_idx
            wrap_length = 0
            end_idx = potential_end_idx
        else:
            # Данные перезаписывают начало буфера
            wrap_length = potential_end_idx - self.maximum_length
            end_idx = self.maximum_length
            self.next_index = wrap_length
            self.is_buffer_full = True

        return start_idx, end_idx, wrap_length

    def calculate_indices_for_wrap(self, batch_size: int) -> Tuple[slice, slice, slice, slice]:
        """Вычислить срезы для записи данных с переполнением."""
        start_idx, end_idx, wrap_length = self.update_after_extend(batch_size)

        if wrap_length == 0:
            # Без переполнения
            state_slice = slice(start_idx, end_idx)
            other_slice = slice(start_idx, end_idx)
            wrap_state_slice = slice(0, 0)  # Пустые срезы
            wrap_other_slice = slice(0, 0)
        else:
            # С переполнением
            state_slice = slice(start_idx, end_idx)
            other_slice = slice(start_idx, end_idx)
            wrap_state_slice = slice(0, wrap_length)
            wrap_other_slice = slice(0, wrap_length)

        return state_slice, other_slice, wrap_state_slice, wrap_other_slice

    def update_current_length(self):
        """Обновить текущую длину буфера."""
        self.current_length = self.maximum_length if self.is_buffer_full else self.next_index

    def reset(self):
        """Сбросить состояние менеджера индексов."""
        self.current_length = 0
        self.next_index = 0
        self.is_buffer_full = False


class ReplayBuffer:
    def __init__(self, max_len: int, state_dim: int, action_dim: int,
                 if_discrete: bool, if_on_policy: bool, if_per_or_gae: bool):
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
        self.index_manager = BufferIndexManager(max_len)
        self.maximum_length = max_len
        self.is_on_policy = if_on_policy
        self.action_dimension = 1 if if_discrete else action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Определение размерности других данных
        if if_on_policy:
            other_dimension = 1 + 1 + self.action_dimension + action_dim
            use_torch_storage = False  # On-policy использует numpy
        else:
            other_dimension = 1 + 1 + self.action_dimension
            use_torch_storage = True  # Off-policy использует torch

        # Инициализация хранилища
        self.storage = BufferStorage(max_len, state_dim, other_dimension, use_torch_storage)

        # Инициализация дерева приоритетов если нужно
        should_use_per = bool(if_per_or_gae and if_on_policy)
        self.priority_tree = BinarySearchTree(max_len) if should_use_per else None

    def append_buffer(self, state, other):
        """Append single state-other pair to buffer (CPU array to CPU array)."""
        self.storage.store(self.index_manager.next_index, state, other)

        if self.priority_tree:
            self.priority_tree.update_id(self.index_manager.next_index)

        self.index_manager.update_after_append()

    def extend_buffer(self, state, other):
        """Extend buffer with multiple state-other pairs (CPU array to CPU array)."""
        batch_size = len(other)

        # Получить срезы для записи данных
        state_slice, other_slice, wrap_state_slice, wrap_other_slice = \
            self.index_manager.calculate_indices_for_wrap(batch_size)

        # Обновить индексы в дереве приоритетов
        if self.priority_tree:
            start_idx = self.index_manager.next_index
            data_indices = np.arange(start_idx, start_idx + batch_size) % self.maximum_length
            self.priority_tree.update_ids(data_indices)

        # Записать данные в хранилище
        if wrap_state_slice.start == wrap_state_slice.stop:  # Без переполнения
            self.storage.batch_store(state_slice.start, state, other)
        else:  # С переполнением
            # Первая часть: в конец буфера
            first_part_size = self.maximum_length - state_slice.start
            self.storage.batch_store(
                state_slice.start,
                state[:first_part_size],
                other[:first_part_size]
            )

            # Вторая часть: в начало буфера
            self.storage.batch_store(
                0,
                state[first_part_size:],
                other[first_part_size:]
            )

    def extend_buffer_from_list(self, trajectory_list: List[Tuple]):
        """Extend buffer from list of trajectories."""
        if self.is_on_policy:
            state_array = np.array([item[0] for item in trajectory_list], dtype=np.float32)
            other_array = np.array([item[1] for item in trajectory_list], dtype=np.float32)
        else:
            state_array = torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float32)
            other_array = torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float32)

        self.extend_buffer(state_array, other_array)

    def sample_batch(self, batch_size: int) -> tuple:
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
            return self._sample_batch_with_priorities(batch_size)
        else:
            return self._sample_batch_uniform(batch_size)

    def _sample_batch_with_priorities(self, batch_size: int) -> tuple:
        """Выборка батча с учетом приоритетов (PER)."""
        begin_index = -self.maximum_length
        end_index = (self.index_manager.current_length - self.maximum_length) \
            if (self.index_manager.current_length < self.maximum_length) else None

        indices, importance_weights = self.priority_tree.get_indices_is_weights(
            batch_size, begin_index, end_index
        )

        states, others = self.storage.get(indices)
        next_states, _ = self.storage.get(indices + 1)

        rewards = others[:, 0:1].type(torch.float32)
        masks = others[:, 1:2].type(torch.float32)
        actions = others[:, 2:].type(torch.float32)

        importance_weights_tensor = torch.as_tensor(
            importance_weights,
            dtype=torch.float32,
            device=self.device
        )

        return (rewards, masks, actions, states, next_states, importance_weights_tensor)

    def _sample_batch_uniform(self, batch_size: int) -> tuple:
        """Равномерная выборка батча."""
        indices = rd.randint(self.index_manager.current_length - 1, size=batch_size)
        states, others = self.storage.get(indices)
        next_states, _ = self.storage.get(indices + 1)

        rewards = others[:, 0:1]
        masks = others[:, 1:2]
        actions = others[:, 2:]

        return (rewards, masks, actions, states, next_states)

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
        states, others = self.storage.get_all(self.index_manager.current_length)

        if isinstance(states, np.ndarray):
            states = torch.as_tensor(states, device=self.device)
            others = torch.as_tensor(others, device=self.device)

        rewards = others[:, 0]
        masks = others[:, 1]
        actions = others[:, 2:2 + self.action_dimension]
        noises_or_probs = others[:, 2 + self.action_dimension:]

        return (rewards, masks, actions, noises_or_probs, states)

    def update_current_length(self):
        """Update the pointer `current_length`, which is the current data number of ReplayBuffer."""
        self.index_manager.update_current_length()

    def empty_buffer(self):
        """Empty the buffer by setting current_length=0. On-policy needs to empty buffer before exploration."""
        self.index_manager.reset()
        if self.priority_tree:
            self.priority_tree.reset()

    def print_state_norm(self, negative_average=None, division_std=None):
        """Print the state norm information: state_avg, state_std.

        We don't suggest to use running stat state. We directly do normalization
        on state using the historical avg and std.

        Args:
            negative_average: negative_average.shape=(state_dim)
            division_std: division_std.shape=(state_dim)
        """
        state_printer = StateStatisticsPrinter(self.storage.buffer_state, self.index_manager.current_length)
        state_printer.print_statistics(negative_average, division_std)

    def td_error_update(self, td_error):
        """Update PER tree with TD error."""
        if self.priority_tree:
            self.priority_tree.td_error_update(td_error)


class StateStatisticsPrinter:
    """Класс для вычисления и печати статистик состояний."""

    def __init__(self, buffer_state, current_length):
        self.buffer_state = buffer_state
        self.current_length = current_length

    def print_statistics(self, negative_average=None, division_std=None):
        """Print the state norm information."""
        # Check if state dimension is too large to print
        state_shape = self.buffer_state.shape
        if len(state_shape) > 2 or state_shape[1] > MAX_STATE_DIM_FOR_PRINT:
            print(f"| print_state_norm(): state_dim: {state_shape} is too large to print its norm.")
            return None

        # Sample state
        batch_state = self._sample_states()
        if batch_state is None:
            return

        # Compute state norm
        array_average, array_std, fixed_std = self._compute_statistics(batch_state)

        if negative_average is not None:
            array_average, array_std = self._apply_normalization(
                array_average, array_std, fixed_std, negative_average, division_std
            )

        self._print_results(array_average, array_std)

    def _sample_states(self):
        """Выборка состояний для статистики."""
        indices = np.arange(self.current_length)
        rd.shuffle(indices)
        indices = indices[:MAX_SAMPLE_SIZE_FOR_STATS]

        batch_state = self.buffer_state[indices]

        if isinstance(batch_state, torch.Tensor):
            batch_state = batch_state.cpu().data.numpy()

        assert isinstance(batch_state, np.ndarray)

        if batch_state.shape[1] > MAX_STATE_DIM_FOR_PRINT:
            print(f"| _print_norm(): state_dim: {batch_state.shape[1]:.0f} is too large to print its norm.")
            return None

        if np.isnan(batch_state).any():
            batch_state = np.nan_to_num(batch_state)

        return batch_state

    def _compute_statistics(self, batch_state):
        """Вычисление статистик."""
        array_average = batch_state.mean(axis=0)
        array_std = batch_state.std(axis=0)
        fixed_std = ((np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6 + array_std) / 2

        return array_average, array_std, fixed_std

    def _apply_normalization(self, array_average, array_std, fixed_std, negative_average, division_std):
        """Применение нормализации к статистикам."""
        array_average = array_average - negative_average / division_std
        array_std = fixed_std / division_std
        return array_average, array_std

    def _print_results(self, array_average, array_std):
        """Печать результатов."""
        print("print_state_norm: state_avg, state_std (fixed)")
        print(f"avg = np.{repr(array_average).replace('=float32', '=np.float32')}")
        print(f"std = np.{repr(array_std).replace('=float32', '=np.float32')}")


class ReplayBufferMP:
    def __init__(self, max_len: int, worker_num: int, state_dim: int, action_dim: int,
                 if_discrete: bool, if_on_policy: bool, if_per_or_gae: bool):
        """Experience Replay Buffer for Multiple Processing.

        Args:
            max_len: The max_len of ReplayBuffer, not the total len of ReplayBufferMP
            worker_num: The rollout workers number
        """
        self.buffers = self._initialize_worker_buffers(
            max_len, worker_num, state_dim, action_dim, if_discrete, if_on_policy, if_per_or_gae
        )
        self.worker_number = worker_num

    def _initialize_worker_buffers(self, max_len, worker_num, state_dim, action_dim,
                                   if_discrete, if_on_policy, if_per_or_gae):
        """Инициализация буферов для каждого воркера."""
        buffer_max_len = max_len // worker_num
        return [
            ReplayBuffer(buffer_max_len, state_dim, action_dim, if_discrete, if_on_policy, if_per_or_gae)
            for _ in range(worker_num)
        ]

    @property
    def current_length(self):
        """Текущая общая длина всех буферов."""
        total_length = 0
        for buffer in self.buffers:
            buffer.update_current_length()
            total_length += buffer.index_manager.current_length
        return total_length

    def sample_batch(self, batch_size: int) -> list:
        """Sample batch from multiple buffers."""
        batch_per_worker = batch_size // self.worker_number
        items_per_worker = [self.buffers[i].sample_batch(batch_per_worker) for i in range(self.worker_number)]

        # Transpose 2D-list
        transposed_items = list(map(list, zip(*items_per_worker)))
        return [torch.cat(item, dim=0) for item in transposed_items]

    def update_current_length(self):
        """Update total length from all buffers."""
        # Длина вычисляется через property, метод оставлен для обратной совместимости
        pass

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

    def __init__(self, memory_length: int):
        self.memory_length = memory_length
        self.probability_array = np.zeros((memory_length - 1) + memory_length)
        self.maximum_length = len(self.probability_array)
        self.current_length = self.memory_length - 1
        self.indices = None
        self.tree_depth = int(np.log2(self.maximum_length))

        self.per_alpha = PER_ALPHA_DEFAULT
        self.per_beta = PER_BETA_DEFAULT

    def update_id(self, data_id: int, probability: float = MAX_PROBABILITY):
        """Update single data id probability."""
        tree_id = data_id + self.memory_length - 1
        if self.current_length == tree_id:
            self.current_length += 1

        delta = probability - self.probability_array[tree_id]
        self.probability_array[tree_id] = probability

        while tree_id != 0:
            tree_id = (tree_id - 1) // 2
            self.probability_array[tree_id] += delta

    def update_ids(self, data_ids, probability: float = MAX_PROBABILITY):
        """Update multiple data ids probability."""
        tree_ids = data_ids + self.memory_length - 1
        self.current_length += (tree_ids >= self.current_length).sum()

        steps_remaining = self.tree_depth - 1
        self.probability_array[tree_ids] = probability
        parent_ids = (tree_ids - 1) // 2

        while steps_remaining:
            left_child_ids = parent_ids * 2 + 1
            self.probability_array[parent_ids] = \
                self.probability_array[left_child_ids] + self.probability_array[left_child_ids + 1]
            parent_ids = (parent_ids - 1) // 2
            steps_remaining -= 1

        self.probability_array[0] = self.probability_array[1] + self.probability_array[2]

    def get_leaf_id(self, value: float) -> int:
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

    def get_indices_is_weights(self, batch_size: int, begin_index: int, end_index: Optional[int]) -> Tuple[
        np.ndarray, np.ndarray]:
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

    def reset(self):
        """Reset the tree to initial state."""
        self.probability_array.fill(0)
        self.current_length = self.memory_length - 1
        self.indices = None