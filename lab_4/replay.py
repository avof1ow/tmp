import numpy as np
import numpy.random as rd
import torch
from typing import Optional, Tuple, List, Union, Any
from abc import ABC, abstractmethod

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


class DataConverter(ABC):
    """Абстрактный класс для конвертации данных."""

    @staticmethod
    @abstractmethod
    def convert_state(trajectory_list: List[Tuple]) -> Any:
        """Конвертировать список состояний в нужный формат."""
        pass

    @staticmethod
    @abstractmethod
    def convert_other(trajectory_list: List[Tuple]) -> Any:
        """Конвертировать список других данных в нужный формат."""
        pass


class NumpyDataConverter(DataConverter):
    """Конвертер данных в NumPy массивы."""

    @staticmethod
    def convert_state(trajectory_list: List[Tuple]) -> np.ndarray:
        """Конвертировать список состояний в NumPy массив."""
        return np.array([item[0] for item in trajectory_list], dtype=np.float32)

    @staticmethod
    def convert_other(trajectory_list: List[Tuple]) -> np.ndarray:
        """Конвертировать список других данных в NumPy массив."""
        return np.array([item[1] for item in trajectory_list], dtype=np.float32)


class TorchDataConverter(DataConverter):
    """Конвертер данных в PyTorch тензоры."""

    @staticmethod
    def convert_state(trajectory_list: List[Tuple]) -> torch.Tensor:
        """Конвертировать список состояний в PyTorch тензор."""
        return torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float32)

    @staticmethod
    def convert_other(trajectory_list: List[Tuple]) -> torch.Tensor:
        """Конвертировать список других данных в PyTorch тензор."""
        return torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float32)


class BufferStorage:
    """Базовый класс для хранения данных буфера."""

    def __init__(self, max_len: int, state_dim: int, other_dim: int, use_torch: bool):
        self.maximum_length = max_len
        self.state_dimension = state_dim
        self.other_dimension = other_dim
        self.use_torch_storage = use_torch

        self._initialize_storage()

    def _initialize_storage(self):
        """Инициализация хранилища в зависимости от типа."""
        if self.use_torch_storage:
            self.buffer_state = torch.empty((self.maximum_length, self.state_dimension), dtype=torch.float32)
            self.buffer_other = torch.empty((self.maximum_length, self.other_dimension), dtype=torch.float32)
        else:
            self.buffer_state = np.empty((self.maximum_length, self.state_dimension), dtype=np.float32)
            self.buffer_other = np.empty((self.maximum_length, self.other_dimension), dtype=np.float32)

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

    def calculate_indices_for_batch(self, batch_size: int) -> Tuple[slice, slice, slice, slice]:
        """Вычислить срезы для записи пакета данных.

        Returns:
            Tuple[state_slice, other_slice, wrap_state_slice, wrap_other_slice]
        """
        start_idx = self.next_index
        potential_end_idx = start_idx + batch_size

        if potential_end_idx <= self.maximum_length:
            # Данные помещаются без переполнения
            end_idx = potential_end_idx
            wrap_length = 0
            self.next_index = end_idx
        else:
            # Данные перезаписывают начало буфера
            end_idx = self.maximum_length
            wrap_length = potential_end_idx - self.maximum_length
            self.next_index = wrap_length
            self.is_buffer_full = True

        return self._create_slices(start_idx, end_idx, wrap_length)

    def _create_slices(self, start_idx: int, end_idx: int, wrap_length: int) -> Tuple[slice, slice, slice, slice]:
        """Создать срезы для записи данных."""
        state_slice = slice(start_idx, end_idx)
        other_slice = slice(start_idx, end_idx)

        if wrap_length == 0:
            wrap_state_slice = slice(0, 0)
            wrap_other_slice = slice(0, 0)
        else:
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


class BatchSampler(ABC):
    """Абстрактный класс для выборки батчей."""

    @abstractmethod
    def sample(self, batch_size: int, storage: BufferStorage,
               current_length: int, device: torch.device) -> tuple:
        """Выбрать батч данных."""
        pass


class UniformBatchSampler(BatchSampler):
    """Равномерная выборка батчей."""

    def sample(self, batch_size: int, storage: BufferStorage,
               current_length: int, device: torch.device) -> tuple:
        """Равномерная выборка батча."""
        indices = rd.randint(current_length - 1, size=batch_size)
        return self._extract_batch_data(indices, storage)

    def _extract_batch_data(self, indices: np.ndarray, storage: BufferStorage) -> tuple:
        """Извлечь данные батча по индексам."""
        states, others = storage.get(indices)
        next_states, _ = storage.get(indices + 1)

        rewards = others[:, 0:1]
        masks = others[:, 1:2]
        actions = others[:, 2:]

        return rewards, masks, actions, states, next_states


class PriorityBatchSampler(BatchSampler):
    """Выборка батчей с учетом приоритетов (PER)."""

    def __init__(self, priority_tree, maximum_length: int):
        self.priority_tree = priority_tree
        self.maximum_length = maximum_length

    def sample(self, batch_size: int, storage: BufferStorage,
               current_length: int, device: torch.device) -> tuple:
        """Выборка батча с учетом приоритетов."""
        begin_index, end_index = self._calculate_sample_range(current_length)

        indices, importance_weights = self.priority_tree.get_indices_is_weights(
            batch_size, begin_index, end_index
        )

        states, others = storage.get(indices)
        next_states, _ = storage.get(indices + 1)

        rewards = others[:, 0:1].type(torch.float32)
        masks = others[:, 1:2].type(torch.float32)
        actions = others[:, 2:].type(torch.float32)

        importance_weights_tensor = torch.as_tensor(
            importance_weights,
            dtype=torch.float32,
            device=device
        )

        return rewards, masks, actions, states, next_states, importance_weights_tensor

    def _calculate_sample_range(self, current_length: int) -> Tuple[int, Optional[int]]:
        """Вычислить диапазон для выборки."""
        begin_index = -self.maximum_length
        end_index = (current_length - self.maximum_length) if (current_length < self.maximum_length) else None
        return begin_index, end_index


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

        # Инициализация компонентов
        other_dimension = self._calculate_other_dimension(if_on_policy, action_dim)
        use_torch_storage = not if_on_policy

        self.storage = BufferStorage(max_len, state_dim, other_dimension, use_torch_storage)
        self.data_converter = self._create_data_converter(if_on_policy)
        self.batch_sampler = self._create_batch_sampler(if_per_or_gae, if_on_policy, max_len)

    def _calculate_other_dimension(self, if_on_policy: bool, action_dim: int) -> int:
        """Вычислить размерность других данных."""
        if if_on_policy:
            return 1 + 1 + self.action_dimension + action_dim
        else:
            return 1 + 1 + self.action_dimension

    def _create_data_converter(self, if_on_policy: bool) -> DataConverter:
        """Создать конвертер данных в зависимости от политики."""
        if if_on_policy:
            return NumpyDataConverter()
        else:
            return TorchDataConverter()

    def _create_batch_sampler(self, if_per_or_gae: bool, if_on_policy: bool,
                              max_len: int) -> BatchSampler:
        """Создать сэмплер батчей."""
        should_use_per = bool(if_per_or_gae and if_on_policy)

        if should_use_per:
            self.priority_tree = BinarySearchTree(max_len)
            return PriorityBatchSampler(self.priority_tree, max_len)
        else:
            self.priority_tree = None
            return UniformBatchSampler()

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
            self.index_manager.calculate_indices_for_batch(batch_size)

        # Обновить индексы в дереве приоритетов
        if self.priority_tree:
            self._update_priority_indices(batch_size)

        # Записать данные в хранилище
        self._write_to_storage(state, other, state_slice, other_slice,
                               wrap_state_slice, wrap_other_slice)

    def _update_priority_indices(self, batch_size: int):
        """Обновить индексы в дереве приоритетов."""
        start_idx = self.index_manager.next_index
        data_indices = np.arange(start_idx, start_idx + batch_size) % self.maximum_length
        self.priority_tree.update_ids(data_indices)

    def _write_to_storage(self, state, other, state_slice: slice, other_slice: slice,
                          wrap_state_slice: slice, wrap_other_slice: slice):
        """Записать данные в хранилище."""
        if wrap_state_slice.start == wrap_state_slice.stop:  # Без переполнения
            self.storage.batch_store(state_slice.start, state, other)
        else:  # С переполнением
            self._write_with_wrap(state, other, state_slice, wrap_state_slice)

    def _write_with_wrap(self, state, other, state_slice: slice, wrap_state_slice: slice):
        """Записать данные с переполнением."""
        first_part_size = self.maximum_length - state_slice.start

        # Первая часть: в конец буфера
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
        state_array = self.data_converter.convert_state(trajectory_list)
        other_array = self.data_converter.convert_other(trajectory_list)

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
        self.index_manager.update_current_length()
        return self.batch_sampler.sample(
            batch_size,
            self.storage,
            self.index_manager.current_length,
            self.device
        )

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
        self.index_manager.update_current_length()
        states, others = self.storage.get_all(self.index_manager.current_length)

        # Конвертировать в тензоры если нужно
        if isinstance(states, np.ndarray):
            states = torch.as_tensor(states, device=self.device)
            others = torch.as_tensor(others, device=self.device)

        return self._extract_all_data(others, states)

    def _extract_all_data(self, others: torch.Tensor, states: torch.Tensor) -> tuple:
        """Извлечь все данные из буфера."""
        rewards = others[:, 0]
        masks = others[:, 1]
        actions = others[:, 2:2 + self.action_dimension]
        noises_or_probs = others[:, 2 + self.action_dimension:]

        return rewards, masks, actions, noises_or_probs, states

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
        if self._is_state_too_large():
            return

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

    def _is_state_too_large(self) -> bool:
        """Проверить, не слишком ли велика размерность состояния."""
        state_shape = self.buffer_state.shape
        if len(state_shape) > 2 or state_shape[1] > MAX_STATE_DIM_FOR_PRINT:
            print(f"| print_state_norm(): state_dim: {state_shape} is too large to print its norm.")
            return True
        return False

    def _sample_states(self) -> Optional[np.ndarray]:
        """Выборка состояний для статистики."""
        indices = np.arange(self.current_length)
        rd.shuffle(indices)
        indices = indices[:MAX_SAMPLE_SIZE_FOR_STATS]

        batch_state = self.buffer_state[indices]
        batch_state = self._convert_to_numpy(batch_state)

        if batch_state is None or batch_state.shape[1] > MAX_STATE_DIM_FOR_PRINT:
            return None

        return self._handle_nan_values(batch_state)

    def _convert_to_numpy(self, batch_state) -> Optional[np.ndarray]:
        """Конвертировать batch_state в numpy массив."""
        if isinstance(batch_state, torch.Tensor):
            return batch_state.cpu().data.numpy()
        elif isinstance(batch_state, np.ndarray):
            return batch_state
        else:
            return None

    def _handle_nan_values(self, batch_state: np.ndarray) -> np.ndarray:
        """Обработать NaN значения в данных."""
        if np.isnan(batch_state).any():
            return np.nan_to_num(batch_state)
        return batch_state

    def _compute_statistics(self, batch_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Вычисление статистик."""
        array_average = batch_state.mean(axis=0)
        array_std = batch_state.std(axis=0)
        fixed_std = ((np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6 + array_std) / 2

        return array_average, array_std, fixed_std

    def _apply_normalization(self, array_average: np.ndarray, array_std: np.ndarray,
                             fixed_std: np.ndarray, negative_average, division_std) -> Tuple[np.ndarray, np.ndarray]:
        """Применение нормализации к статистикам."""
        array_average = array_average - negative_average / division_std
        array_std = fixed_std / division_std
        return array_average, array_std

    def _print_results(self, array_average: np.ndarray, array_std: np.ndarray):
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
        self.worker_number = worker_num
        self.buffers = self._initialize_worker_buffers(
            max_len, worker_num, state_dim, action_dim, if_discrete, if_on_policy, if_per_or_gae
        )

    def _initialize_worker_buffers(self, max_len: int, worker_num: int, state_dim: int,
                                   action_dim: int, if_discrete: bool, if_on_policy: bool,
                                   if_per_or_gae: bool) -> List[ReplayBuffer]:
        """Инициализация буферов для каждого воркера."""
        buffer_max_len = max_len // worker_num
        return [
            ReplayBuffer(buffer_max_len, state_dim, action_dim, if_discrete, if_on_policy, if_per_or_gae)
            for _ in range(worker_num)
        ]

    @property
    def current_length(self) -> int:
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

        # Объединить результаты от всех воркеров
        return self._combine_worker_results(items_per_worker)

    def _combine_worker_results(self, items_per_worker: List[tuple]) -> list:
        """Объединить результаты от всех воркеров."""
        # Транспонировать 2D-список и объединить тензоры
        transposed_items = list(map(list, zip(*items_per_worker)))
        return [torch.cat(item, dim=0) for item in transposed_items]

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
        tree_id = self._calculate_tree_id(data_id)
        self._update_tree_node(tree_id, probability)

    def _calculate_tree_id(self, data_id: int) -> int:
        """Вычислить ID узла в дереве."""
        return data_id + self.memory_length - 1

    def _update_tree_node(self, tree_id: int, probability: float):
        """Обновить узел дерева и его родителей."""
        if self.current_length == tree_id:
            self.current_length += 1

        delta = probability - self.probability_array[tree_id]
        self.probability_array[tree_id] = probability
        self._propagate_changes(tree_id, delta)

    def _propagate_changes(self, tree_id: int, delta: float):
        """Распространить изменения вверх по дереву."""
        while tree_id != 0:
            tree_id = (tree_id - 1) // 2
            self.probability_array[tree_id] += delta

    def update_ids(self, data_ids, probability: float = MAX_PROBABILITY):
        """Update multiple data ids probability."""
        tree_ids = self._calculate_tree_ids(data_ids)
        self.current_length += (tree_ids >= self.current_length).sum()

        self.probability_array[tree_ids] = probability
        self._update_parent_nodes(tree_ids)

    def _calculate_tree_ids(self, data_ids) -> np.ndarray:
        """Вычислить IDs узлов в дереве для массива данных."""
        return data_ids + self.memory_length - 1

    def _update_parent_nodes(self, tree_ids: np.ndarray):
        """Обновить родительские узлы для массива данных."""
        steps_remaining = self.tree_depth - 1
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
            left_child_index, right_child_index = self._get_child_indices(parent_index)

            if left_child_index >= len(self.probability_array):
                leaf_index = parent_index
                break
            else:
                parent_index = self._select_child(value, left_child_index, right_child_index)
                if parent_index == right_child_index:
                    value -= self.probability_array[left_child_index]

        return min(leaf_index, self.current_length - 2)

    def _get_child_indices(self, parent_index: int) -> Tuple[int, int]:
        """Получить индексы дочерних узлов."""
        left_child_index = 2 * parent_index + 1
        right_child_index = left_child_index + 1
        return left_child_index, right_child_index

    def _select_child(self, value: float, left_child_index: int, right_child_index: int) -> int:
        """Выбрать дочерний узел на основе значения."""
        if value <= self.probability_array[left_child_index]:
            return left_child_index
        else:
            return right_child_index

    def get_indices_is_weights(self, batch_size: int, begin_index: int,
                               end_index: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Get indices and importance sampling weights."""
        self._increment_beta()

        random_values = self._generate_random_values(batch_size)
        leaf_ids = np.array([self.get_leaf_id(value) for value in random_values])
        self.indices = leaf_ids - (self.memory_length - 1)

        min_probability = self._calculate_min_probability(begin_index, end_index)
        probabilities = self.probability_array[leaf_ids] / min_probability
        importance_weights = np.power(probabilities, -self.per_beta)

        return self.indices, importance_weights

    def _increment_beta(self):
        """Увеличить beta параметр."""
        self.per_beta = min(1.0, self.per_beta + PER_BETA_INCREMENT)

    def _generate_random_values(self, batch_size: int) -> np.ndarray:
        """Сгенерировать случайные значения для выборки."""
        return (rd.rand(batch_size) + np.arange(batch_size)) * (self.probability_array[0] / batch_size)

    def _calculate_min_probability(self, begin_index: int, end_index: Optional[int]) -> float:
        """Вычислить минимальную вероятность в диапазоне."""
        if end_index:
            return self.probability_array[begin_index:end_index].min()
        else:
            return self.probability_array[begin_index:].min()

    def td_error_update(self, td_error):
        """Update probabilities based on TD error."""
        probability = self._calculate_probability_from_td_error(td_error)
        self.update_ids(self.indices, probability)

    def _calculate_probability_from_td_error(self, td_error) -> np.ndarray:
        """Вычислить вероятность на основе TD ошибки."""
        probability = td_error.squeeze().clamp(CLAMP_MIN, CLAMP_MAX).pow(self.per_alpha)
        return probability.cpu().numpy()

    def reset(self):
        """Reset the tree to initial state."""
        self.probability_array.fill(0)
        self.current_length = self.memory_length - 1
        self.indices = None