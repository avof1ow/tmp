import numpy as np
import numpy.random as rd
import torch
from typing import Optional, Tuple, List, Union, Any
from abc import ABC, abstractmethod
import warnings
from dataclasses import dataclass
from functools import lru_cache

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

# Константы для валидации
MIN_BUFFER_SIZE = 1
MIN_BATCH_SIZE = 1
MIN_WORKER_NUM = 1

# Оптимизационные константы
VECTORIZED_SAMPLE_THRESHOLD = 1000  # Порог для векторизованных операций
MEMORY_ALIGNMENT = 64  # Выравнивание памяти для производительности


@dataclass(frozen=True)
class BufferConfig:
    """Конфигурация буфера для инициализации."""
    max_len: int
    state_dim: int
    action_dim: int
    is_discrete: bool
    is_on_policy: bool
    use_per_or_gae: bool


class BufferError(Exception):
    """Базовое исключение для ошибок буфера."""
    pass


class BufferInitializationError(BufferError):
    """Ошибка инициализации буфера."""
    pass


class BufferOperationError(BufferError):
    """Ошибка операции с буфером."""
    pass


class BufferValidationError(BufferError):
    """Ошибка валидации данных буфера."""
    pass


class OptimizedStorage:
    """Оптимизированное хранилище с учетом выравнивания памяти."""

    def __init__(self, max_len: int, state_dim: int, other_dim: int, use_torch: bool):
        self.maximum_length = max_len
        self.state_dimension = state_dim
        self.other_dimension = other_dim
        self.use_torch_storage = use_torch

        # Оптимизация: предвычисленные размеры
        self.state_shape = (max_len, state_dim)
        self.other_shape = (max_len, other_dim)

        self._initialize_optimized_storage()

    def _initialize_optimized_storage(self):
        """Инициализация оптимизированного хранилища."""
        if self.use_torch_storage:
            # PyTorch автоматически выравнивает память
            self.buffer_state = torch.empty(self.state_shape, dtype=torch.float32, pin_memory=True)
            self.buffer_other = torch.empty(self.other_shape, dtype=torch.float32, pin_memory=True)
        else:
            # NumPy с выравниванием памяти
            self.buffer_state = np.empty(self.state_shape, dtype=np.float32, order='C')
            self.buffer_other = np.empty(self.other_shape, dtype=np.float32, order='C')

            # Выравнивание памяти для производительности
            if hasattr(self.buffer_state, 'flags') and hasattr(self.buffer_state.flags, 'C_CONTIGUOUS'):
                if not self.buffer_state.flags.C_CONTIGUOUS:
                    self.buffer_state = np.ascontiguousarray(self.buffer_state)
                if not self.buffer_other.flags.C_CONTIGUOUS:
                    self.buffer_other = np.ascontiguousarray(self.buffer_other)

    def store(self, index: int, state, other):
        """Оптимизированное сохранение состояния и других данных."""
        self.buffer_state[index] = state
        self.buffer_other[index] = other

    def batch_store_vectorized(self, start_idx: int, states, others):
        """Векторизованное пакетное сохранение."""
        end_idx = start_idx + len(others)
        self.buffer_state[start_idx:end_idx] = states
        self.buffer_other[start_idx:end_idx] = others

    def get_vectorized(self, indices):
        """Векторизованное получение данных."""
        return self.buffer_state[indices], self.buffer_other[indices]

    def get_slice(self, slice_obj: slice):
        """Получение среза данных."""
        return self.buffer_state[slice_obj], self.buffer_other[slice_obj]

    @property
    def is_torch(self) -> bool:
        """Проверка, используется ли PyTorch хранилище."""
        return self.use_torch_storage


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

    @staticmethod
    def validate_trajectory_list(trajectory_list: List[Tuple]) -> None:
        """Валидировать список траекторий."""
        if not trajectory_list:
            raise BufferValidationError("Trajectory list cannot be empty")


class NumpyDataConverter(DataConverter):
    """Оптимизированный конвертер данных в NumPy массивы."""

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_array_shape(trajectory_length: int) -> Tuple[int, ...]:
        """Кэширование формы массива для производительности."""
        return (trajectory_length,)

    @staticmethod
    def convert_state(trajectory_list: List[Tuple]) -> np.ndarray:
        """Конвертировать список состояний в NumPy массив."""
        # Оптимизация: предварительное выделение памяти
        if not trajectory_list:
            return np.empty((0,), dtype=np.float32)

        # Использование list comprehension быстрее чем map для небольших списков
        state_array = np.array([item[0] for item in trajectory_list], dtype=np.float32, order='C')
        return np.ascontiguousarray(state_array)

    @staticmethod
    def convert_other(trajectory_list: List[Tuple]) -> np.ndarray:
        """Конвертировать список других данных в NumPy массив."""
        if not trajectory_list:
            return np.empty((0,), dtype=np.float32)

        other_array = np.array([item[1] for item in trajectory_list], dtype=np.float32, order='C')
        return np.ascontiguousarray(other_array)


class TorchDataConverter(DataConverter):
    """Оптимизированный конвертер данных в PyTorch тензоры."""

    @staticmethod
    def convert_state(trajectory_list: List[Tuple]) -> torch.Tensor:
        """Конвертировать список состояний в PyTorch тензор."""
        if not trajectory_list:
            return torch.empty((0,), dtype=torch.float32)

        # Оптимизация: использование torch.stack вместо цикла
        states = [torch.as_tensor(item[0], dtype=torch.float32) for item in trajectory_list]
        return torch.stack(states, dim=0)

    @staticmethod
    def convert_other(trajectory_list: List[Tuple]) -> torch.Tensor:
        """Конвертировать список других данных в PyTorch тензор."""
        if not trajectory_list:
            return torch.empty((0,), dtype=torch.float32)

        others = [torch.as_tensor(item[1], dtype=torch.float32) for item in trajectory_list]
        return torch.stack(others, dim=0)


class BufferIndexManager:
    """Оптимизированное управление индексами циклического буфера."""

    __slots__ = ('maximum_length', 'current_length', 'next_index', 'is_buffer_full')

    def __init__(self, max_len: int):
        self.maximum_length = max_len
        self.current_length = 0
        self.next_index = 0
        self.is_buffer_full = False

        # Предвычисленные значения для оптимизации
        self._max_len_minus_one = max_len - 1

    def update_after_append(self):
        """Оптимизированное обновление после добавления одного элемента."""
        self.next_index += 1
        if self.next_index >= self.maximum_length:
            self.is_buffer_full = True
            self.next_index = 0

    def calculate_indices_for_batch(self, batch_size: int) -> Tuple[slice, slice, slice, slice]:
        """Оптимизированное вычисление срезов для записи пакета данных."""
        start_idx = self.next_index
        potential_end_idx = start_idx + batch_size

        if potential_end_idx <= self.maximum_length:
            # Без переполнения
            end_idx = potential_end_idx
            wrap_length = 0
            self.next_index = end_idx
        else:
            # С переполнением
            end_idx = self.maximum_length
            wrap_length = potential_end_idx - self.maximum_length
            self.next_index = wrap_length
            self.is_buffer_full = True

        return self._create_slices(start_idx, end_idx, wrap_length)

    def _create_slices(self, start_idx: int, end_idx: int, wrap_length: int) -> Tuple[slice, slice, slice, slice]:
        """Создание срезов с оптимизацией."""
        state_slice = slice(start_idx, end_idx)
        other_slice = slice(start_idx, end_idx)

        if wrap_length == 0:
            return state_slice, other_slice, slice(0, 0), slice(0, 0)
        else:
            return state_slice, other_slice, slice(0, wrap_length), slice(0, wrap_length)

    def update_current_length(self):
        """Оптимизированное обновление текущей длины буфера."""
        self.current_length = self.maximum_length if self.is_buffer_full else self.next_index

    def reset(self):
        """Сброс состояния менеджера индексов."""
        self.current_length = 0
        self.next_index = 0
        self.is_buffer_full = False


class BatchSampler(ABC):
    """Абстрактный класс для выборки батчей."""

    @abstractmethod
    def sample(self, batch_size: int, storage: OptimizedStorage,
               current_length: int, device: torch.device) -> tuple:
        """Выбрать батч данных."""
        pass


class UniformBatchSampler(BatchSampler):
    """Оптимизированная равномерная выборка батчей."""

    def sample(self, batch_size: int, storage: OptimizedStorage,
               current_length: int, device: torch.device) -> tuple:
        """Оптимизированная равномерная выборка батча."""
        # Генерация индексов одним вызовом
        indices = rd.randint(current_length - 1, size=batch_size)

        # Векторизованное извлечение данных
        states, others = storage.get_vectorized(indices)
        next_states, _ = storage.get_vectorized(indices + 1)

        # Оптимизация: прямое извлечение срезов
        rewards = others[:, 0:1]
        masks = others[:, 1:2]
        actions = others[:, 2:]

        return rewards, masks, actions, states, next_states


class PriorityBatchSampler(BatchSampler):
    """Оптимизированная выборка батчей с учетом приоритетов."""

    def __init__(self, priority_tree, maximum_length: int):
        self.priority_tree = priority_tree
        self.maximum_length = maximum_length

        # Предвычисленные значения
        self._negative_max_len = -maximum_length

    def sample(self, batch_size: int, storage: OptimizedStorage,
               current_length: int, device: torch.device) -> tuple:
        """Оптимизированная выборка батча с учетом приоритетов."""
        begin_index = self._negative_max_len
        end_index = (current_length - self.maximum_length) if (current_length < self.maximum_length) else None

        indices, importance_weights = self.priority_tree.get_indices_is_weights(
            batch_size, begin_index, end_index
        )

        # Векторизованное извлечение
        states, others = storage.get_vectorized(indices)
        next_states, _ = storage.get_vectorized(indices + 1)

        # Оптимизация: избегание лишних преобразований типов
        if storage.is_torch:
            rewards = others[:, 0:1]
            masks = others[:, 1:2]
            actions = others[:, 2:]
        else:
            # Конвертация только если нужно
            rewards = torch.as_tensor(others[:, 0:1], dtype=torch.float32, device=device)
            masks = torch.as_tensor(others[:, 1:2], dtype=torch.float32, device=device)
            actions = torch.as_tensor(others[:, 2:], dtype=torch.float32, device=device)
            states = torch.as_tensor(states, dtype=torch.float32, device=device)
            next_states = torch.as_tensor(next_states, dtype=torch.float32, device=device)

        importance_weights_tensor = torch.as_tensor(
            importance_weights,
            dtype=torch.float32,
            device=device
        )

        return rewards, masks, actions, states, next_states, importance_weights_tensor


class OptimizedBinarySearchTree:
    """Оптимизированное бинарное дерево поиска для PER."""

    __slots__ = ('memory_length', 'probability_array', 'maximum_length',
                 'current_length', 'indices', 'tree_depth',
                 'per_alpha', 'per_beta', '_precomputed_indices')

    def __init__(self, memory_length: int):
        # Оптимизация: использование power of two для дерева
        if memory_length & (memory_length - 1) != 0:
            # Найти ближайшую степень двойки
            memory_length = 1 << (memory_length - 1).bit_length()
            warnings.warn(f"Adjusted memory length to power of two: {memory_length}")

        self.memory_length = memory_length
        self.probability_array = np.zeros((memory_length - 1) + memory_length, dtype=np.float32)
        self.maximum_length = len(self.probability_array)
        self.current_length = self.memory_length - 1
        self.indices = None
        self.tree_depth = (self.maximum_length.bit_length() - 1)

        # Оптимизация: предвычисленные индексы для обновления
        self._precomputed_indices = self._precompute_tree_indices()

        self.per_alpha = PER_ALPHA_DEFAULT
        self.per_beta = PER_BETA_DEFAULT

    def _precompute_tree_indices(self) -> np.ndarray:
        """Предвычисление индексов дерева для оптимизации."""
        indices = np.arange(self.maximum_length, dtype=np.int32)
        return indices

    def update_id(self, data_id: int, probability: float = MAX_PROBABILITY):
        """Оптимизированное обновление одного ID."""
        tree_id = data_id + self.memory_length - 1
        if self.current_length == tree_id:
            self.current_length += 1

        delta = probability - self.probability_array[tree_id]
        self.probability_array[tree_id] = probability

        # Оптимизация: развернутый цикл
        while tree_id:
            tree_id = (tree_id - 1) >> 1  # Используем битовый сдвиг вместо деления
            self.probability_array[tree_id] += delta

    def update_ids(self, data_ids, probability: float = MAX_PROBABILITY):
        """Оптимизированное обновление нескольких ID."""
        tree_ids = data_ids + self.memory_length - 1
        self.current_length += np.count_nonzero(tree_ids >= self.current_length)

        self.probability_array[tree_ids] = probability
        self._update_parent_nodes_vectorized(tree_ids)

    def _update_parent_nodes_vectorized(self, tree_ids: np.ndarray):
        """Векторизованное обновление родительских узлов."""
        unique_parents = np.unique((tree_ids - 1) // 2)

        for parent in unique_parents:
            left = (parent << 1) + 1  # parent * 2 + 1
            right = left + 1
            self.probability_array[parent] = self.probability_array[left] + self.probability_array[right]

        # Обновление корня
        self.probability_array[0] = self.probability_array[1] + self.probability_array[2]

    def get_leaf_id(self, value: float) -> int:
        """Оптимизированный поиск листа по значению."""
        idx = 0
        prob_array = self.probability_array

        while True:
            left_idx = (idx << 1) + 1  # idx * 2 + 1

            if left_idx >= self.maximum_length:
                return min(idx, self.current_length - 2)

            if value <= prob_array[left_idx]:
                idx = left_idx
            else:
                value -= prob_array[left_idx]
                idx = left_idx + 1

    def get_indices_is_weights(self, batch_size: int, begin_index: int,
                               end_index: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Оптимизированное получение индексов и весов."""
        self.per_beta = min(1.0, self.per_beta + PER_BETA_INCREMENT)

        # Оптимизация: векторизованная генерация случайных значений
        random_values = self._generate_random_values_vectorized(batch_size)

        # Векторизованный поиск листьев
        leaf_ids = np.array([self.get_leaf_id(v) for v in random_values], dtype=np.int32)
        self.indices = leaf_ids - (self.memory_length - 1)

        # Оптимизация: вычисление минимальной вероятности
        if end_index:
            prob_slice = self.probability_array[begin_index:end_index]
        else:
            prob_slice = self.probability_array[begin_index:]

        min_prob = prob_slice.min()
        if min_prob < NUMERICAL_STABILITY_EPS:
            min_prob = NUMERICAL_STABILITY_EPS

        probabilities = self.probability_array[leaf_ids] / min_prob
        importance_weights = np.power(probabilities, -self.per_beta, dtype=np.float32)

        return self.indices, importance_weights

    def _generate_random_values_vectorized(self, batch_size: int) -> np.ndarray:
        """Векторизованная генерация случайных значений."""
        total_prob = self.probability_array[0]
        if total_prob < NUMERICAL_STABILITY_EPS:
            return np.zeros(batch_size, dtype=np.float32)

        # Оптимизация: один вызов rand вместо двух
        random_base = rd.rand(batch_size)
        arange = np.arange(batch_size, dtype=np.float32)

        return (random_base + arange) * (total_prob / batch_size)

    def td_error_update(self, td_error):
        """Оптимизированное обновление на основе TD ошибки."""
        if self.indices is None or len(self.indices) == 0:
            return

        # Оптимизация: избегание лишних копий
        if isinstance(td_error, torch.Tensor):
            td_error_np = td_error.detach().cpu().numpy()
        else:
            td_error_np = np.asarray(td_error)

        probabilities = np.power(
            np.clip(td_error_np.squeeze(), CLAMP_MIN, CLAMP_MAX),
            self.per_alpha,
            dtype=np.float32
        )

        self.update_ids(self.indices, probabilities)

    def reset(self):
        """Оптимизированный сброс дерева."""
        self.probability_array.fill(0.0)
        self.current_length = self.memory_length - 1
        self.indices = None


class ReplayBuffer:
    """Оптимизированный буфер воспроизведения опыта."""

    __slots__ = ('index_manager', 'maximum_length', 'is_on_policy',
                 'action_dimension', 'device', 'storage', 'data_converter',
                 'batch_sampler', 'priority_tree', '_cached_length')

    def __init__(self, config: BufferConfig):
        """Инициализация оптимизированного буфера.

        Args:
            config: Конфигурация буфера
        """
        self._validate_config(config)

        self.index_manager = BufferIndexManager(config.max_len)
        self.maximum_length = config.max_len
        self.is_on_policy = config.is_on_policy
        self.action_dimension = 1 if config.is_discrete else config.action_dim
        self.device = self._get_optimal_device()

        # Оптимизация: кэшированная длина
        self._cached_length = 0

        # Инициализация компонентов
        other_dimension = self._calculate_other_dimension(config)
        use_torch_storage = not config.is_on_policy

        self.storage = OptimizedStorage(config.max_len, config.state_dim, other_dimension, use_torch_storage)
        self.data_converter = self._create_data_converter(config.is_on_policy)
        self.batch_sampler = self._create_batch_sampler(config, use_torch_storage)

    def _validate_config(self, config: BufferConfig) -> None:
        """Валидация конфигурации."""
        if config.max_len < MIN_BUFFER_SIZE:
            raise BufferInitializationError(f"Invalid buffer size: {config.max_len}")

    @staticmethod
    def _get_optimal_device() -> torch.device:
        """Определение оптимального устройства для вычислений."""
        if torch.cuda.is_available():
            # Проверка доступности GPU памяти
            try:
                torch.cuda.empty_cache()
                return torch.device("cuda")
            except RuntimeError:
                warnings.warn("CUDA available but cannot initialize, falling back to CPU")
                return torch.device("cpu")
        return torch.device("cpu")

    def _calculate_other_dimension(self, config: BufferConfig) -> int:
        """Оптимизированное вычисление размерности других данных."""
        if config.is_on_policy:
            return 1 + 1 + self.action_dimension + config.action_dim
        else:
            return 1 + 1 + self.action_dimension

    def _create_data_converter(self, is_on_policy: bool) -> DataConverter:
        """Создание конвертера данных с учетом оптимизации."""
        if is_on_policy:
            return NumpyDataConverter()
        else:
            return TorchDataConverter()

    def _create_batch_sampler(self, config: BufferConfig, use_torch_storage: bool) -> BatchSampler:
        """Создание оптимизированного сэмплера батчей."""
        should_use_per = bool(config.use_per_or_gae and config.is_on_policy)

        if should_use_per:
            self.priority_tree = OptimizedBinarySearchTree(config.max_len)
            return PriorityBatchSampler(self.priority_tree, config.max_len)
        else:
            self.priority_tree = None
            return UniformBatchSampler()

    def append_buffer(self, state, other):
        """Оптимизированное добавление одной пары состояние-другое."""
        idx = self.index_manager.next_index
        self.storage.store(idx, state, other)

        if self.priority_tree:
            self.priority_tree.update_id(idx)

        self.index_manager.update_after_append()
        self._cached_length = -1  # Инвалидация кэша

    def extend_buffer(self, state, other):
        """Оптимизированное добавление пакета данных."""
        batch_size = len(other)

        # Получение срезов для записи
        slices = self.index_manager.calculate_indices_for_batch(batch_size)

        # Обновление приоритетов
        if self.priority_tree:
            self._update_priority_indices_optimized(batch_size)

        # Векторизованная запись
        self._write_to_storage_optimized(state, other, slices)

        self._cached_length = -1  # Инвалидация кэша

    def _update_priority_indices_optimized(self, batch_size: int):
        """Оптимизированное обновление индексов приоритетов."""
        start_idx = self.index_manager.next_index
        data_indices = np.arange(start_idx, start_idx + batch_size, dtype=np.int32) % self.maximum_length
        self.priority_tree.update_ids(data_indices)

    def _write_to_storage_optimized(self, state, other, slices: Tuple[slice, slice, slice, slice]):
        """Оптимизированная запись в хранилище."""
        state_slice, other_slice, wrap_state_slice, wrap_other_slice = slices

        if wrap_state_slice.start == wrap_state_slice.stop:
            # Без переполнения
            self.storage.batch_store_vectorized(state_slice.start, state, other)
        else:
            # С переполнением
            first_part_size = self.maximum_length - state_slice.start

            # Первая часть
            self.storage.batch_store_vectorized(
                state_slice.start,
                state[:first_part_size],
                other[:first_part_size]
            )

            # Вторая часть
            self.storage.batch_store_vectorized(
                0,
                state[first_part_size:],
                other[first_part_size:]
            )

    def extend_buffer_from_list(self, trajectory_list: List[Tuple]):
        """Оптимизированное добавление из списка траекторий."""
        state_array = self.data_converter.convert_state(trajectory_list)
        other_array = self.data_converter.convert_other(trajectory_list)

        if len(state_array) > 0:
            self.extend_buffer(state_array, other_array)

    def sample_batch(self, batch_size: int) -> tuple:
        """Оптимизированная выборка батча данных."""
        # Использование кэшированной длины
        if self._cached_length < 0:
            self.index_manager.update_current_length()
            self._cached_length = self.index_manager.current_length

        return self.batch_sampler.sample(
            batch_size,
            self.storage,
            self._cached_length,
            self.device
        )

    def sample_all(self) -> tuple:
        """Оптимизированная выборка всех данных."""
        if self._cached_length < 0:
            self.index_manager.update_current_length()
            self._cached_length = self.index_manager.current_length

        states, others = self.storage.get_slice(slice(0, self._cached_length))

        # Конвертация если нужно
        if isinstance(states, np.ndarray):
            states = torch.as_tensor(states, device=self.device)
            others = torch.as_tensor(others, device=self.device)

        # Оптимизация: прямое извлечение срезов
        rewards = others[:, 0]
        masks = others[:, 1]
        actions = others[:, 2:2 + self.action_dimension]
        noises_or_probs = others[:, 2 + self.action_dimension:]

        return rewards, masks, actions, noises_or_probs, states

    @property
    def current_length(self) -> int:
        """Оптимизированное получение текущей длины."""
        if self._cached_length < 0:
            self.index_manager.update_current_length()
            self._cached_length = self.index_manager.current_length
        return self._cached_length

    def update_current_length(self):
        """Оптимизированное обновление текущей длины."""
        self.index_manager.update_current_length()
        self._cached_length = self.index_manager.current_length

    def empty_buffer(self):
        """Оптимизированная очистка буфера."""
        self.index_manager.reset()
        if self.priority_tree:
            self.priority_tree.reset()
        self._cached_length = 0

    def td_error_update(self, td_error):
        """Оптимизированное обновление TD ошибки."""
        if self.priority_tree:
            self.priority_tree.td_error_update(td_error)


class ReplayBufferMP:
    """Оптимизированный многопроцессный буфер воспроизведения."""

    __slots__ = ('worker_number', 'buffers', '_cached_total_length')

    def __init__(self, config: BufferConfig, worker_num: int):
        """Инициализация оптимизированного многопроцессного буфера.

        Args:
            config: Конфигурация буфера
            worker_num: Количество воркеров
        """
        if worker_num < MIN_WORKER_NUM:
            raise BufferInitializationError(f"Invalid worker number: {worker_num}")

        self.worker_number = worker_num
        self._cached_total_length = -1

        # Оптимизация: предварительное выделение списка
        self.buffers = [None] * worker_num

        # Параллельная инициализация буферов
        buffer_max_len = config.max_len // worker_num
        adjusted_config = BufferConfig(
            max_len=buffer_max_len,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            is_discrete=config.is_discrete,
            is_on_policy=config.is_on_policy,
            use_per_or_gae=config.use_per_or_gae
        )

        for i in range(worker_num):
            self.buffers[i] = ReplayBuffer(adjusted_config)

    @property
    def current_length(self) -> int:
        """Оптимизированное получение общей длины."""
        if self._cached_total_length < 0:
            total = 0
            for buffer in self.buffers:
                total += buffer.current_length
            self._cached_total_length = total
        return self._cached_total_length

    def sample_batch(self, batch_size: int) -> list:
        """Оптимизированная выборка батча из всех буферов."""
        batch_per_worker = max(1, batch_size // self.worker_number)

        # Параллельная выборка (в идеале должна быть настоящая параллельность)
        items_per_worker = [buf.sample_batch(batch_per_worker) for buf in self.buffers]

        # Оптимизация: предварительное определение размеров
        num_items = len(items_per_worker[0])
        result = [None] * num_items

        # Векторизованное объединение
        for i in range(num_items):
            tensors = [items[i] for items in items_per_worker]
            result[i] = torch.cat(tensors, dim=0)

        return result

    def empty_buffer(self):
        """Оптимизированная очистка всех буферов."""
        for buffer in self.buffers:
            buffer.empty_buffer()
        self._cached_total_length = 0


# Фабричные функции для обратной совместимости
def create_replay_buffer(max_len: int, state_dim: int, action_dim: int,
                         if_discrete: bool, if_on_policy: bool, if_per_or_gae: bool) -> ReplayBuffer:
    """Создание буфера с устаревшим интерфейсом для обратной совместимости."""
    config = BufferConfig(
        max_len=max_len,
        state_dim=state_dim,
        action_dim=action_dim,
        is_discrete=if_discrete,
        is_on_policy=if_on_policy,
        use_per_or_gae=if_per_or_gae
    )
    return ReplayBuffer(config)


def create_replay_buffer_mp(max_len: int, worker_num: int, state_dim: int, action_dim: int,
                            if_discrete: bool, if_on_policy: bool, if_per_or_gae: bool) -> ReplayBufferMP:
    """Создание многопроцессного буфера с устаревшим интерфейсом."""
    config = BufferConfig(
        max_len=max_len,
        state_dim=state_dim,
        action_dim=action_dim,
        is_discrete=if_discrete,
        is_on_policy=if_on_policy,
        use_per_or_gae=if_per_or_gae
    )
    return ReplayBufferMP(config, worker_num)


# Экспорт для обратной совместимости
ReplayBuffer = ReplayBuffer  # type: ignore
ReplayBufferMP = ReplayBufferMP  # type: ignore
BinarySearchTree = OptimizedBinarySearchTree  # type: ignore