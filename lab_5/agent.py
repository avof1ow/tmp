import torch
import numpy as np
import numpy.random as rd
from copy import deepcopy
import logging
import os
import traceback
from datetime import datetime
from elegantrl2.tutorial.net import QNet, QNetTwin
from elegantrl2.tutorial.net import Actor, ActorSAC, ActorPPO, ActorDiscretePPO
from elegantrl2.tutorial.net import Critic, CriticAdv, CriticTwin


# Настройка системы логирования
def setup_logging(log_level=logging.INFO):
    """Настройка системы логирования для RL агентов"""

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rl_agent_{timestamp}.log")

    logger = logging.getLogger("RLAgent")
    logger.setLevel(log_level)

    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info("Система логирования инициализирована")
    logger.info(f"Логи будут сохраняться в: {log_file}")
    logger.info("=" * 60)

    return logger


# Инициализируем глобальный логгер
try:
    logger = setup_logging()
    logger.info("Глобальный логгер успешно инициализирован")
except Exception as e:
    print(f"Ошибка при инициализации логирования: {e}")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("RLAgent")


class AgentBase:
    def __init__(self):
        self.state = None
        self.device = None
        self.action_dim = None
        self.if_on_policy = False
        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_optim = self.Cri = None
        self.act = self.act_optim = self.Act = None
        self.cri_target = self.if_use_cri_target = None
        self.act_target = self.if_use_act_target = None
        self.logger = logger.getChild(self.__class__.__name__)

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        try:
            self.logger.info("Инициализация агента...")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.action_dim = action_dim

            self.cri = self.Cri(net_dim, state_dim, action_dim).to(self.device)
            self.act = self.Act(net_dim, state_dim, action_dim).to(self.device) if self.Act is not None else self.cri
            self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
            self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

            self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
            self.act_optim = torch.optim.Adam(self.act.parameters(),
                                              learning_rate) if self.Act is not None else self.cri
            del self.Cri, self.Act, self.if_use_cri_target, self.if_use_act_target

            self.logger.info(f"Агент инициализирован: device={self.device}")

        except Exception as e:
            self.logger.error(f"Ошибка инициализации: {str(e)}")
            raise

    def select_action(self, state) -> np.ndarray:
        pass

    def explore_env(self, env, target_step, reward_scale, gamma) -> list:
        trajectory_list = []

        state = self.state
        for _ in range(target_step):
            action = self.select_action(state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            trajectory_list.append((state, other))

            state = env.reset() if done else next_s
        self.state = state
        return trajectory_list

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1 - tau))


class AgentDQN(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_rate = 0.25
        self.if_use_cri_target = True
        self.Cri = QNet

    def select_action(self, state) -> int:
        if rd.rand() < self.explore_rate:
            a_int = rd.randint(self.action_dim)
        else:
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
            action = self.act(states)[0]
            a_int = action.argmax(dim=0).detach().cpu().numpy()
        return a_int


class AgentDoubleDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.softMax = torch.nn.Softmax(dim=1)
        self.Cri = QNetTwin


class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.1
        self.if_use_cri_target = self.if_use_act_target = True
        self.Act = Actor
        self.Cri = Critic

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act(states)[0]
        action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.cpu().numpy()


class AgentTD3(AgentDDPG):
    def __init__(self):
        super().__init__()
        self.policy_noise = 0.2
        self.update_freq = 2
        self.Cri = CriticTwin


class AgentSAC(AgentBase):
    def __init__(self):
        super().__init__()
        self.if_use_cri_target = True
        self.Act = ActorSAC
        self.Cri = CriticTwin

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act.get_action(states)[0]
        return action.cpu().numpy()


class AgentPPO(AgentBase):
    def __init__(self):
        super().__init__()
        self.if_on_policy = True
        self.ratio_clip = 0.2
        self.lambda_entropy = 0.02
        self.Act = ActorPPO
        self.Cri = CriticAdv

    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(states)
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()


class AgentDiscretePPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.Act = ActorDiscretePPO


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_discrete, if_on_policy):
        """Инициализация буфера с подробным логированием"""
        try:
            self.now_len = 0
            self.next_idx = 0
            self.if_full = False
            self.max_len = max_len
            self.if_on_policy = if_on_policy
            self.action_dim = 1 if if_discrete else action_dim
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger = logger.getChild("ReplayBuffer")

            # Статистика использования
            self.stats = {
                'add_count': 0,
                'sample_count': 0,
                'overflow_count': 0,
                'last_sample_time': None,
                'sample_sizes': []
            }

            self.logger.info("=" * 50)
            self.logger.info(f"ИНИЦИАЛИЗАЦИЯ REPLAY BUFFER")
            self.logger.info(f"Максимальный размер: {max_len:,}")
            self.logger.info(f"Размер состояния: {state_dim}")
            self.logger.info(f"Размер действия: {action_dim}")
            self.logger.info(f"Тип: {'On-Policy' if if_on_policy else 'Off-Policy'}")
            self.logger.info(f"Дискретный: {if_discrete}")

            # Расчет памяти
            if if_on_policy:
                other_dim = 1 + 1 + self.action_dim + action_dim
                self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
                self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)

                state_mb = self.buf_state.nbytes / (1024 * 1024)
                other_mb = self.buf_other.nbytes / (1024 * 1024)
                total_mb = state_mb + other_mb

                self.logger.info(f"Память буфера (on-policy):")
                self.logger.info(f"  - Состояния: {state_mb:.2f} MB")
                self.logger.info(f"  - Другие данные: {other_mb:.2f} MB")
                self.logger.info(f"  - Всего: {total_mb:.2f} MB")

            else:
                other_dim = 1 + 1 + self.action_dim
                self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)
                self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)

                # Расчет памяти для PyTorch тензоров
                state_elements = self.buf_state.numel()
                other_elements = self.buf_other.numel()
                element_size = self.buf_state.element_size()

                state_mb = (state_elements * element_size) / (1024 * 1024)
                other_mb = (other_elements * element_size) / (1024 * 1024)
                total_mb = state_mb + other_mb

                self.logger.info(f"Память буфера (off-policy):")
                self.logger.info(f"  - Состояния: {state_mb:.2f} MB")
                self.logger.info(f"  - Другие данные: {other_mb:.2f} MB")
                self.logger.info(f"  - Всего: {total_mb:.2f} MB")
                self.logger.info(f"  - Устройство: {self.device}")

            self.logger.info("Буфер успешно инициализирован ✓")
            self.logger.info("=" * 50)

        except Exception as e:
            logger.error(f"КРИТИЧЕСКАЯ ОШИБКА ПРИ СОЗДАНИИ БУФЕРА: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def append_buffer(self, state, other):
        """Добавление одного элемента в буфер"""
        try:
            if self.next_idx >= self.max_len:
                self.logger.warning("Попытка добавить в полный буфер, будет перезапись")
                self.stats['overflow_count'] += 1

            self.buf_state[self.next_idx] = state
            self.buf_other[self.next_idx] = other

            old_idx = self.next_idx
            self.next_idx += 1

            if self.next_idx >= self.max_len:
                if not self.if_full:
                    self.logger.info(f"БУФЕР ПОЛОН! Начата перезапись")
                self.if_full = True
                self.next_idx = 0

            self.stats['add_count'] += 1

            # Периодическое логирование
            if self.stats['add_count'] % 10000 == 0:
                self._log_buffer_status()

        except Exception as e:
            self.logger.error(f"Ошибка при добавлении в буфер: {str(e)}")
            raise

    def extend_buffer(self, state, other):
        """Добавление нескольких элементов в буфер"""
        try:
            size = len(other)
            if size == 0:
                self.logger.warning("Попытка добавить 0 элементов")
                return

            self.logger.debug(f"Добавление {size} элементов в буфер")

            old_len = self.now_len
            old_next_idx = self.next_idx

            next_idx = self.next_idx + size

            if next_idx > self.max_len:
                # Переполнение с перезаписью
                first_part = self.max_len - self.next_idx
                second_part = size - first_part

                self.buf_state[self.next_idx:self.max_len] = state[:first_part]
                self.buf_other[self.next_idx:self.max_len] = other[:first_part]

                self.buf_state[0:second_part] = state[-second_part:]
                self.buf_other[0:second_part] = other[-second_part:]

                self.if_full = True
                self.next_idx = second_part

                self.logger.info(f"БУФЕР ПЕРЕПОЛНЕН! Добавлено {size} элементов")
                self.logger.info(f"  - Первая часть: {first_part} элементов")
                self.logger.info(f"  - Вторая часть: {second_part} элементов (перезапись)")
                self.stats['overflow_count'] += 1

            else:
                # Обычное добавление
                self.buf_state[self.next_idx:next_idx] = state
                self.buf_other[self.next_idx:next_idx] = other
                self.next_idx = next_idx

            self.stats['add_count'] += size
            self.update_now_len()

            # Логирование изменения размера
            if self.now_len != old_len:
                self.logger.debug(f"Размер буфера изменился: {old_len:,} → {self.now_len:,}")

            # Предупреждение при быстром заполнении
            if self.now_len >= self.max_len * 0.7 and old_len < self.max_len * 0.7:
                self.logger.warning(f"Буфер заполнен на {self.now_len / self.max_len:.1%}")

        except Exception as e:
            self.logger.error(f"Ошибка при расширении буфера: {str(e)}")
            raise

    def extend_buffer_from_list(self, trajectory_list):
        """Добавление данных из списка траекторий"""
        try:
            if not trajectory_list:
                self.logger.warning("Пустой список траекторий")
                return

            num_trajectories = len(trajectory_list)
            self.logger.info(f"Добавление {num_trajectories} траекторий")

            if self.if_on_policy:
                state = np.array([item[0] for item in trajectory_list], dtype=np.float32)
                other = np.array([item[1] for item in trajectory_list], dtype=np.float32)
                self.logger.debug(f"On-policy: конвертировано в numpy массивы")
            else:
                state = torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float32)
                other = torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float32)
                self.logger.debug(f"Off-policy: конвертировано в torch тензоры")

            old_fill_ratio = self.now_len / self.max_len
            self.extend_buffer(state, other)
            new_fill_ratio = self.now_len / self.max_len

            # Логирование изменений
            fill_change = new_fill_ratio - old_fill_ratio
            self.logger.info(f"Буфер обновлен: +{num_trajectories} траекторий")
            self.logger.info(f"Заполнение: {old_fill_ratio:.1%} → {new_fill_ratio:.1%} (+{fill_change:.1%})")

            # Статистика по наградам (для отладки)
            if trajectory_list and len(trajectory_list) > 0:
                rewards = [t[1][0] for t in trajectory_list]
                avg_reward = np.mean(rewards) if self.if_on_policy else torch.tensor(rewards).mean().item()
                self.logger.debug(f"Средняя награда в добавленных траекториях: {avg_reward:.3f}")

        except Exception as e:
            self.logger.error(f"Ошибка при добавлении траекторий: {str(e)}")
            raise

    def sample_batch(self, batch_size) -> tuple:
        """Выборка батча данных с валидацией"""
        try:
            self.stats['last_sample_time'] = datetime.now()

            if self.now_len < batch_size:
                error_msg = f"Недостаточно данных: {self.now_len} < {batch_size}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            if self.now_len == 0:
                error_msg = "Буфер пуст"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Валидация индексов
            max_idx = min(self.now_len - 1, self.max_len - 2)
            if max_idx <= 0:
                error_msg = "Недостаточно данных для выборки"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            indices = rd.randint(0, max_idx, size=batch_size)
            self.stats['sample_count'] += 1
            self.stats['sample_sizes'].append(batch_size)

            # Ограничение истории выборок
            if len(self.stats['sample_sizes']) > 1000:
                self.stats['sample_sizes'] = self.stats['sample_sizes'][-1000:]

            self.logger.debug(f"Выборка батча: размер={batch_size}, "
                              f"доступно={self.now_len}, "
                              f"индексы {indices.min()}-{indices.max()}")

            other = self.buf_other[indices]

            return (other[:, 0:1],  # reward
                    other[:, 1:2],  # mask
                    other[:, 2:],  # action
                    self.buf_state[indices],  # state
                    self.buf_state[indices + 1])  # next state

        except Exception as e:
            self.logger.error(f"Ошибка при выборке батча: {str(e)}")
            raise

    def sample_all(self) -> tuple:
        """Выборка всех данных из буфера (для on-policy)"""
        try:
            if self.now_len == 0:
                self.logger.error("Попытка выборки из пустого буфера")
                return None

            self.logger.info(f"Выборка ВСЕХ данных: {self.now_len} элементов")

            all_state = torch.as_tensor(self.buf_state[:self.now_len], device=self.device)
            all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)

            # Логирование статистики
            if self.now_len > 0:
                rewards = all_other[:, 0]
                avg_reward = rewards.mean().item()
                std_reward = rewards.std().item()

                self.logger.info(f"Статистика выборки:")
                self.logger.info(f"  - Средняя награда: {avg_reward:.3f}")
                self.logger.info(f"  - Std награды: {std_reward:.3f}")
                self.logger.info(f"  - Min награда: {rewards.min().item():.3f}")
                self.logger.info(f"  - Max награда: {rewards.max().item():.3f}")

            return (all_other[:, 0],  # reward
                    all_other[:, 1],  # mask
                    all_other[:, 2:2 + self.action_dim],  # action
                    all_other[:, 2 + self.action_dim:],  # action_noise or action_prob
                    all_state,)  # state

        except Exception as e:
            self.logger.error(f"Ошибка при выборке всех данных: {str(e)}")
            raise

    def update_now_len(self):
        """Обновление текущей длины буфера"""
        old_len = self.now_len
        self.now_len = self.max_len if self.if_full else self.next_idx

        # Логирование при значительных изменениях
        if old_len != self.now_len:
            if self.now_len == self.max_len and old_len < self.max_len:
                self.logger.info("БУФЕР ДОСТИГ МАКСИМАЛЬНОГО РАЗМЕРА")

            fill_ratio = self.now_len / self.max_len

            # Логирование при достижении ключевых точек заполнения
            if fill_ratio >= 0.5 and old_len / self.max_len < 0.5:
                self.logger.info(f"Буфер заполнен на 50% ({self.now_len:,}/{self.max_len:,})")
            elif fill_ratio >= 0.9 and old_len / self.max_len < 0.9:
                self.logger.warning(f"Буфер заполнен на 90% ({self.now_len:,}/{self.max_len:,})")

    def empty_buffer(self):
        """Очистка буфера с логированием"""
        old_len = self.now_len
        old_fill_ratio = self.now_len / self.max_len if self.max_len > 0 else 0

        self.now_len = 0
        self.next_idx = 0
        self.if_full = False

        self.logger.info("=" * 40)
        self.logger.info("ОЧИСТКА БУФЕРА")
        self.logger.info(f"Было элементов: {old_len:,}")
        self.logger.info(f"Заполнение было: {old_fill_ratio:.1%}")
        self.logger.info("Буфер успешно очищен ✓")
        self.logger.info("=" * 40)

        # Сброс статистики
        self.stats = {
            'add_count': 0,
            'sample_count': 0,
            'overflow_count': 0,
            'last_sample_time': None,
            'sample_sizes': []
        }

    def _log_buffer_status(self):
        """Логирование текущего статуса буфера"""
        fill_ratio = self.now_len / self.max_len

        self.logger.info("=" * 40)
        self.logger.info("СТАТУС БУФЕРА")
        self.logger.info(f"Текущий размер: {self.now_len:,}/{self.max_len:,}")
        self.logger.info(f"Заполнение: {fill_ratio:.1%}")
        self.logger.info(f"Добавлено элементов: {self.stats['add_count']:,}")
        self.logger.info(f"Выборок: {self.stats['sample_count']:,}")
        self.logger.info(f"Переполнений: {self.stats['overflow_count']:,}")

        if self.stats['sample_sizes']:
            avg_sample_size = np.mean(self.stats['sample_sizes'][-100:])
            self.logger.info(f"Средний размер выборки: {avg_sample_size:.1f}")

        if self.stats['last_sample_time']:
            time_since = (datetime.now() - self.stats['last_sample_time']).total_seconds()
            self.logger.info(f"Время с последней выборки: {time_since:.1f} сек")

        self.logger.info("=" * 40)

    def get_buffer_stats(self):
        """Получение статистики буфера"""
        return {
            'current_size': self.now_len,
            'max_size': self.max_len,
            'fill_ratio': self.now_len / self.max_len if self.max_len > 0 else 0,
            'is_full': self.if_full,
            'add_count': self.stats['add_count'],
            'sample_count': self.stats['sample_count'],
            'overflow_count': self.stats['overflow_count']
        }