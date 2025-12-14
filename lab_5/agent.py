import torch
import numpy as np
import numpy.random as rd
from copy import deepcopy
import logging
import os
import sys
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from elegantrl2.tutorial.net import QNet, QNetTwin
from elegantrl2.tutorial.net import Actor, ActorSAC, ActorPPO, ActorDiscretePPO
from elegantrl2.tutorial.net import Critic, CriticAdv, CriticTwin


# ============================================================================
# КОНФИГУРАЦИЯ ЛОГИРОВАНИЯ
# ============================================================================

class RLFilter(logging.Filter):
    """Фильтр для RL логов"""

    def filter(self, record):
        # Добавляем дополнительную информацию в записи логов
        if not hasattr(record, 'agent_type'):
            record.agent_type = 'Unknown'
        if not hasattr(record, 'training_step'):
            record.training_step = 0
        return True


def setup_logging(log_level=logging.INFO, log_to_file=True, max_log_size=10, backup_count=5):
    """
    Настройка системы логирования с ротацией файлов

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Записывать ли логи в файл
        max_log_size: Максимальный размер лог-файла в MB
        backup_count: Количество backup файлов
    """

    # Создаем директорию для логов
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Создаем основной логгер
    logger = logging.getLogger("RLAgent")
    logger.setLevel(log_level)

    # Очищаем существующие обработчики
    logger.handlers.clear()

    # Добавляем фильтр
    logger.addFilter(RLFilter())

    # Форматтер с цветами для консоли
    class ColorFormatter(logging.Formatter):
        """Форматтер с цветами для разных уровней логирования"""
        COLORS = {
            'DEBUG': '\033[94m',  # Синий
            'INFO': '\033[92m',  # Зеленый
            'WARNING': '\033[93m',  # Желтый
            'ERROR': '\033[91m',  # Красный
            'CRITICAL': '\033[91m',  # Красный
            'RESET': '\033[0m'  # Сброс
        }

        def format(self, record):
            # Добавляем цвета для консоли
            if sys.stdout.isatty():  # Только если вывод в терминал
                levelname = record.levelname
                if levelname in self.COLORS:
                    record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                    record.msg = f"{self.COLORS[levelname]}{record.msg}{self.COLORS['RESET']}"

            return super().format(record)

    # Базовый форматтер (без цветов для файла)
    base_formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(name)-20s | %(levelname)-8s | %(agent_type)-15s | Step: %(training_step)-8d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Форматтер для консоли (с цветами)
    console_formatter = ColorFormatter(
        '%(asctime)s.%(msecs)03d | %(name)-20s | %(levelname)-8s | %(agent_type)-15s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(RLFilter())
    logger.addHandler(console_handler)

    # Обработчик для файла (с ротацией)
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"rl_training_{timestamp}.log")

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_log_size * 1024 * 1024,  # MB в байты
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(base_formatter)
        file_handler.addFilter(RLFilter())
        logger.addHandler(file_handler)

        # Создаем также summary лог
        summary_handler = RotatingFileHandler(
            os.path.join(log_dir, "rl_summary.log"),
            maxBytes=max_log_size * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        summary_handler.setLevel(logging.INFO)  # Только важные сообщения
        summary_handler.setFormatter(base_formatter)
        summary_handler.addFilter(RLFilter())

        # Добавляем фильтр для summary лога
        class SummaryFilter(logging.Filter):
            def filter(self, record):
                # Фильтруем только важные сообщения
                if record.levelno >= logging.INFO:
                    # Исключаем слишком частые DEBUG сообщения
                    if "DEBUG" in record.getMessage() or "шаг" in record.getMessage().lower():
                        return False
                    return True
                return False

        summary_handler.addFilter(SummaryFilter())
        logger.addHandler(summary_handler)

        logger.info("=" * 80)
        logger.info("СИСТЕМА ЛОГИРОВАНИЯ RL АГЕНТОВ")
        logger.info(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Уровень логирования: {logging.getLevelName(log_level)}")
        logger.info(f"Лог файл: {log_file}")
        logger.info(f"Макс. размер файла: {max_log_size} MB")
        logger.info(f"Backup файлов: {backup_count}")
        logger.info("=" * 80)

    return logger


# Инициализируем глобальный логгер
try:
    # Можно менять уровень логирования через переменные окружения
    log_level = os.getenv('RL_LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level, logging.INFO)

    logger = setup_logging(
        log_level=log_level,
        log_to_file=True,
        max_log_size=10,  # 10MB
        backup_count=3
    )
    logger.info("Глобальный логгер инициализирован", extra={'agent_type': 'System', 'training_step': 0})

except Exception as e:
    print(f"Ошибка при инициализации логирования: {e}")
    # Fallback на базовое логирование
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger("RLAgent")


# ============================================================================
# БАЗОВЫЙ КЛАСС АГЕНТА С УЛУЧШЕННЫМ ЛОГИРОВАНИЕМ
# ============================================================================

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

        # Инициализация логгера с дополнительными атрибутами
        self.logger = logger.getChild(self.__class__.__name__)
        self.agent_type = self.__class__.__name__
        self.training_step = 0

        # Статистика для логирования
        self.episode_count = 0
        self.total_reward = 0.0
        self.best_reward = -float('inf')

        self.logger.debug(f"Агент {self.agent_type} создан",
                          extra={'agent_type': self.agent_type, 'training_step': self.training_step})

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        """Инициализация агента"""
        try:
            self.action_dim = action_dim
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.cri = self.Cri(net_dim, state_dim, action_dim).to(self.device)
            self.act = self.Act(net_dim, state_dim, action_dim).to(self.device) if self.Act is not None else self.cri
            self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
            self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

            self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
            self.act_optim = torch.optim.Adam(self.act.parameters(),
                                              learning_rate) if self.Act is not None else self.cri
            del self.Cri, self.Act, self.if_use_cri_target, self.if_use_act_target

            # Логирование инициализации
            cri_params = sum(p.numel() for p in self.cri.parameters())
            log_msg = f"Агент инициализирован: device={self.device}, params={cri_params:,}"
            if self.Act is not None:
                act_params = sum(p.numel() for p in self.act.parameters())
                log_msg += f", actor_params={act_params:,}"

            self.logger.info(log_msg,
                             extra={'agent_type': self.agent_type, 'training_step': self.training_step})

        except Exception as e:
            self.logger.error(f"Ошибка инициализации: {str(e)}",
                              extra={'agent_type': self.agent_type, 'training_step': self.training_step})
            raise

    def select_action(self, state) -> np.ndarray:
        pass

    def explore_env(self, env, target_step, reward_scale, gamma) -> list:
        trajectory_list = []

        state = self.state
        for i in range(target_step):
            action = self.select_action(state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            trajectory_list.append((state, other))

            # Логирование прогресса каждые 100 шагов
            if i % 100 == 0:
                self.logger.debug(f"Exploration progress: {i}/{target_step} steps",
                                  extra={'agent_type': self.agent_type, 'training_step': self.training_step})

            if done:
                self.episode_count += 1
                self.logger.info(f"Episode {self.episode_count} completed",
                                 extra={'agent_type': self.agent_type, 'training_step': self.training_step})
                state = env.reset()
            else:
                state = next_s

        self.state = state

        if trajectory_list:
            rewards = [t[1][0] for t in trajectory_list]
            avg_reward = np.mean(rewards)
            self.logger.info(f"Exploration complete: {len(trajectory_list)} steps, avg reward: {avg_reward:.3f}",
                             extra={'agent_type': self.agent_type, 'training_step': self.training_step})

        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """Базовый метод обновления сети (переопределяется в дочерних классах)"""
        self.logger.info(f"Starting network update",
                         extra={'agent_type': self.agent_type, 'training_step': self.training_step})
        self.training_step += 1
        return 0.0, 0.0

    def log_training_progress(self, metrics):
        """Логирование прогресса обучения"""
        if self.training_step % 100 == 0:  # Каждые 100 шагов
            metric_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
            self.logger.info(f"Training progress - {metric_str}",
                             extra={'agent_type': self.agent_type, 'training_step': self.training_step})

        if self.training_step % 1000 == 0:  # Каждые 1000 шагов
            self.logger.info(f"Training milestone: {self.training_step} steps completed",
                             extra={'agent_type': self.agent_type, 'training_step': self.training_step})

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1 - tau))


# ============================================================================
# СПЕЦИАЛИЗИРОВАННЫЕ АГЕНТЫ
# ============================================================================

class AgentDQN(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_rate = 0.25
        self.if_use_cri_target = True
        self.Cri = QNet
        self.logger.info(f"DQN agent created with explore_rate={self.explore_rate}",
                         extra={'agent_type': self.agent_type, 'training_step': self.training_step})

    def select_action(self, state) -> int:
        if rd.rand() < self.explore_rate:
            a_int = rd.randint(self.action_dim)
            self.logger.debug(f"Random action: {a_int}",
                              extra={'agent_type': self.agent_type, 'training_step': self.training_step})
        else:
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
            action = self.act(states)[0]
            a_int = action.argmax(dim=0).detach().cpu().numpy()
            self.logger.debug(f"Greedy action: {a_int}, Q-values: {action.detach().cpu().numpy()}",
                              extra={'agent_type': self.agent_type, 'training_step': self.training_step})
        return a_int

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        super().update_net(buffer, batch_size, repeat_times, soft_update_tau)

        self.logger.info(f"DQN update started",
                         extra={'agent_type': self.agent_type, 'training_step': self.training_step})

        # Симуляция обновления
        loss = np.random.rand()
        self.log_training_progress({'loss': loss})

        return loss, 0.0


class AgentDoubleDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.softMax = torch.nn.Softmax(dim=1)
        self.Cri = QNetTwin
        self.logger.info("DoubleDQN agent created",
                         extra={'agent_type': self.agent_type, 'training_step': self.training_step})


class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.1
        self.if_use_cri_target = self.if_use_act_target = True
        self.Act = Actor
        self.Cri = Critic
        self.logger.info(f"DDPG agent created with noise={self.explore_noise}",
                         extra={'agent_type': self.agent_type, 'training_step': self.training_step})

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act(states)[0]
        noise = torch.randn_like(action) * self.explore_noise
        action = (action + noise).clamp(-1, 1)

        noise_norm = torch.norm(noise).item()
        if noise_norm > 1.0:
            self.logger.warning(f"High exploration noise: {noise_norm:.3f}",
                                extra={'agent_type': self.agent_type, 'training_step': self.training_step})

        return action.cpu().numpy()


class AgentTD3(AgentDDPG):
    def __init__(self):
        super().__init__()
        self.policy_noise = 0.2
        self.update_freq = 2
        self.Cri = CriticTwin
        self.logger.info(f"TD3 agent created with policy_noise={self.policy_noise}",
                         extra={'agent_type': self.agent_type, 'training_step': self.training_step})


class AgentSAC(AgentBase):
    def __init__(self):
        super().__init__()
        self.if_use_cri_target = True
        self.Act = ActorSAC
        self.Cri = CriticTwin
        self.logger.info("SAC agent created",
                         extra={'agent_type': self.agent_type, 'training_step': self.training_step})


class AgentPPO(AgentBase):
    def __init__(self):
        super().__init__()
        self.if_on_policy = True
        self.ratio_clip = 0.2
        self.lambda_entropy = 0.02
        self.Act = ActorPPO
        self.Cri = CriticAdv
        self.logger.info(f"PPO agent created with clip={self.ratio_clip}",
                         extra={'agent_type': self.agent_type, 'training_step': self.training_step})

    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(states)
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()


class AgentDiscretePPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.Act = ActorDiscretePPO
        self.logger.info("DiscretePPO agent created",
                         extra={'agent_type': self.agent_type, 'training_step': self.training_step})


# ============================================================================
# БУФЕР ВОСПРОИЗВЕДЕНИЯ С УЛУЧШЕННЫМ ЛОГИРОВАНИЕМ
# ============================================================================

class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_discrete, if_on_policy):
        try:
            self.now_len = 0
            self.next_idx = 0
            self.if_full = False
            self.max_len = max_len
            self.if_on_policy = if_on_policy
            self.action_dim = 1 if if_discrete else action_dim
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Создаем логгер для буфера
            self.logger = logger.getChild("ReplayBuffer")
            self.agent_type = "Buffer"
            self.training_step = 0

            # Статистика
            self.stats = {'added': 0, 'sampled': 0, 'overflows': 0}

            self.logger.info(f"Initializing buffer: max_len={max_len:,}, on_policy={if_on_policy}",
                             extra={'agent_type': self.agent_type, 'training_step': self.training_step})

            if if_on_policy:
                other_dim = 1 + 1 + self.action_dim + action_dim
                self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
                self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)
            else:
                other_dim = 1 + 1 + self.action_dim
                self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)
                self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)

            self.logger.info(f"Buffer initialized successfully",
                             extra={'agent_type': self.agent_type, 'training_step': self.training_step})

        except Exception as e:
            logger.error(f"Buffer initialization failed: {str(e)}",
                         extra={'agent_type': 'System', 'training_step': 0})
            raise

    def append_buffer(self, state, other):
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0
            self.stats['overflows'] += 1

            if self.stats['overflows'] % 10 == 0:
                self.logger.warning(f"Buffer overflowed {self.stats['overflows']} times",
                                    extra={'agent_type': self.agent_type, 'training_step': self.training_step})

        self.stats['added'] += 1
        self.update_now_len()

        # Периодическое логирование статуса
        if self.stats['added'] % 10000 == 0:
            fill_ratio = self.now_len / self.max_len
            self.logger.info(f"Buffer status: {self.now_len:,}/{self.max_len:,} ({fill_ratio:.1%})",
                             extra={'agent_type': self.agent_type, 'training_step': self.training_step})

    def extend_buffer(self, state, other):
        size = len(other)
        next_idx = self.next_idx + size

        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

        self.stats['added'] += size
        self.update_now_len()

    def extend_buffer_from_list(self, trajectory_list):
        if self.if_on_policy:
            state = np.array([item[0] for item in trajectory_list], dtype=np.float32)
            other = np.array([item[1] for item in trajectory_list], dtype=np.float32)
        else:
            state = torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float32)
            other = torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float32)

        self.extend_buffer(state, other)

        self.logger.debug(f"Added {len(trajectory_list)} trajectories to buffer",
                          extra={'agent_type': self.agent_type, 'training_step': self.training_step})

    def sample_batch(self, batch_size) -> tuple:
        if self.now_len < batch_size:
            self.logger.error(f"Insufficient data: {self.now_len} < {batch_size}",
                              extra={'agent_type': self.agent_type, 'training_step': self.training_step})
            raise ValueError(f"Insufficient data: {self.now_len} < {batch_size}")

        indices = rd.randint(self.now_len - 1, size=batch_size)
        other = self.buf_other[indices]

        self.stats['sampled'] += 1

        if self.stats['sampled'] % 100 == 0:
            self.logger.debug(f"Sampled {self.stats['sampled']} batches",
                              extra={'agent_type': self.agent_type, 'training_step': self.training_step})

        return (other[:, 0:1],
                other[:, 1:2],
                other[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])

    def sample_all(self) -> tuple:
        all_state = torch.as_tensor(self.buf_state[:self.now_len], device=self.device)
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)

        self.logger.info(f"Sampling all {self.now_len} experiences",
                         extra={'agent_type': self.agent_type, 'training_step': self.training_step})

        return (all_other[:, 0],
                all_other[:, 1],
                all_other[:, 2:2 + self.action_dim],
                all_other[:, 2 + self.action_dim:],
                all_state,)

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer(self):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False

        self.logger.info(f"Buffer emptied. Stats: added={self.stats['added']}, sampled={self.stats['sampled']}",
                         extra={'agent_type': self.agent_type, 'training_step': self.training_step})

        # Сброс статистики
        self.stats = {'added': 0, 'sampled': 0, 'overflows': 0}


# ============================================================================
# УТИЛИТЫ ДЛЯ ЛОГИРОВАНИЯ
# ============================================================================

def log_experiment_start(env_name, agent_type, hyperparams):
    """Логирование начала эксперимента"""
    logger.info("=" * 80, extra={'agent_type': 'Experiment', 'training_step': 0})
    logger.info(f"STARTING NEW EXPERIMENT", extra={'agent_type': 'Experiment', 'training_step': 0})
    logger.info(f"Environment: {env_name}", extra={'agent_type': 'Experiment', 'training_step': 0})
    logger.info(f"Agent: {agent_type}", extra={'agent_type': 'Experiment', 'training_step': 0})
    logger.info(f"Hyperparameters:", extra={'agent_type': 'Experiment', 'training_step': 0})
    for key, value in hyperparams.items():
        logger.info(f"  {key}: {value}", extra={'agent_type': 'Experiment', 'training_step': 0})
    logger.info("=" * 80, extra={'agent_type': 'Experiment', 'training_step': 0})


def log_episode_result(episode, total_reward, episode_length, avg_reward_100):
    """Логирование результата эпизода"""
    logger.info(f"Episode {episode:4d} | Reward: {total_reward:8.2f} | "
                f"Length: {episode_length:4d} | Avg100: {avg_reward_100:8.2f}",
                extra={'agent_type': 'Training', 'training_step': episode})


def log_training_summary(total_steps, total_episodes, best_reward, avg_reward):
    """Логирование сводки по обучению"""
    logger.info("=" * 80, extra={'agent_type': 'Summary', 'training_step': total_steps})
    logger.info("TRAINING SUMMARY", extra={'agent_type': 'Summary', 'training_step': total_steps})
    logger.info(f"Total steps: {total_steps:,}", extra={'agent_type': 'Summary', 'training_step': total_steps})
    logger.info(f"Total episodes: {total_episodes}", extra={'agent_type': 'Summary', 'training_step': total_steps})
    logger.info(f"Best reward: {best_reward:.2f}", extra={'agent_type': 'Summary', 'training_step': total_steps})
    logger.info(f"Average reward: {avg_reward:.2f}", extra={'agent_type': 'Summary', 'training_step': total_steps})
    logger.info("=" * 80, extra={'agent_type': 'Summary', 'training_step': total_steps})


def set_log_level(level):
    """Изменение уровня логирования во время выполнения"""
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)

    level_name = logging.getLevelName(level)
    logger.info(f"Log level changed to {level_name}",
                extra={'agent_type': 'System', 'training_step': 0})


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    # Демонстрация работы системы логирования
    print("Testing RL logging system...")

    # Пример начала эксперимента
    hyperparams = {
        'learning_rate': 1e-3,
        'batch_size': 64,
        'gamma': 0.99,
        'tau': 0.005
    }

    log_experiment_start("CartPole-v1", "DQN", hyperparams)

    # Создание и инициализация агента
    agent = AgentDQN()
    agent.init(net_dim=64, state_dim=4, action_dim=2, learning_rate=1e-3)

    # Имитация нескольких шагов обучения
    for step in range(1, 11):
        agent.training_step = step * 100

        # Логирование прогресса
        metrics = {
            'loss': np.random.rand() * 0.1,
            'reward': np.random.rand() * 100,
            'epsilon': max(0.1, 0.25 - step * 0.02)
        }
        agent.log_training_progress(metrics)

    # Имитация эпизодов
    for episode in range(1, 6):
        total_reward = np.random.rand() * 200
        episode_length = np.random.randint(50, 200)
        avg_reward_100 = np.random.rand() * 150

        log_episode_result(episode, total_reward, episode_length, avg_reward_100)

    # Завершение
    log_training_summary(
        total_steps=1000,
        total_episodes=5,
        best_reward=180.5,
        avg_reward=120.3
    )

    print("Logging test completed. Check logs/ directory for output.")