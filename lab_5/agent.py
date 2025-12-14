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

    # Создаем директорию для логов, если её нет
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Создаем имя файла с timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rl_agent_{timestamp}.log")

    # Настраиваем логгер
    logger = logging.getLogger("RLAgent")
    logger.setLevel(log_level)

    # Очищаем существующие обработчики (на случай перезапуска)
    logger.handlers.clear()

    # Форматтер для логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Обработчик для вывода в файл
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Обработчик для вывода в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Добавляем обработчики к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info("Система логирования инициализирована")
    logger.info(f"Логи будут сохраняться в: {log_file}")
    logger.info(f"Уровень логирования: {logging.getLevelName(log_level)}")
    logger.info("=" * 60)

    return logger


# Инициализируем глобальный логгер
try:
    logger = setup_logging()
    logger.info("Глобальный логгер успешно инициализирован")
except Exception as e:
    print(f"Ошибка при инициализации логирования: {e}")
    # Создаем минимальный логгер в случае ошибки
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

        # Статистика взаимодействия со средой
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
        self.explore_stats = {
            'random_actions': 0,
            'exploit_actions': 0,
            'episode_completions': 0
        }

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        """Инициализация агента"""
        try:
            self.logger.info("Начало инициализации агента...")

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

            self.logger.info(f"Агент инициализирован: device={self.device}, action_dim={action_dim}")

        except Exception as e:
            self.logger.error(f"Ошибка при инициализации агента: {str(e)}")
            raise

    def select_action(self, state) -> np.ndarray:
        pass

    def explore_env(self, env, target_step, reward_scale, gamma) -> list:
        """Взаимодействие со средой с подробным логированием"""
        trajectory_list = []
        episode_started = self.state is None

        self.logger.info(f"Начало исследования: target_step={target_step}, reward_scale={reward_scale:.3f}")

        if self.state is None:
            self.state = env.reset()
            self.logger.debug("Среда сброшена, начальное состояние установлено")

        state = self.state
        step_in_episode = 0
        episode_reward = 0.0

        for step in range(target_step):
            action = self.select_action(state)

            # Выполнение действия в среде
            next_s, reward, done, info = env.step(action)
            scaled_reward = reward * reward_scale
            episode_reward += reward
            step_in_episode += 1
            self.step_count += 1

            # Сохранение траектории
            other = (scaled_reward, 0.0 if done else gamma, *action)
            trajectory_list.append((state, other))

            # Логирование значимых событий
            if step % 100 == 0:
                self.logger.debug(f"Шаг {step}/{target_step}: reward={reward:.3f}, action={action}")

            # Обработка завершения эпизода
            if done:
                self.episode_count += 1
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(step_in_episode)
                self.total_reward += episode_reward
                self.explore_stats['episode_completions'] += 1

                # Логирование завершения эпизода
                self.log_episode_completion(step_in_episode, episode_reward)

                # Сброс для нового эпизода
                state = env.reset()
                episode_reward = 0.0
                step_in_episode = 0
            else:
                state = next_s

        # Сохранение финального состояния
        self.state = state

        # Логирование статистики исследования
        self.log_exploration_summary(target_step, trajectory_list)

        return trajectory_list

    def log_episode_completion(self, episode_length, episode_reward):
        """Логирование завершения эпизода"""
        avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward

        self.logger.info(f"Эпизод {self.episode_count} завершен: "
                         f"длина={episode_length}, награда={episode_reward:.2f}, "
                         f"средняя(10)={avg_reward:.2f}")

        # Дополнительное логирование для особых случаев
        if episode_reward > max(self.episode_rewards[:-1] if len(self.episode_rewards) > 1 else [0]):
            self.logger.info(f"Новый рекорд! Предыдущий лучший: "
                             f"{max(self.episode_rewards[:-1]) if len(self.episode_rewards) > 1 else 0:.2f}")

        if episode_length >= 1000:
            self.logger.warning(f"Длинный эпизод: {episode_length} шагов")

    def log_exploration_summary(self, target_step, trajectory_list):
        """Логирование сводки по исследованию"""
        collected_steps = len(trajectory_list)

        self.logger.info(f"Исследование завершено: "
                         f"собрано {collected_steps}/{target_step} шагов, "
                         f"эпизодов={self.explore_stats['episode_completions']}")

        # Статистика по действиям
        if self.explore_stats['random_actions'] + self.explore_stats['exploit_actions'] > 0:
            random_ratio = self.explore_stats['random_actions'] / (
                    self.explore_stats['random_actions'] + self.explore_stats['exploit_actions'])
            self.logger.debug(f"Статистика действий: случайные={random_ratio:.1%}, "
                              f"жадные={1 - random_ratio:.1%}")

        # Сброс статистики для следующего исследования
        self.explore_stats['random_actions'] = 0
        self.explore_stats['exploit_actions'] = 0
        self.explore_stats['episode_completions'] = 0

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
            self.explore_stats['random_actions'] += 1
        else:
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
            action = self.act(states)[0]
            a_int = action.argmax(dim=0).detach().cpu().numpy()
            self.explore_stats['exploit_actions'] += 1
        return a_int

    def explore_env(self, env, target_step, reward_scale, gamma) -> list:
        trajectory_list = []

        state = self.state if self.state is not None else env.reset()

        self.logger.info(f"DQN исследование: epsilon={self.explore_rate:.3f}")

        for step in range(target_step):
            action = self.select_action(state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, action)
            trajectory_list.append((state, other))

            if done:
                state = env.reset()
            else:
                state = next_s

        self.state = state
        return trajectory_list


class AgentDoubleDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.softMax = torch.nn.Softmax(dim=1)
        self.Cri = QNetTwin

    def select_action(self, state) -> int:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions = self.act(states)

        if rd.rand() < self.explore_rate:
            a_prob = self.softMax(actions)[0].detach().cpu().numpy()
            a_int = rd.choice(self.action_dim, p=a_prob)
            self.explore_stats['random_actions'] += 1
        else:
            action = actions[0]
            a_int = action.argmax(dim=0).detach().cpu().numpy()
            self.explore_stats['exploit_actions'] += 1

        return a_int


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
        noise = torch.randn_like(action) * self.explore_noise
        action = (action + noise).clamp(-1, 1)

        # Логирование уровня шума
        noise_norm = torch.norm(noise).item()
        if noise_norm > 2.0:
            self.logger.warning(f"Высокий уровень шума DDPG: {noise_norm:.3f}")

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

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = []

        state = self.state if self.state is not None else env.reset()

        self.logger.info(f"PPO исследование: on-policy сбор данных")

        for step in range(target_step):
            action, noise = self.select_action(state)
            next_s, reward, done, _ = env.step(np.tanh(action))
            other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
            trajectory_list.append((state, other))

            if done:
                state = env.reset()
            else:
                state = next_s

        self.state = state

        if trajectory_list:
            rewards = [t[1][0] for t in trajectory_list]
            self.logger.debug(f"PPO собрано {len(trajectory_list)} шагов, "
                              f"средняя награда={np.mean(rewards):.3f}")

        return trajectory_list


class AgentDiscretePPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.Act = ActorDiscretePPO

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = []

        state = self.state if self.state is not None else env.reset()

        for step in range(target_step):
            a_int, a_prob = self.select_action(state)
            next_s, reward, done, _ = env.step(a_int)
            other = (reward * reward_scale, 0.0 if done else gamma, a_int, *a_prob)
            trajectory_list.append((state, other))

            if done:
                state = env.reset()
            else:
                state = next_s

        self.state = state
        return trajectory_list


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
            self.logger = logger.getChild("ReplayBuffer")

            if if_on_policy:
                other_dim = 1 + 1 + self.action_dim + action_dim
                self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
                self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)
            else:
                other_dim = 1 + 1 + self.action_dim
                self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)
                self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)

            self.logger.info(f"ReplayBuffer создан: max_len={max_len:,}, "
                             f"on_policy={if_on_policy}")

        except Exception as e:
            logger.error(f"Ошибка при создании ReplayBuffer: {str(e)}")
            raise

    def append_buffer(self, state, other):
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

            if not self.if_on_policy:
                self.logger.debug("Буфер переполнен, начата перезапись")

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

            self.logger.debug(f"Буфер переполнен, добавлено {size} элементов "
                              f"(перезапись: {size - (self.max_len - self.next_idx)})")
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other

        self.next_idx = next_idx
        self.update_now_len()

    def extend_buffer_from_list(self, trajectory_list):
        """Добавление данных из списка траекторий с логированием"""
        if not trajectory_list:
            self.logger.warning("Попытка добавить пустой список траекторий")
            return

        num_samples = len(trajectory_list)

        if self.if_on_policy:
            state = np.array([item[0] for item in trajectory_list], dtype=np.float32)
            other = np.array([item[1] for item in trajectory_list], dtype=np.float32)
        else:
            state = torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float32)
            other = torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float32)

        old_len = self.now_len
        self.extend_buffer(state, other)
        new_len = self.now_len

        self.logger.debug(f"Добавлено {num_samples} траекторий: "
                          f"буфер {old_len} -> {new_len} элементов "
                          f"(заполнение {new_len / self.max_len:.1%})")

        # Предупреждение при почти полном буфере
        if new_len / self.max_len > 0.9 and not self.if_full:
            self.logger.warning(f"Буфер почти полон: {new_len}/{self.max_len} "
                                f"({new_len / self.max_len:.1%})")

    def sample_batch(self, batch_size) -> tuple:
        """Выборка батча с логированием"""
        if self.now_len < batch_size:
            self.logger.error(f"Недостаточно данных для выборки: "
                              f"{self.now_len} < {batch_size}")
            raise ValueError("Недостаточно данных в буфере")

        indices = rd.randint(self.now_len - 1, size=batch_size)
        other = self.buf_other[indices]

        self.logger.debug(f"Выборка батча: размер={batch_size}, "
                          f"доступно={self.now_len}")

        return (other[:, 0:1],
                other[:, 1:2],
                other[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])

    def sample_all(self) -> tuple:
        """Выборка всех данных (для on-policy)"""
        all_state = torch.as_tensor(self.buf_state[:self.now_len], device=self.device)
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)

        self.logger.debug(f"Выборка всех данных: {self.now_len} элементов")

        return (all_other[:, 0],
                all_other[:, 1],
                all_other[:, 2:2 + self.action_dim],
                all_other[:, 2 + self.action_dim:],
                all_state,)

    def update_now_len(self):
        """Обновление текущей длины с логированием"""
        old_len = self.now_len
        self.now_len = self.max_len if self.if_full else self.next_idx

        if self.now_len != old_len:
            self.logger.debug(f"Длина буфера обновлена: {old_len} -> {self.now_len}")

        # Логирование при значительном изменении заполненности
        if self.now_len >= self.max_len * 0.8 and not self.if_full:
            self.logger.info(f"Буфер заполнен на {self.now_len / self.max_len:.1%}")

    def empty_buffer(self):
        """Очистка буфера с логированием"""
        old_len = self.now_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False

        self.logger.info(f"Буфер очищен: было {old_len} элементов")