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
        self.cri = self.cri_optim = self.Cri = None  # self.Cri is the class of cri
        self.act = self.act_optim = self.Act = None  # self.Act is the class of cri
        self.cri_target = self.if_use_cri_target = None
        self.act_target = self.if_use_act_target = None
        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.debug(f"Создан экземпляр {self.__class__.__name__}")

        # Статистика обучения
        self.train_step = 0
        self.episode_rewards = []
        self.recent_rewards = []

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        """Инициализация агента с логированием критических параметров"""
        try:
            self.logger.info("Начало инициализации агента...")

            # Определение устройства (CPU/GPU)
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"Обнаружена GPU: {gpu_name}")
                self.logger.info(f"CUDA версия: {torch.version.cuda}")
            else:
                self.device = torch.device("cpu")
                self.logger.warning("GPU не обнаружена, используется CPU")

            self.action_dim = action_dim
            self.logger.info(f"Параметры агента: action_dim={action_dim}, "
                             f"net_dim={net_dim}, state_dim={state_dim}, lr={learning_rate}")

            # Инициализация критической сети
            self.logger.debug(f"Инициализация критической сети: {self.Cri.__name__}")
            self.cri = self.Cri(net_dim, state_dim, action_dim).to(self.device)

            # Инициализация акторской сети (если есть)
            if self.Act is not None:
                self.logger.debug(f"Инициализация акторской сети: {self.Act.__name__}")
                self.act = self.Act(net_dim, state_dim, action_dim).to(self.device)
            else:
                self.act = self.cri
                self.logger.debug("Акторская сеть совпадает с критической")

            # Создание целевых сетей (если требуется)
            if self.if_use_cri_target:
                self.cri_target = deepcopy(self.cri)
                self.logger.debug("Создана целевая критическая сеть")
            else:
                self.cri_target = self.cri
                self.logger.debug("Целевая критическая сеть не используется")

            if self.if_use_act_target:
                self.act_target = deepcopy(self.act)
                self.logger.debug("Создана целевая акторская сеть")
            else:
                self.act_target = self.act
                self.logger.debug("Целевая акторская сеть не используется")

            # Подсчет параметров сетей
            cri_params = sum(p.numel() for p in self.cri.parameters())
            self.logger.info(f"Критическая сеть: {cri_params:,} параметров")

            if self.Act is not None:
                act_params = sum(p.numel() for p in self.act.parameters())
                self.logger.info(f"Акторская сеть: {act_params:,} параметров")

            # Инициализация оптимизаторов
            self.logger.debug("Инициализация оптимизаторов...")
            self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)

            if self.Act is not None:
                self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate)
            else:
                self.act_optim = self.cri

            # Очистка временных атрибутов
            del self.Cri, self.Act, self.if_use_cri_target, self.if_use_act_target

            # Проверка памяти GPU
            if self.device.type == 'cuda':
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                allocated_memory = torch.cuda.memory_allocated(0) / 1e9
                self.logger.info(f"GPU память: {allocated_memory:.2f} GB / {total_memory:.2f} GB выделено")

            self.logger.info("Инициализация агента успешно завершена ✓")

        except Exception as e:
            self.logger.error(f"Ошибка при инициализации агента: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def select_action(self, state) -> np.ndarray:
        pass  # sample form an action distribution

    def explore_env(self, env, target_step, reward_scale, gamma) -> list:
        trajectory_list = list()

        state = self.state
        for _ in range(target_step):
            action = self.select_action(state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            trajectory_list.append((state, other))

            state = env.reset() if done else next_s
        self.state = state
        return trajectory_list

    def optim_update(self, optimizer, objective, network_name="network"):
        """Обновление оптимизатора с логированием градиентов"""
        try:
            optimizer.zero_grad()
            objective.backward()

            # Логирование градиентов перед обновлением
            total_norm = 0.0
            max_grad = -float('inf')
            min_grad = float('inf')

            for param in optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    max_grad = max(max_grad, param.grad.data.max().item())
                    min_grad = min(min_grad, param.grad.data.min().item())

            grad_norm = total_norm ** 0.5
            self.logger.debug(f"Градиенты {network_name}: norm={grad_norm:.6f}, "
                              f"max={max_grad:.6f}, min={min_grad:.6f}")

            optimizer.step()
            return grad_norm

        except Exception as e:
            self.logger.error(f"Ошибка при обновлении {network_name}: {str(e)}")
            raise

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1 - tau))


class AgentDQN(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy
        self.if_use_cri_target = True
        self.Cri = QNet
        self.logger.info(f"AgentDQN создан: explore_rate={self.explore_rate}")

    def select_action(self, state) -> int:  # for discrete action space
        try:
            if rd.rand() < self.explore_rate:  # epsilon-greedy
                a_int = rd.randint(self.action_dim)  # choosing action randomly
                self.logger.debug(f"Случайное действие (epsilon-greedy): {a_int}")
            else:
                states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
                action = self.act(states)[0]
                a_int = action.argmax(dim=0).detach().cpu().numpy()
                q_values = action.detach().cpu().numpy()
                self.logger.debug(f"Жадное действие: {a_int}, Q-значения: {q_values}")
            return a_int
        except Exception as e:
            self.logger.error(f"Ошибка при выборе действия: {str(e)}")
            return rd.randint(self.action_dim)

    def explore_env(self, env, target_step, reward_scale, gamma) -> list:
        trajectory_list = list()

        state = self.state
        for _ in range(target_step):
            action = self.select_action(state)  # assert isinstance(action, int)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, action)
            trajectory_list.append((state, other))

            state = env.reset() if done else next_s
        self.state = state
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        self.logger.info(f"Начало обновления сети: buffer={buffer.now_len}, "
                         f"batch_size={batch_size}, repeat_times={repeat_times}")

        buffer.update_now_len()

        if buffer.now_len < batch_size:
            self.logger.warning(f"Недостаточно данных в буфере: {buffer.now_len} < {batch_size}")
            return 0.0, 0.0

        obj_critic = q_value = None
        total_critic_loss = 0.0
        total_q_value = 0.0
        update_count = 0

        num_updates = int(buffer.now_len / batch_size * repeat_times)
        self.logger.info(f"Запланировано обновлений: {num_updates}")

        for i in range(num_updates):
            obj_critic, q_value = self.get_obj_critic(buffer, batch_size)

            # Логирование перед обновлением
            current_loss = obj_critic.item()
            current_q = q_value.mean().item()
            self.logger.debug(f"Итерация {i + 1}/{num_updates}: loss={current_loss:.6f}, "
                              f"avg_Q={current_q:.6f}")

            grad_norm = self.optim_update(self.cri_optim, obj_critic, "critic")
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            total_critic_loss += current_loss
            total_q_value += current_q
            update_count += 1

            self.train_step += 1
            if self.train_step % 100 == 0:
                self.logger.info(f"Шаг обучения {self.train_step}: "
                                 f"средний loss={total_critic_loss / update_count:.6f}, "
                                 f"средний Q={total_q_value / update_count:.6f}")

        avg_loss = total_critic_loss / max(update_count, 1)
        avg_q = total_q_value / max(update_count, 1)

        self.logger.info(f"Обновление завершено: средний loss={avg_loss:.6f}, "
                         f"средний Q={avg_q:.6f}, обновлений={update_count}")

        return avg_loss, avg_q

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value


class AgentDoubleDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.softMax = torch.nn.Softmax(dim=1)
        self.Cri = QNetTwin
        self.logger.info(f"AgentDoubleDQN создан")

    def select_action(self, state) -> int:  # for discrete action space
        try:
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
            actions = self.act(states)

            if rd.rand() < self.explore_rate:  # epsilon-greedy
                a_prob = self.softMax(actions)[0].detach().cpu().numpy()
                a_int = rd.choice(self.action_dim, p=a_prob)
                self.logger.debug(f"Взвешенное случайное действие: {a_int}, "
                                  f"вероятности: {a_prob}")
            else:
                action = actions[0]
                a_int = action.argmax(dim=0).detach().cpu().numpy()
                q_values = action.detach().cpu().numpy()
                self.logger.debug(f"Жадное действие (DoubleDQN): {a_int}, "
                                  f"Q-значения: {q_values}")
            return a_int
        except Exception as e:
            self.logger.error(f"Ошибка при выборе действия DoubleDQN: {str(e)}")
            return rd.randint(self.action_dim)

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]

        # Логирование разницы между двумя Q-сетями
        q_diff = torch.abs(q1 - q2).mean().item()
        if q_diff > 1.0:  # Если разница большая
            self.logger.warning(f"Большая разница между Q-сетями: {q_diff:.4f}")

        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, q1


class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.1  # explore noise of action
        self.if_use_cri_target = self.if_use_act_target = True
        self.Act = Actor
        self.Cri = Critic
        self.logger.info(f"AgentDDPG создан: explore_noise={self.explore_noise}")

    def select_action(self, state) -> np.ndarray:
        try:
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
            action = self.act(states)[0]
            noise = torch.randn_like(action) * self.explore_noise
            action = (action + noise).clamp(-1, 1)

            action_np = action.cpu().numpy()
            noise_np = noise.cpu().numpy()

            self.logger.debug(f"Выбор действия DDPG: действие={action_np}, "
                              f"шум={noise_np}, норма шума={np.linalg.norm(noise_np):.4f}")

            return action_np
        except Exception as e:
            self.logger.error(f"Ошибка при выборе действия DDPG: {str(e)}")
            return np.random.uniform(-1, 1, self.action_dim)

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        self.logger.info(f"Начало обновления DDPG: buffer={buffer.now_len}, "
                         f"batch_size={batch_size}")

        buffer.update_now_len()

        if buffer.now_len < batch_size:
            self.logger.warning(f"Недостаточно данных в буфере: {buffer.now_len} < {batch_size}")
            return 0.0, 0.0

        obj_critic = obj_actor = None
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        update_count = 0

        num_updates = int(buffer.now_len / batch_size * repeat_times)
        self.logger.info(f"Запланировано обновлений DDPG: {num_updates}")

        for i in range(num_updates):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            critic_loss = obj_critic.item()

            # Обновление критической сети
            critic_grad_norm = self.optim_update(self.cri_optim, obj_critic, "critic")
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            # Обновление акторской сети
            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri(state, action_pg).mean()
            actor_loss = obj_actor.item()

            actor_grad_norm = self.optim_update(self.act_optim, obj_actor, "actor")
            self.soft_update(self.act_target, self.act, soft_update_tau)

            # Логирование
            self.logger.debug(f"DDPG итерация {i + 1}/{num_updates}: "
                              f"critic_loss={critic_loss:.6f}, actor_loss={actor_loss:.6f}, "
                              f"critic_grad_norm={critic_grad_norm:.6f}, "
                              f"actor_grad_norm={actor_grad_norm:.6f}")

            total_critic_loss += critic_loss
            total_actor_loss += actor_loss
            update_count += 1

            self.train_step += 1
            if self.train_step % 50 == 0:
                self.logger.info(f"DDPG шаг {self.train_step}: "
                                 f"critic_loss={total_critic_loss / update_count:.6f}, "
                                 f"actor_loss={total_actor_loss / update_count:.6f}")

        avg_critic_loss = total_critic_loss / max(update_count, 1)
        avg_actor_loss = total_actor_loss / max(update_count, 1)

        self.logger.info(f"DDPG обновление завершено: "
                         f"avg_critic_loss={avg_critic_loss:.6f}, "
                         f"avg_actor_loss={avg_actor_loss:.6f}, "
                         f"updates={update_count}")

        return avg_actor_loss, avg_critic_loss

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state


class AgentTD3(AgentDDPG):
    def __init__(self):
        super().__init__()
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency
        self.Cri = CriticTwin
        self.logger.info(f"AgentTD3 создан: policy_noise={self.policy_noise}, update_freq={self.update_freq}")

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        self.logger.info(f"Начало обновления TD3: buffer={buffer.now_len}")

        buffer.update_now_len()

        if buffer.now_len < batch_size:
            self.logger.warning(f"Недостаточно данных в буфере: {buffer.now_len} < {batch_size}")
            return 0.0, 0.0

        obj_critic = obj_actor = None
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        update_count = 0

        num_updates = int(buffer.now_len / batch_size * repeat_times)
        self.logger.info(f"Запланировано обновлений TD3: {num_updates}")

        for update_c in range(num_updates):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            critic_loss = obj_critic.item()

            # Обновление критической сети
            self.optim_update(self.cri_optim, obj_critic, "critic_twin")

            # Обновление акторской сети
            action_pg = self.act(state)
            obj_actor = -self.cri_target(state, action_pg).mean()
            actor_loss = obj_actor.item()

            self.optim_update(self.act_optim, obj_actor, "actor")

            # Задержанное обновление
            if update_c % self.update_freq == 0:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
                self.soft_update(self.act_target, self.act, soft_update_tau)
                self.logger.debug(f"TD3 задержанное обновление на итерации {update_c}")

            # Логирование статистики
            if update_c % 10 == 0:
                self.logger.debug(f"TD3 итерация {update_c}: "
                                  f"critic_loss={critic_loss:.6f}, actor_loss={actor_loss:.6f}")

            total_critic_loss += critic_loss
            total_actor_loss += actor_loss
            update_count += 1

            self.train_step += 1

        avg_critic_loss = total_critic_loss / max(update_count, 1) / 2  # /2 для двух критиков
        avg_actor_loss = total_actor_loss / max(update_count, 1)

        self.logger.info(f"TD3 обновление завершено: "
                         f"avg_critic_loss={avg_critic_loss:.6f}, "
                         f"avg_actor_loss={avg_actor_loss:.6f}, "
                         f"updates={update_count}")

        return avg_critic_loss, avg_actor_loss

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)

        # Логирование различий между критиками
        q_diff = torch.abs(q1 - q2).mean().item()
        if q_diff > 0.5:
            self.logger.warning(f"Большая разница между критиками TD3: {q_diff:.4f}")

        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, state


class AgentSAC(AgentBase):
    def __init__(self):
        super().__init__()
        self.if_use_cri_target = True
        self.Act = ActorSAC
        self.Cri = CriticTwin
        self.logger.info(f"AgentSAC создан")

    def select_action(self, state) -> np.ndarray:
        try:
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
            action = self.act.get_action(states)[0]
            action_np = action.cpu().numpy()

            self.logger.debug(f"Выбор действия SAC: действие={action_np}")
            return action_np
        except Exception as e:
            self.logger.error(f"Ошибка при выборе действия SAC: {str(e)}")
            return np.random.uniform(-1, 1, self.action_dim)

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        self.logger.info(f"Начало обновления SAC: buffer={buffer.now_len}")

        buffer.update_now_len()

        if buffer.now_len < batch_size:
            self.logger.warning(f"Недостаточно данных в буфере: {buffer.now_len} < {batch_size}")
            return 0.0, 0.0, -1.0

        log_alpha = self.act.log_alpha
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        alpha_values = []
        update_count = 0

        num_updates = int(buffer.now_len / batch_size * repeat_times)
        self.logger.info(f"Запланировано обновлений SAC: {num_updates}")

        for update_c in range(num_updates):
            obj_critic, state = self.get_obj_critic(buffer, batch_size, log_alpha.exp())
            critic_loss = obj_critic.item()

            # Обновление критической сети
            self.optim_update(self.cri_optim, obj_critic, "critic_sac")
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            # Обновление акторской сети
            action_pg, logprob = self.act.get_action_logprob(state)
            obj_actor = (-torch.min(*self.cri_target.get_q1_q2(state, action_pg)).mean()
                         + logprob.mean() * log_alpha.exp().detach()
                         + self.act.get_obj_alpha(logprob))
            actor_loss = obj_actor.item()

            self.optim_update(self.act_optim, obj_actor, "actor_sac")

            # Логирование температуры (alpha)
            current_alpha = log_alpha.exp().item()
            alpha_values.append(current_alpha)

            if update_c % 20 == 0:
                self.logger.debug(f"SAC итерация {update_c}: "
                                  f"critic_loss={critic_loss:.6f}, actor_loss={actor_loss:.6f}, "
                                  f"alpha={current_alpha:.6f}, "
                                  f"avg_logprob={logprob.mean().item():.6f}")

            total_critic_loss += critic_loss
            total_actor_loss += actor_loss
            update_count += 1

            self.train_step += 1

        avg_critic_loss = total_critic_loss / max(update_count, 1) / 2
        avg_actor_loss = total_actor_loss / max(update_count, 1)
        avg_alpha = np.mean(alpha_values) if alpha_values else -1.0

        self.logger.info(f"SAC обновление завершено: "
                         f"avg_critic_loss={avg_critic_loss:.6f}, "
                         f"avg_actor_loss={avg_actor_loss:.6f}, "
                         f"avg_alpha={avg_alpha:.6f}, "
                         f"updates={update_count}")

        return avg_critic_loss, avg_actor_loss, avg_alpha

    def get_obj_critic(self, buffer, batch_size, alpha) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a, next_logprob = self.act.get_action_logprob(next_s)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
            q_label = reward + mask * (next_q + next_logprob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)  # twin critics

        # Логирование энтропийного бонуса
        entropy_bonus = (next_logprob * alpha).mean().item()
        self.logger.debug(f"SAC энтропийный бонус: {entropy_bonus:.6f}")

        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, state


class AgentPPO(AgentBase):
    def __init__(self):
        super().__init__()
        self.if_on_policy = True
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.02
        self.Act = ActorPPO
        self.Cri = CriticAdv
        self.logger.info(f"AgentPPO создан: ratio_clip={self.ratio_clip}, lambda_entropy={self.lambda_entropy}")

    def select_action(self, state):
        try:
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
            actions, noises = self.act.get_action(states)

            action_np = actions[0].detach().cpu().numpy()
            noise_np = noises[0].detach().cpu().numpy()

            self.logger.debug(f"Выбор действия PPO: действие={action_np}, шум={noise_np}")
            return action_np, noise_np
        except Exception as e:
            self.logger.error(f"Ошибка при выборе действия PPO: {str(e)}")
            return np.zeros(self.action_dim), np.zeros(self.action_dim)

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = list()

        state = self.state
        for _ in range(target_step):
            action, noise = self.select_action(state)
            next_s, reward, done, _ = env.step(np.tanh(action))
            other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
            trajectory_list.append((state, other))

            state = env.reset() if done else next_s
        self.state = state
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        self.logger.info(f"Начало обновления PPO: buffer={buffer.now_len}")

        buffer.update_now_len()
        buf_len, buf_state, buf_action, buf_r_sum, buf_logprob, buf_advantage = self.prepare_buffer(buffer)
        buffer.empty_buffer()

        self.logger.info(f"PPO подготовка данных: buf_len={buf_len}, "
                         f"batch_size={batch_size}, repeat_times={repeat_times}")

        '''PPO: Surrogate objective of Trust Region'''
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_entropy = 0.0
        total_ratio = 0.0
        update_count = 0

        num_updates = int(buf_len / batch_size * repeat_times)
        self.logger.info(f"Запланировано эпох PPO: {num_updates}")

        for epoch in range(num_updates):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            advantage = buf_advantage[indices]
            old_logprob = buf_logprob[indices]

            new_logprob, obj_entropy = self.act.get_new_logprob_entropy(state, action)
            ratio = (new_logprob - old_logprob.detach()).exp()

            # Логирование клиппинга
            clip_fraction = ((ratio < 1 - self.ratio_clip) | (ratio > 1 + self.ratio_clip)).float().mean().item()
            if clip_fraction > 0.3:  # Если много клиппинга
                self.logger.warning(f"Высокий процент клиппинга PPO: {clip_fraction:.2%}")

            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            # Обновление актора
            actor_grad_norm = self.optim_update(self.act_optim, obj_actor, "actor_ppo")

            value = self.cri(state).squeeze(1)
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

            # Обновление критика
            critic_grad_norm = self.optim_update(self.cri_optim, obj_critic, "critic_ppo")

            if self.cri_target is not self.cri:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            # Сбор статистики
            total_critic_loss += obj_critic.item()
            total_actor_loss += obj_actor.item()
            total_entropy += obj_entropy.item()
            total_ratio += ratio.mean().item()
            update_count += 1

            if epoch % 10 == 0:
                self.logger.debug(f"PPO эпоха {epoch}: "
                                  f"critic_loss={obj_critic.item():.6f}, "
                                  f"actor_loss={obj_actor.item():.6f}, "
                                  f"entropy={obj_entropy.item():.6f}, "
                                  f"clip_fraction={clip_fraction:.2%}")

            self.train_step += 1

        # Итоговая статистика
        avg_critic_loss = total_critic_loss / max(update_count, 1)
        avg_actor_loss = total_actor_loss / max(update_count, 1)
        avg_entropy = total_entropy / max(update_count, 1)
        avg_ratio = total_ratio / max(update_count, 1)

        self.logger.info(f"PPO обновление завершено: "
                         f"avg_critic_loss={avg_critic_loss:.6f}, "
                         f"avg_actor_loss={avg_actor_loss:.6f}, "
                         f"avg_entropy={avg_entropy:.6f}, "
                         f"avg_ratio={avg_ratio:.4f}, "
                         f"epochs={update_count}")

        return avg_critic_loss, avg_actor_loss, avg_ratio

    def prepare_buffer(self, buffer):
        buf_len = buffer.now_len
        with torch.no_grad():  # compute reverse reward
            reward, mask, action, a_noise, state = buffer.sample_all()

            # print(';', [t.shape for t in (reward, mask, action, a_noise, state)])
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            value = torch.cat([self.cri_target(state[i:i + bs]) for i in range(0, state.size(0), bs)], dim=0)
            logprob = self.act.get_old_logprob(action, a_noise)

            pre_state = torch.as_tensor((self.state,), dtype=torch.float32, device=self.device)
            pre_r_sum = self.cri(pre_state).detach()
            r_sum, advantage = self.get_reward_sum(buf_len, reward, mask, value, pre_r_sum)

            # Логирование преимуществ
            adv_mean = advantage.mean().item()
            adv_std = advantage.std().item()
            self.logger.info(f"PPO подготовка буфера: преимущество mean={adv_mean:.4f}, std={adv_std:.4f}")

        return buf_len, state, action, r_sum, logprob, advantage

    def get_reward_sum(self, buf_len, reward, mask, value, pre_r_sum) -> (torch.Tensor, torch.Tensor):
        r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        for i in range(buf_len - 1, -1, -1):
            r_sum[i] = reward[i] + mask[i] * pre_r_sum
            pre_r_sum = r_sum[i]
        advantage = r_sum - (mask * value.squeeze(1))
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
        return r_sum, advantage


class AgentDiscretePPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.Act = ActorDiscretePPO
        self.logger.info(f"AgentDiscretePPO создан")

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = list()

        state = self.state
        for _ in range(target_step):
            a_int, a_prob = self.select_action(state)
            next_s, reward, done, _ = env.step(a_int)
            other = (reward * reward_scale, 0.0 if done else gamma, a_int, *a_prob)
            trajectory_list.append((state, other))

            state = env.reset() if done else next_s
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

            self.logger.info(f"Инициализация ReplayBuffer: max_len={max_len:,}, "
                             f"state_dim={state_dim}, action_dim={action_dim}, "
                             f"if_discrete={if_discrete}, if_on_policy={if_on_policy}")

            if if_on_policy:
                other_dim = 1 + 1 + self.action_dim + action_dim
                # other = (reward, mask, action, a_noise) for continuous action
                # other = (reward, mask, a_int, a_prob) for discrete action
                self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
                self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)
                buffer_size_mb = (self.buf_other.nbytes + self.buf_state.nbytes) / (1024 * 1024)
                self.logger.info(f"On-policy буфер: {buffer_size_mb:.2f} MB")
            else:
                other_dim = 1 + 1 + self.action_dim
                self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)
                self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)

                # Расчет занимаемой памяти
                element_size = self.buf_other.element_size()
                total_elements = self.buf_other.numel() + self.buf_state.numel()
                buffer_size_mb = (element_size * total_elements) / (1024 * 1024)
                self.logger.info(f"Off-policy буфер: {buffer_size_mb:.2f} MB")

            self.logger.info("ReplayBuffer успешно инициализирован ✓")

        except Exception as e:
            logger.error(f"Критическая ошибка при создании ReplayBuffer: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def append_buffer(self, state, other):
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

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

    def extend_buffer_from_list(self, trajectory_list):
        if self.if_on_policy:
            state = np.array([item[0] for item in trajectory_list], dtype=np.float32)
            other = np.array([item[1] for item in trajectory_list], dtype=np.float32)
        else:
            state = torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float32)
            other = torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float32)
        self.extend_buffer(state, other)

    def sample_batch(self, batch_size) -> tuple:  # for off-policy only
        indices = rd.randint(self.now_len - 1, size=batch_size)
        other = self.buf_other[indices]  # reward, mask, action
        return (other[:, 0:1],  # reward
                other[:, 1:2],  # mask = 0.0 if done else gamma
                other[:, 2:],  # action
                self.buf_state[indices],  # state
                self.buf_state[indices + 1])  # next state

    def sample_all(self) -> tuple:  # for on-policy only
        all_state = torch.as_tensor(self.buf_state[:self.now_len], device=self.device)
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],  # reward
                all_other[:, 1],  # mask = 0.0 if done else gamma
                all_other[:, 2:2 + self.action_dim],  # action
                all_other[:, 2 + self.action_dim:],  # action_noise or action_prob
                all_state,)  # state without last_state

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer(self):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False