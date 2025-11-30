from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


# class EpisodeTimeseriesResult:
#     def __init__(self):
#         self.mastery: List[float] = []
#         self.recency: List[float] = []
#         self.i_type: List[int] = []
#         self.rule_id: List[int] = []
#         self.reward: List[float] = []

#     def __repr__(self):
#         out = ''
#         out += 'mastery\n'
#         out += str(self.mastery) + '\n'
#         out += 'recency\n'
#         out += str(self.recency) + '\n'
#         out += 'i_type\n'
#         out += str(self.i_type) + '\n'
#         out += 'rule_id\n'
#         out += str(self.rule_id) + '\n'
#         out += 'reward\n'
#         out += str(self.reward) + '\n'
#         return out
    
#     def plot(self):
#         # This method plots the results from a single episode using
#         # a 2x2 subplot layout to avoid scale conflicts between series.
#         # Subplots: mastery, recency, cumulative reward, and i_type counts.

#         # x-axis over steps
#         t = np.arange(len(self.mastery))

#         # Prepare sliding window counts for i_types: teach(1), quiz(2), review(3)
#         i_type = np.array(self.i_type, dtype=int) if len(self.i_type) else np.array([], dtype=int)
#         # window size ~10% of episode length, at least 1
#         win_size = max(1, len(t) // 10)
#         window = np.ones(win_size, dtype=float)
#         if i_type.size >= 1 and window.size >= 1 and i_type.size >= window.size:
#             teach = np.convolve(i_type == 1, window, mode='valid')
#             quiz = np.convolve(i_type == 2, window, mode='valid')
#             review = np.convolve(i_type == 3, window, mode='valid')
#             t_short = t[len(window) - 1:]
#         else:
#             teach = np.array([])
#             quiz = np.array([])
#             review = np.array([])
#             t_short = np.array([])

#         fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

#         # Mastery subplot
#         ax = axes[0, 0]
#         ax.plot(t, self.mastery, color='C0', label='mastery')
#         ax.set_title('Average Mastery')
#         ax.set_ylabel('mastery')
#         ax.grid(True, alpha=0.3)
 
#         # Recency subplot
#         ax = axes[0, 1]
#         ax.plot(t, self.recency, color='C1', label='recency')
#         ax.set_title('Average Recency')
#         ax.grid(True, alpha=0.3)

#         # Cumulative reward subplot
#         ax = axes[1, 0]
#         ax.plot(t, self.reward, color='C2', label='cum_reward')
#         ax.set_title('Cumulative Reward')
#         ax.set_xlabel('step')
#         ax.set_ylabel('reward')
#         ax.grid(True, alpha=0.3)

#         # Instruction types counts subplot
#         ax = axes[1, 1]
#         if t_short.size > 0:
#             ax.plot(t_short, teach, label='teach', color='C3')
#             ax.plot(t_short, quiz, label='quiz', color='C4')
#             ax.plot(t_short, review, label='review', color='C5')
#         ax.set_title(f'i_type counts (window={win_size})')
#         ax.set_xlabel('step')
#         ax.set_ylabel('count')
#         ax.grid(True, alpha=0.3)
#         ax.legend()

#         fig.suptitle('Results in One Episode', fontsize=14)
#         fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.show()
        

class Trainer:
    def __init__(self, env, tutor):
        self.env = env
        self.tutor = tutor

    def run_episode(
        self,
        *,
        max_steps: Optional[int] = None,
        on_step = None,
    ):

        # reset 
        obs, _ = self.env.reset()
        total_reward = 0.0
        steps = 0

        step_cap = max_steps if max_steps is not None else getattr(self.env, "max_steps", None)

        terminated = False
        truncated = False
        last_info: Dict[str, Any] = {}

        # loop step
        s = 0
        while True:
            s += 1
            action = int(self.tutor.get_action(obs))
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # Update tutor with transition
            self.tutor.update(obs, action, float(reward), bool(terminated), next_obs)

            total_reward += float(reward)
            steps += 1
            last_info = info

            if on_step is not None:
                try:
                    i_type, rule_id = self.env._decode_action(action)
                    on_step(steps, obs, i_type, rule_id, total_reward)
                except Exception:
                    # Callback errors should not break training
                    pass

            if terminated or truncated:
                break
            if step_cap is not None and steps >= step_cap:
                # Respect explicit cap even if env didn't truncate
                truncated = True
                break

            obs = next_obs

        return EpisodeResult(
            total_reward=total_reward,
            steps=steps,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=last_info if isinstance(last_info, dict) else {},
        )

    def train(
        self,
        n_episodes: int,
        *,
        max_steps: Optional[int] = None,
        decay_epsilon_each_episode: bool = True,
        on_step = None,
        on_episode_begin = None,
    ) -> List[EpisodeResult]:
        """Run multiple episodes and return per-episode results.
        """
        results: List[EpisodeResult] = []
        for ep in range(int(n_episodes)):
            if on_episode_begin is not None:
                try:
                    on_episode_begin()
                except Exception:
                    pass

            result = self.run_episode(max_steps=max_steps, on_step=on_step)
            results.append(result)

            if decay_epsilon_each_episode and hasattr(self.tutor, "decay_epsilon"):
                self.tutor.decay_epsilon()

        return results
