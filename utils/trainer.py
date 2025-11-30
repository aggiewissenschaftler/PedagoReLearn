from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np

import utils.data_util as data_util
from utils.data_util import EpisodeTimeseriesResult


"""
Episode flow:
0. Reset env and (optionally) tutor state
1. Get state from env
2. Tutor selects an action for the state
3. Env steps with action → (next_state, reward, terminated, truncated, info)
4. Tutor updates from transition
5. Repeat until terminated or truncated
"""
@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class Trainer:
    def __init__(self, env, tutor):
        self.env = env
        self.tutor = tutor

    def run_episode(
        self,
        *,
        max_steps: Optional[int] = None,
        on_step: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ):# EpisodeResult, EpisodeTimeseriesResult:
        """Run a single episode and update the tutor on the fly.

        Args:
            max_steps: Optional hard cap for steps in this episode.
            on_step: Optional callback called with (t, step_info) each step.

        Returns:
            EpisodeResult with summary metrics.
        """
        # reset 
        obs, _ = self.env.reset()
        total_reward = 0.0
        steps = 0

        step_cap = max_steps if max_steps is not None else getattr(self.env, "max_steps", None)

        terminated = False
        truncated = False
        last_info: Dict[str, Any] = {}

        tsResult = EpisodeTimeseriesResult()

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
                    on_step(steps, {
                        "obs": obs,
                        "action": action,
                        "reward": reward,
                        "next_obs": next_obs,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info,
                    })
                except Exception:
                    # Callback errors should not break training
                    pass

            
            # at each step, save
            # 1. average mastery
            # 2. average recency
            # i_type, rule_id
            # reward
            tsResult.mastery.append(np.mean(obs['mastery']))
            tsResult.recency.append(np.mean(obs['recency']))
            i_type, rule_id = self.env._decode_action(action)
            i_type = ['noop','teach','quiz','review'].index(i_type)
            tsResult.i_type.append(i_type)
            tsResult.rule_id.append(rule_id)
            tsResult.reward.append(total_reward)

            

            if terminated or truncated:
                break
            if step_cap is not None and steps >= step_cap:
                # Respect explicit cap even if env didn't truncate
                truncated = True
                break

            obs = next_obs

        print(f"steps {steps} terminated {terminated} truncated {truncated} ")

        # save per-episode results in data_util
        data_util.append_tsresult(tsResult)

        return EpisodeResult(
            total_reward=total_reward,
            steps=steps,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=last_info if isinstance(last_info, dict) else {},
        ), tsResult

    def train(
        self,
        n_episodes: int,
        *,
        max_steps: Optional[int] = None,
        decay_epsilon_each_episode: bool = True,
        on_episode_end: Optional[Callable[[int, EpisodeResult], None]] = None,
    ) -> List[EpisodeResult]:
        """Run multiple episodes and return per-episode results.

        Args:
            n_episodes: Number of episodes to run.
            max_steps: Optional hard cap for steps in each episode.
            decay_epsilon_each_episode: If True, calls tutor.decay_epsilon().
            on_episode_end: Optional callback invoked with (episode_idx, result).
        """
        results: List[EpisodeResult] = []
        # full_results: List[EpisodeTimeseriesResult] = []
        for ep in range(int(n_episodes)):
            result, tsResult = self.run_episode(max_steps=max_steps)
            results.append(result)
            # full_results.append(tsResult)
            if ep%10 == 0:
                print(ep, end='='*30)
                tsResult.plot()

            if decay_epsilon_each_episode and hasattr(self.tutor, "decay_epsilon"):
                try:
                    self.tutor.decay_epsilon()
                except Exception:
                    pass

            if on_episode_end is not None:
                try:
                    on_episode_end(ep, result)
                except Exception:
                    pass

        # with open('results\\'+datetime.strftime(datetime.now(),"%d%m%Y-%H%M%S.pkl"), 'wb') as f:
        #     pickle.dump(full_results,f)

        # save full timeseries data
        data_util.save_tsresult()
        return results
