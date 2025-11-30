from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass
import pickle
import numpy as np
import matplotlib.pyplot as plt

class EpisodeTimeseriesResult:
    def __init__(self):
        self.mastery: List[float] = []
        self.recency: List[float] = []
        self.i_type: List[int] = []
        self.rule_id: List[int] = []
        self.reward: List[float] = []

DATA: List[EpisodeTimeseriesResult] = []

def append_new_EpisodeResult():
    DATA.append(EpisodeTimeseriesResult())

def append_stepResult(step, obs, i_type, rule_id, reward):
    tsResult:EpisodeTimeseriesResult = DATA[-1]

    tsResult.mastery.append(np.mean(obs['mastery']))
    tsResult.recency.append(np.mean(obs['recency']))
    i_id = ['noop','teach','quiz','review'].index(i_type)
    tsResult.i_type.append(i_id)
    tsResult.rule_id.append(rule_id)
    tsResult.reward.append(reward)
    
def append_episodeResult(result: EpisodeTimeseriesResult):
    DATA.append(result)

def save_n_reset(dscr=''):
    fname = 'ts_results\\'+dscr+datetime.strftime(datetime.now(),"%d%m%Y-%H%M%S.pkl")
    with open(fname, 'wb') as f:
        pickle.dump(DATA,f)
    # print("Full Episodes results saved to "+fname)
    
    DATA.clear()


## ==== old ====




# class EpisodeTimeseriesResult_old:
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
        
