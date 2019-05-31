import os,sys
import random
import numpy as np
import matplotlib.pyplot as plt
from agents.learning_agent import LearningAgent
from agents.policy import EpsGreedyQPolicy
from envs.gsd import GSD
import ipdb
import copy

if __name__ == '__main__':
    # 報酬関数の種類。G:Grobal Reward、D:Difference Rewards
    reward_funs=[
            "G" , 
            "D"
            ]
    iter_num = 5  # 試行回数
    nb_agents = 10
    nb_episode = 3000   #エピソード数
    mu = 0.5 
    sigma = 1.0

    results = []
    for reward_fun in reward_funs:
        results = []
        for it in range(iter_num):
            grid_env = GSD(mu=mu, sigma=sigma) # grid worldの環境の初期化
            policy = EpsGreedyQPolicy(epsilon=0.1) # 方策の初期化。ここではε-greedy

            agents = []
            for i in range(nb_agents):
                agent = LearningAgent(aid=i, actions=grid_env.actions,  policy=policy) # Q Learning エージェントの初期化
                agents.append(agent)

            reward_history = []    # 評価用報酬の保存
            for episode in range(nb_episode):
                episode_reward = [] # 1エピソードの累積報酬
                actions = []
                rewards = {}
                # 全エージェントの行動選択
                for agent in agents:
                    action = agent.act()    # 行動選択
                    actions.append(action)

                # 行動の実行
                _, G = grid_env.step(actions)

                # 報酬の決定
                # Grobal Reward
                if reward_fun == "G":
                    for agent in agents:
                        rewards[agent.aid] = G 

                # Difference Rewards
                elif reward_fun == "D":
                    for agent in agents:
                        actions2 = copy.deepcopy(actions)
                        action = agent.get_previous_action()
                        actions2.remove(action)
                        _, reward2 = grid_env.step(actions2)
                        D = G-reward2
                        rewards[agent.aid] = D

                for agent in agents:
                    agent.observe(rewards[agent.aid])   # 状態と報酬の観測

                # 評価
                actions = []
                for agent in agents:
                    action = agent.act(training=False)    # 行動選択
                    actions.append(action)
                _, G = grid_env.step(actions)
                reward_history.append(G)

            results.append(copy.deepcopy(reward_history))
        results = np.array(results)
        results = results.mean(axis=0)  # 試行毎の平均の保存
        # 結果のプロット 
        plt.plot(np.arange(len(results)), results, label=reward_fun)

    plt.xlabel("episode")
    plt.ylabel("accumulated reward")
    plt.legend()
    plt.savefig("result.jpg")
    plt.show()
