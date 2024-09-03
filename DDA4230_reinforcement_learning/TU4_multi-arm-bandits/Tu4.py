# Import libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, m_arm=10, mean_reward=[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
                 std=np.ones(10)*1):
        self.true_mean = mean_reward
        self.m = m_arm
        self.std = std
        assert self.m == len(self.true_mean) and self.m == len(
            self.std), 'Number of arms must equal to numer of mean rewards/stds'

    def act(self, action):
        return np.random.randn()*self.std[action]+self.true_mean[action]


# Eps-greedy
def eps_greedy(bandits, eps, T=1000):
    m = bandits.m
    N = np.zeros(m)
    q_hat = np.zeros(m)
    avg_return = 0
    avg_return_list = np.zeros(T)
    for t in range(m):
        action = t
        reward = bandits.act(action)
        q_hat[action] = reward
        N[action] += 1
        avg_return += (reward-avg_return)/(t+2)
        avg_return_list[t] = avg_return
    for t in range(m, T):
        if np.random.rand() < eps:
            action = np.random.choice(m)
        else:
            action = np.argmax(q_hat)
        reward = bandits.act(action)
        q_hat[action] = (q_hat[action]*N[action]+reward)/(N[action]+1)
        N[action] += 1
        avg_return += (reward-avg_return)/(t+2)
        avg_return_list[t] = avg_return
    return q_hat, avg_return_list


# ETC

def ETC(bandits, k=10, T=1000):
    m = bandits.m
    N = np.zeros(m)
    q_hat = np.zeros(m)
    avg_return = 0
    avg_return_list = np.zeros(T)
    for t in range(k*m):
        action = t % m
        reward = bandits.act(action)
        q_hat[action] = reward
        N[action] += 1
        avg_return += (reward-avg_return)/(t+2)
        avg_return_list[t] = avg_return
    old_q_hat = q_hat.copy()
    for t in range(k*m, T):
        action = np.argmax(old_q_hat)
        reward = bandits.act(action)
        q_hat[action] = (q_hat[action]*N[action]+reward)/(N[action]+1)
        N[action] += 1
        avg_return += (reward-avg_return)/(t+2)
        avg_return_list[t] = avg_return
    return q_hat, avg_return_list


# UCB
def UCB(bandits, sigma, T=1000):
    m = bandits.m
    #sigma = bandits.std
    sigma = np.ones(m)*sigma
    N = np.zeros(m)
    q_hat = np.zeros(m)
    UCB = np.zeros(m)+np.inf
    avg_return = 0
    avg_return_list = np.zeros(T)

    for t in range(0, T):
        action = np.argmax(UCB)
        reward = bandits.act(action)
        q_hat[action] = (q_hat[action]*N[action]+reward)/(N[action]+1)
        UCB[action] = q_hat[action] + \
            np.sqrt(2*np.log(1/sigma[action])/(N[action]+1))
        N[action] += 1
        avg_return += (reward-avg_return)/(t+2)
        avg_return_list[t] = avg_return
    return q_hat, avg_return_list


if __name__ == '__main__':

    # Plot distribution
    np.random.seed(1)
    m = 10
    bandit_10 = Bandit(m_arm=m)
    data_set = np.zeros([200, m])
    for i in range(m):
        for j in range(200):
            data_set[j, i] = bandit_10.act(i)
    plt.violinplot(data_set, showmeans=True)
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.xticks(list(range(1, 11)), list(range(10)))
    plt.show()

    # Eps-greedy
    T = 10000
    q_hat, avg_return_list = eps_greedy(bandit_10, eps=0, T=T)
    plt.plot(np.arange(T), avg_return_list, c='red')

    q_hat, avg_return_list = eps_greedy(bandit_10, eps=0.01, T=T)
    plt.plot(np.arange(T), avg_return_list, c='green')

    q_hat, avg_return_list = eps_greedy(bandit_10, eps=0.1, T=T)
    plt.plot(np.arange(T), avg_return_list, c='black')
    plt.legend(["$\epsilon=0$", "$\epsilon=0.01$", "$\epsilon$=0.1"])

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.show()

    # ETC
    T = 10000
    q_hat, avg_return_list = ETC(bandit_10, k=1, T=T)
    plt.plot(np.arange(T), avg_return_list, c='red')

    q_hat, avg_return_list = ETC(bandit_10, k=5, T=T)
    plt.plot(np.arange(T), avg_return_list, c='green')

    q_hat, avg_return_list = ETC(bandit_10, k=10, T=T)
    plt.plot(np.arange(T), avg_return_list, c='blue')

    plt.legend(["$k=1$", "$k=5$", "$k=10$"])

    plt.plot(np.arange(T), avg_return_list)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.show()

    # UCB
    T = 10000
    q_hat, avg_return_list = UCB(bandit_10, sigma=1/T**2, T=T)
    plt.plot(np.arange(T), avg_return_list)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.show()

    # Compare
    T = 10000
    q_hat_eps, avg_return_list_eps = eps_greedy(bandit_10, eps=0.1, T=T)
    q_hat_etc, avg_return_list_etc = ETC(bandit_10, k=5, T=T)
    q_hat_ucb, avg_return_list_ucb = UCB(bandit_10, sigma=1/T**2, T=T)
    plt.plot(np.arange(T), avg_return_list_eps, c='red')
    plt.plot(np.arange(T), avg_return_list_etc, c='green')
    plt.plot(np.arange(T), avg_return_list_ucb, c='black')

    plt.legend(["Eps-greedy", "ETC", "UCB"])

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.show()
