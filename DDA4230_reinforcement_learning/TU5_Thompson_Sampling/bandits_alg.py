import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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
    old_q_hat=q_hat.copy()
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
    sigma=np.ones(m)*sigma
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