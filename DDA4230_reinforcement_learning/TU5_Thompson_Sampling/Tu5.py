import numpy as np
import matplotlib.pyplot as plt
from bandits_alg import eps_greedy, ETC, UCB

# Bernoulli MAB
class Bandit:
    def __init__(self, m_arm=10, p=None):
        self.m = m_arm
        if p:
            self.p = p
            assert self.m == len(self.p), 'Number of arms must equal to numer of probabilities'
        else:
            self.p = np.random.random_sample(self.m)
        self.max_prob = np.amax(self.p)
        self.optimal_arm = np.argmax(self.p)

    def act(self, action):
        return np.random.binomial(1, p=self.p[action])

    def regret(self, action):
        return self.max_prob - self.p[action]

# TS
def sample_bandit_distribution(beta_parameter):
    """Sample theta from Beta distribution

    Parameters
    ----------
    beta_parameter: np.array[m，2]
        Parameters of Beta distribution

    Returns
    -------
    theta: np.ndarray[m]
        Value of theta for each arm
    """
    theta = np.random.beta(beta_parameter[0, :], beta_parameter[1, :])
    return theta


def Thompson_sampling(bandits, T=1000):
    m = bandits.m
    # Reward estimation (Optional)
    N = np.zeros(m)
    q_hat = np.zeros(m)
    avg_return = 0
    avg_return_list = np.zeros(T)

    # Initialize Prior Parameters
    beta_parameter = np.ones((m, 2))

    for t in range(0, T):
        # Sample theta from Bets distribution
        theta = sample_bandit_distribution(beta_parameter)
        action = np.argmax(theta)
        reward = bandits.act(action)

        # Update Beta Parameters
        if reward == 1:
            beta_parameter[action, 0] += 1
        else:
            beta_parameter[action, 1] += 1

        # Reward estimation (Optional)
        q_hat[action] = (q_hat[action]*N[action]+reward)/(N[action]+1)
        N[action] += 1
        avg_return += (reward-avg_return)/(t+1)
        avg_return_list[t] = avg_return
    return q_hat, avg_return_list

# Policy Evaluation
"""
For policy_evaluation, the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
        tuple of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS，nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is the value of state s
    """

    value_function = np.zeros(nS)

    iteration = 0
    while iteration < 1e5:
        iteration += 1
        delta = 0
        ############################
        # YOUR IMPLEMENTATION HERE #
        for s in range(nS):
            last_v = value_function[s]
            v_p = 0
            for a in range(nA):
                transitions = P[s][a]
                v_a = np.array([t[0]*(t[2]+gamma*value_function[t[1]])
                                for t in transitions])
                v_p += policy[s, a]*(np.sum(v_a))
            value_function[s] = v_p
            delta = max(delta, abs(last_v-value_function[s]))
        ############################
        if delta < tol:
            break
    #print('Number of Iteration:', iteration)
    return value_function
if __name__ == '__main__':
    # Plot Reward Distribution
    np.random.seed(1)
    m = 10
    bandit_Bernoulli = Bandit(m_arm=m)
    data_set = np.zeros([200, m])
    for i in range(m):
        for j in range(200):
            data_set[j, i] = bandit_Bernoulli.act(i)
    plt.violinplot(data_set, showmeans=True)
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.xticks(list(range(1, 11)), list(range(10)))
    plt.show()
    print('The Best Arm is', bandit_Bernoulli.optimal_arm,
        'with mean', bandit_Bernoulli.max_prob)

    # Test TS, Eps-greedy, ETC, UCB
    T = 10000
    q_hat_ts, avg_return_list_ts = Thompson_sampling(bandit_Bernoulli, T=T)
    q_hat_eps, avg_return_list_eps = eps_greedy(bandit_Bernoulli, eps=0.1, T=T)
    q_hat_etc, avg_return_list_etc = ETC(bandit_Bernoulli, k=5, T=T)
    q_hat_ucb, avg_return_list_ucb = UCB(bandit_Bernoulli, sigma=1, T=T)
    plt.plot(np.arange(T), avg_return_list_ts, c='blue')
    plt.plot(np.arange(T), avg_return_list_eps, c='red')
    plt.plot(np.arange(T), avg_return_list_etc, c='green')
    plt.plot(np.arange(T), avg_return_list_ucb, c='black')

    plt.legend(["Thompson Sampling", "Eps-greedy", "ETC", "UCB"])

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.show()
    plt.savefig('Bandits.png')

    # Gym
    import gym
    # Create Environment
    env = gym.make('Taxi-v3')
    # Reset
    env.reset()
    for _ in range(1):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()

    env = gym.make('CartPole-v0')
    print(env.action_space)
    print(env.observation_space)

    print(env.observation_space.high)
    print(env.observation_space.low)

    '''
    FrozenLake
    SFFF       (S: starting point, safe)
    FHFH       (F: frozen surface, safe)
    FFFH       (H: hole, fall to your doom)
    HFFG       (G: goal, where the frisbee is located)
    '''
    # Create Environment
    env = gym.make('FrozenLake-v0')
    # Reset
    env.reset()
    env.render()

    nA = env.nA
    nS = env.nS
    P = env.P

    policy = np.ones((nS, nA))/nA  # Uniform Policy
    V_pi = policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3)
    print(V_pi)
    env.close()