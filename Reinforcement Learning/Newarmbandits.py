
import matplotlib.pyplot as plt
import numpy as np


def main(Times, Plays):
    # initialisation of arrays
    Output_e01 = np.zeros(Plays)
    Output_e1 = np.zeros(Plays)
    Output_e_greedy = np.zeros(Plays)
    Output_perc_e01 = np.zeros(Plays)
    Output_perc_e1 = np.zeros(Plays)
    Output_perc_e_greedy = np.zeros(Plays)

    # for number of episodes, 2000, set array values and start experiment for different algorithms and different
    # values of epsilon
    for i in range(Times):
        arms = np.array([[np.random.normal(0, 1, 10)]])      # 10 possible actions
        K = np.array([[np.zeros(10)]])                       # K for Incremental Implementation
        Q_est = np.array([[np.zeros(10)]])                   # Each Q(a) is chosen randomly from a normal distribution
        Output_perc_e01 += test(arms[0], K[0], Q_est[0], Plays, 0.01, 1, Output_e01)  # repeat test

        arms = np.array([[np.random.normal(0, 1, 10)]])
        K = np.array([[np.zeros(10)]])
        Q_est = np.array([[np.zeros(10)]])
        Output_perc_e1 += test(arms[0], K[0], Q_est[0], Plays, 0.1, 1, Output_e1)

        arms = np.array([[np.random.normal(0, 1, 10)]])
        K = np.array([[np.zeros(10)]])
        Q_est = np.array([[np.zeros(10)]])
        Output_perc_e_greedy += test(arms[0], K[0], Q_est[0], Plays, 0, 0, Output_e_greedy)

    # take average for rewards
    average_rewards_e01 = Average(Output_e01)
    average_rewards_e1 = Average(Output_e1)
    average_rewards_g = Average(Output_e_greedy)

    # take average for Percentage Optimal Action
    Per_opt_act_e01 = Average(Output_perc_e01)
    Per_opt_act_e1 = Average(Output_perc_e1)
    Per_opt_act_g = Average(Output_perc_e_greedy)

    # draw diagram
    draw(average_rewards_e01, average_rewards_e1, average_rewards_g, "Average Reward")
    draw(Per_opt_act_e01, Per_opt_act_e1, Per_opt_act_g, "Percentage Optimal Action")


def get_reward(action, arms):
    # retrieve reward from reward array
    a = np.random.normal(0, 1)
    reward = arms[0][action] + a
    best_reward = np.argmax(arms[0])
    return reward, best_reward


def update(action, reward, K, Q_est):
    # update all values, together with estimation and pick count
    K[0][action] = 1 + K[0][action]
    alpha = 1. / K[0][action]              # StepSize
    # NewEstimate = OldEstimate + StepSize[Target – OldEstimate]
    Q_est[0][action] += alpha * (reward - Q_est[0][action])


def e_greedy(epsilon, Q_est):      # the simplest way to balance exploration and exploitation
    rand_num = np.random.random()
    if rand_num > (1 - epsilon):
        # if random number is greater than 1-epsilon, pick random action, (exploration part)
        return np.random.randint(10)
    else:
        # choose the best action, (exploitation part)
        return np.argmax(Q_est)


def test(arms, K, Q_est, plays, epsilon, option, Output):
    history = []
    reward = []
    i = 0
    while i < plays:
        # choose action method
        if option == 1:
            action = e_greedy(epsilon, Q_est)  # implementation the e-greedy action
        else:
            action = np.argmax(Q_est)          # perform the greedy action

        # get reward from carried out action
        R, best_reward = get_reward(action, arms)

        # check if action picked is optimal and then chart % optimal pick
        if action == best_reward:
            reward.append(1)
        else:
            reward.append(0)

        # update agent estimations using Incremental Implementation
        update(action, R, K, Q_est)

        # update reward gained array
        history.append(R)
        i += 1
    # add history array to carry out average at the end
    Output += np.array(history)
    return np.array(reward)


def draw(x1, x2, x3, ylabel):
    # plots
    plt.plot(x1, label="eps = 0.01")
    plt.plot(x2, label="eps = 0.1")
    plt.plot(x3, label="greedy")
    plt.ylim(0, 1.5)
    plt.xlabel("Steps")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    return


def Average(Value,):
    mean = Value / 2000
    return mean


if __name__ == '__main__':
    # initialisation of number of episodes and number of pulls per episode
    iterations = 2000
    step = 1000
    main(iterations, step)
