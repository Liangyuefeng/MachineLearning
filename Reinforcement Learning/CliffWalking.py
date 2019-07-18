#!/user/bin/python3
# this program is designed for COMP532 assignment 1
# authors: Rui Zhang (id: 201255800) & Yuefeng Liang (id: 201350099)
# last update: 07/03/2019
import numpy as np
import matplotlib.pyplot as plt

def main():
    # start position for agent
    startPos = [3, 0]
    # end position
    endPos = [3, 11]
    # assign some constants 
    alpha = 0.1
    gamma = 1
    epsilon = [0.1, 0.01, 0.001]
    episodes = 500
    # number of rounds
    times = 10
    
    # define smoothed results for Q-learning and SARSA
    smoothRewardsQ = np.zeros((len(epsilon),episodes),dtype=float)
    smoothRewardsSARSA = np.zeros((len(epsilon),episodes),dtype=float)

    for k in range(0, len(epsilon)):
        # define action values, there are 4 * 12 states, and each state has 4 kind of actions: up, down, left and right
        actionValueQ = np.zeros((4,12,4), dtype=float)
        actionValueSARSA = np.zeros((4,12,4), dtype=float)
        # define sum of rewards during episodes
        rewardsQ = np.zeros((episodes), dtype=float)
        rewardsSARSA = np.zeros((episodes),dtype=float)
        for t in range(0, times):
            for i in range(0, episodes):
                # sum of rewards get by Q-learning
                sumRewardQ = qLearning(epsilon[k], alpha, gamma, startPos, endPos, actionValueQ)
                rewardsQ[i] = rewardsQ[i] + sumRewardQ

                # sum of rewards get by SARSA
                sumRewardSARSA = sarsa(epsilon[k], alpha, gamma, startPos,endPos, actionValueSARSA)
                rewardsSARSA[i] = rewardsSARSA[i] + sumRewardSARSA

        # averaging over 10 times
        rewardsQ = rewardsQ / times
        rewardsSARSA = rewardsSARSA / times

        smoothRewardsQ[k] = np.copy(rewardsQ)
        smoothRewardsSARSA[k] = np.copy(rewardsSARSA)

        # averaging over the last 10 episodes
        for i in range(times, episodes):
            smoothRewardsQ[k][i] = np.mean(rewardsQ[i - times: i + 1])
            smoothRewardsSARSA[k][i] = np.mean(rewardsSARSA[i - times: i + 1])

    # draw the results of Q-learning with different epsilon value
    plt.figure()
    plt.plot(smoothRewardsQ[0], label="Q-learning epsilon = 0.1")
    plt.plot(smoothRewardsQ[1], label="Q-learning epsilon = 0.01")
    plt.plot(smoothRewardsQ[2], label="Q-learning epsilon = 0.001")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episodes")
    plt.ylim(-100,0)
    plt.legend()

    # draw the results of SARSA with different epsilon value
    plt.figure()
    plt.plot(smoothRewardsSARSA[0], label="SARSA epsilon = 0.1")
    plt.plot(smoothRewardsSARSA[1], label="SARSA epsilon = 0.01")
    plt.plot(smoothRewardsSARSA[2], label="SARSA epsilon = 0.001")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episodes")
    plt.ylim(-100,0)
    plt.legend()

    # draw the results of Q-learning and SARSA with epsilon equals to 0.1
    plt.figure()
    plt.plot(smoothRewardsQ[0], label="Q-learning epsilon = 0.1")
    plt.plot(smoothRewardsSARSA[0], label="SARSA epsilon = 0.1")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episodes")
    plt.ylim(-100,0)
    plt.legend()

    # # draw the results of Q-learning and SARSA with epsilon equals to 0.001
    plt.figure()
    plt.plot(smoothRewardsQ[2], label="Q-learning epsilon = 0.001")
    plt.plot(smoothRewardsSARSA[2], label="SARSA epsilon = 0.001")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episodes")
    plt.ylim(-100,0)
    plt.legend()
    plt.show()

def movement(position, action):
    # define a new position
    newPosition = [0,0]
    # define reward of current action
    actionReward = 0
    # if agent is going up
    if action == 0:
        # get new position
        newPosition[0] = max(position[0] - 1, 0)
        newPosition[1] = position[1]
        # get reward
        actionReward = -1
    # if agent is going down
    elif action == 1:
        # if agent is at position [2,1],[2,2],...,[2,10], and prepare to go down, in that way, it will be send to start position [3,0], and get reward for -100
        if position[0] == 2 and 0<position[1]<11:
            newPosition[0] = 3
            newPosition[1] = 0
            actionReward = -100
        # if agent is at other position, then get new position as normal
        else:
            newPosition[0] = min(position[0] + 1, 3)
            newPosition[1] = position[1]
            # get reward
            actionReward = -1
    # if agent is going left
    elif action == 2:
        newPosition[0] = position[0]
        newPosition[1] = max(position[1] - 1, 0)
        actionReward = -1
    # if agent is gong right
    elif action == 3:
        # if agent is at start position [3,0], and preapre to go right, in that way, it will be send to start position [3,0], and get reward for -100 
        if position[0] == 3 and position[1] == 0:
            newPosition[0] = 3
            newPosition[1] = 0
            actionReward = -100
        # if agent is at other position, then get new position as normal
        else:
            newPosition[0] = position[0]
            newPosition[1] = min(position[1] + 1, 11)
            actionReward = -1
    return newPosition, actionReward

def actionGreedy(position, actionValue, epsilon):
    p = np.random.random()
    # if random number is greater equal than epsilon,
    if p >= epsilon:
        # get all action value of current position
        actionValueCopy = np.copy(actionValue[position[0]][position[1]][:])
        # choose action with maximum action value
        action = np.random.choice(np.argwhere(actionValueCopy == np.amax(actionValueCopy)).flatten().tolist())
    # if random number is smaller than epsilon, then choose action randomly
    else:
        action = np.random.randint(4)
    return action

def qLearning(epsilon, alpha, gamma, position, endPos, actionValue):
    sumReward = 0
    # if agent has not reached end position
    while position != endPos:
        # get a action by using greedy rules
        action = actionGreedy(position, actionValue, epsilon)
        # get new position and reward for current action
        [newPosition, actionReward] = movement(position, action)
        # update sum of reward
        sumReward = sumReward + actionReward
        # update action value by using Q-learning rules
        actionValue[position[0]][position[1]][action] += alpha * (actionReward + gamma * np.max(actionValue[newPosition[0]][newPosition[1]][:]) - actionValue[position[0]][position[1]][action])
        # set agent to new states
        position = newPosition
    return sumReward

def sarsa(epsilon, alpha, gamma, position, endPos, actionValue):
    sumReward = 0
    # get a action by using greedy rules
    action = actionGreedy(position, actionValue, epsilon)
    # if agent has not reached end position
    while position != endPos:
        # get new position and reward for current action
        [newPosition, actionReward] = movement(position, action)
        # update sum of reward
        sumReward = sumReward + actionReward
        # get a new action by using new position
        newAction = actionGreedy(newPosition, actionValue, epsilon)
        # update action value by using SARSA rules
        actionValue[position[0]][position[1]][action] += alpha * (actionReward + gamma * actionValue[newPosition[0]][newPosition[1]][newAction] - actionValue[position[0]][position[1]][action])
        # set agent to new states
        position = newPosition
        action = newAction
    return sumReward

if __name__  == '__main__':
    main()
