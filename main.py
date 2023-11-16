import gymnasium as gym
import numpy as npy
import pickle

def main(episodes, steps, render=None, trainingMode=False):
    env = gym.make('CliffWalking-v0', render_mode = render)

    # q function variables
    learning_rate = 0.9
    discount = 0.9

    # policy variables (epsilon-greedy)
    epsilon = 1 # random actions
    decay = 0.0001 # 1/0.0001 = 10000 episodes for full learning
    rng = npy.random.default_rng()

    # initialize q-table
    if trainingMode:
        rows = env.observation_space.n
        columns = env.action_space.n
        q_table = npy.zeros((rows, columns)) # empty 2D array with zeroes as values
    else:
        qdata = open('data.pkl', 'rb') # trained q table data
        q_table = pickle.load(qdata)
        qdata.close()

    # go through episodes
    for episode in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        for step in range(steps):
            if trainingMode and rng.random() < epsilon:
                action = env.action_space.sample() # random action
            else:
                action = npy.argmax(q_table[state,:]) # largest value from q-table in row called state

            new_state, reward, terminated, truncated, info = env.step(action)

            # q function
            if trainingMode:
                q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount * npy.max(q_table[new_state,:]) - q_table[state, action])

            state = new_state
            if terminated or truncated:
                if not trainingMode:
                    env.end()
                state = env.reset()[0]
        
        epsilon = max(epsilon - decay, 0) # if epsilon < 0, default to 0

        if epsilon <= 0:
            learning_rate = 0.0001 # stop learning once we start relying on q-table

    env.close()

    if trainingMode: # save qtable data after training
        newData = open("data.pkl", "wb")
        pickle.dump(q_table, newData)
        newData.close()
        print("Training completed after " + str(episode+1) + " episodes!")

if __name__ == '__main__':
    step_default = 25
    trainValue = int(input("0 - Use saved Q-table\n1 - Begin training\nInput: "))
    if trainValue == 1: # Train
        episodeValue = int(input("Please enter the amount of episodes you want to train the agent: "))
        main(episodeValue, step_default, None, True)
        print("Please re-run the program and use the q-table to view the results.")
    elif trainValue == 0: # Use saved Q-table
        try:
            main(1, step_default, 'human', False)
        except:
            print("There is no available q-table to read.")   