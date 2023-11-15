import gymnasium as gym
import numpy as npy

def main(episodes):
    env = gym.make('CliffWalking-v0', render_mode='human')
    state = env.reset()

    # q function variables
    learning_rate = 0.9
    discount = 0.9

    # policy variables (epsilon-greedy)
    epsilon = 1 # random actions
    decay = 0.0001 # 1/0.0001 steps per episode
    rng = npy.random.default_rng()

    # initialize q-table
    rows = env.observation_space.n
    columns = env.action_space.n
    q_table = npy.zeros((rows, columns)) # empty 2D array with zeroes as values

    # go through episodes
    for _ in range(episodes):
        while (not terminated and not truncated):
            if rng.random() < epsilon:
                action = env.action_space.sample() # random action
            else:
                action = npy.argmax(q_table[state,:]) # largest value from q-table in row state

            new_state, reward, terminated, truncated, info = env.step(action)

            # q function
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount * npy.max(q_table[new_state,:]) - q_table[state, action])

            state = new_state

            if terminated or truncated:
                env.reset() # reset game if agent fails or stalls
        
        epsilon = max(epsilon - decay, 0) # if epsilon < 0, default to 0
    



    env.close()

if __name__ == '__main__':
    main(10000)