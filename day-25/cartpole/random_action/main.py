"""
Agent taking random actions.
Based on the example in OpenAI docs: https://gym.openai.com/docs/
"""
import gym
import matplotlib.pyplot as plt


N_EPISODES = 200 
EPISODE_LENGTH = 200

def plot_rewards(rewards, n_episodes, algo):
    plt.plot(list(range(n_episodes)), rewards)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.ylim(0, EPISODE_LENGTH+5)
    plt.title('Rewards over episodes ({})'.format(algo))
    plt.savefig('{}.png'.format('_'.join(algo.split(' '))))

def main():
    env = gym.make('CartPole-v0')

    all_rewards = []
    for i_episode in range(N_EPISODES):
        observation = env.reset() # reset environment to initial state for each episode
        rewards = 0 # accumulate rewards for each episode
        for t in range(EPISODE_LENGTH):
            env.render()

            action = env.action_space.sample() # choose a random action
            observation, reward, done, info = env.step(action) # do the action, get the reward
            rewards += reward

            if done:
                print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break
        all_rewards.append(rewards)

    env.close() # need to close, or errors will be reported
    plot_rewards(all_rewards, N_EPISODES, 'random action')


if __name__ == '__main__':
    main()
