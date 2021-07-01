"""
Agent taking reasonable actions based on hand-made rules,
i.e. go to right if leaning towards left, vice versa.

Observation space: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
Action space: [moving cart to left, moving cart to right]

Note that good agents learn the policies by themselves,
and don't need to know the meaning of the observations.
"""
import gym
import matplotlib.pyplot as plt


N_EPISODES = 200 
EPISODE_LENGTH = 200

def plot_rewards(rewards, n_episodes, algo):
    plt.plot(list(range(n_episodes)), rewards)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.ylim(0, EPISODE_LENGTH)
    plt.title('Rewards over episodes ({})'.format(algo))
    plt.savefig('{}.png'.format('_'.join(algo.split(' '))))

def choose_action(observation):
    pos, v, ang, rot = observation
    return 0 if ang < 0 else 1 # a simple rule based only on angles

def main():
    env = gym.make('CartPole-v0')

    all_rewards = []
    for i_episode in range(N_EPISODES):
        observation = env.reset() # reset environment to initial state for each episode
        rewards = 0 # accumulate rewards for each episode
        for t in range(EPISODE_LENGTH):
            env.render()

            action = choose_action(observation) # choose an action based on hand-made rule 
            observation, reward, done, info = env.step(action) # do the action, get the reward
            rewards += reward

            if done:
                print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break
        all_rewards.append(rewards)

    env.close() # need to close, or errors will be reported
    plot_rewards(all_rewards, N_EPISODES, 'hand made')


if __name__ == '__main__':
    main()
