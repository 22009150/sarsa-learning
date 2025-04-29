# SARSA Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.
## PROBLEM STATEMENT
The problem statement is a Five stage slippery walk where there are five stages excluding goal and hole.The problem is stochastic thus doesnt allow transition probability of 1 for each action it takes.It changes according to the state and policy.

State Space:
The states include two terminal states: 0-Hole[H] and 6-Goal[G]. It has five non terminal states includin starting state.

Action Space:
Left:0 Right:1

Transition probability:
The transition probabilities for the problem statement is: 50% - The agent moves in intended direction. 33.33% - The agent stays in the same state. 16.66% - The agent moves in orthogonal direction.

Reward:
To reach state 7 (Goal) : +1 otherwise : 0

## SARSA LEARNING ALGORITHM

1. Initialize Q-table with zeros, where each row represents a state, and each column represents an action.
2. Define an epsilon-greedy strategy to select actions: With probability epsilon, choose a random action, otherwise choose the action with the highest Q-value for the current state.
3. Set up learning rate (alpha) and exploration rate (epsilon) decay schedules for gradually reducing these parameters over episodes.
4. Loop through a fixed number of episodes (n_episodes), initializing the environment and setting the initial state and action.
5. In each episode, interact with the environment by taking actions, observing rewards, and transitioning to the next state until the episode terminates.
6. Update the Q-values using the SARSA update rule, incorporating the observed reward and the estimated future Q-value of the next state-action pair.
7. Track the Q-values, policy, and other information over episodes and return the final Q-values, state values (V), policy (pi), Q-value history (Q_track), and policy history (pi_track).
   
## SARSA LEARNING FUNCTION
### Name: Archana K
### Register Number:212222240011

```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
        if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, 
                           min_alpha, 
                           alpha_decay_ratio, 
                           n_episodes)
    epsilons = decay_schedule(init_epsilon, 
                              min_epsilon, 
                              epsilon_decay_ratio, 
                              n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
        state = env.reset()
        action = select_action(state, Q, epsilons[e])
        # Initialize done before the loop
        done = False  
        while not done:
          next_state, reward, done, _ = env.step(action)
          next_action = select_action(next_state, Q, epsilons[e])
          td_target = reward + gamma * Q[next_state][next_action]
          td_error = td_target - Q[state][action]
          Q[state][action] = Q[state][action] + alphas[e] * td_error
          state = next_state
          action = next_action
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q,axis=1) # Fix: Assign V instead of v
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track # Fix: Return V instead of v
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/ec49914b-f254-4d3e-987a-2e3f352a28ef)
![image](https://github.com/user-attachments/assets/48ddf994-51b9-464c-aadc-da90d9549f61)
![image](https://github.com/user-attachments/assets/1611a62f-020f-473c-ac81-b09e97988992)
![image](https://github.com/user-attachments/assets/47050b8c-2f28-4f0a-a889-84589f8da57c)
![image](https://github.com/user-attachments/assets/5457a052-a870-46bc-8507-53c6ac0241be)




## RESULT:
Thus, the implementation of SARSA learning algorithm was implemented successfully.
