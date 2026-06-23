from matplotlib.pyplot import grid
import random
from collections import deque
import numpy as np 
import matplotlib.pyplot as plt

def grid_reward(prev_pos, curr_pos, goal_pos, obstacles, visited):
    #shut the fuck up
    reward = 0
    if curr_pos == goal_pos:
        reward += 10
        return reward
    if curr_pos in obstacles:
        reward -= 20
        return reward
    
    

    prev_distance = abs(prev_pos[0] - goal_pos[0]) + abs(prev_pos[1] - goal_pos[1])
    curr_distance = abs(curr_pos[0] - goal_pos[0]) + abs(curr_pos[1] - goal_pos[1])
    if curr_distance < prev_distance:
        reward += 1
    elif curr_distance > prev_distance:
        reward -= 0.5
    
    if curr_pos in visited:
        reward -= 1
    else:
        reward += 1
    visited.add(curr_pos)

    return reward

def check_possible(start, goal, obstacles, grid_size):
    visited = set()
    queue = deque([start])

    
    while queue:
        pos = queue.popleft()
        if pos == goal:
            return True
        if pos in visited:
            continue
        visited.add(pos)
        
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= ny < grid_size and 0 <= nx < grid_size:
                if (nx, ny) not in obstacles and (nx, ny) not in visited:
                    queue.append((nx, ny))  
    return False

def grid_creation(grid_size, num_obs, actions, max_attempts):
    table = {}
    obstacles = set()
    #make table 4*4
    for i in range(grid_size):
        for j in range(grid_size):
            table[(i, j)] = {action: 0.0 for action in actions}

    attempts = 0
    while len(obstacles) < num_obs and attempts < max_attempts:
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        attempts += 1
        if x == 0 and y == 0:
            continue
        elif x == grid_size - 1 and y == grid_size - 1:
            continue
        obstacles.add((x, y))
        if not check_possible((0, 0), (grid_size-1, grid_size-1), obstacles, grid_size):
            obstacles.remove((x, y))
    return table, obstacles

def move(state, action, grid_size):
    org = state
    if action == 'UP':
        state = (state[0], state[1] + 1)
    elif action == 'DOWN':
        state = (state[0], state[1] - 1)
    elif action == 'LEFT':
        state = (state[0] - 1, state[1])
    elif action == 'RIGHT':
        state = (state[0] + 1, state[1])
    if 0 <= state[0] < grid_size and 0 <= state[1] < grid_size:
        return state
    else:
        return org


actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
grid_size = 20
num_obs = 200
max_attempts = 300
alpha = 0.1
gamma = 0.9
epsilon = 0.2
start_x = start_y = 0
goal_x = goal_y = grid_size - 1
goal_pos = (goal_x, goal_y)
episodes = 2000

Q, obs = grid_creation(grid_size, num_obs, actions, max_attempts)

for episode in range(episodes):
    state = (start_x, start_y)
    visited = set()

    while state != (goal_x, goal_y):

        if random.random() < epsilon:
            action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        else:
            action = max(Q[state], key=Q[state].get)

        next_state = move(state, action, grid_size)

        r = grid_reward(state, next_state, goal_pos, obs, visited)

        old_value = Q[state][action]
        future_best = max(Q[next_state].values())
        Q[state][action] = old_value + alpha * (r + gamma * future_best - old_value)

        state = next_state

print('Done')

# After training, test the greedy policy
state = (0, 0)
path = [state]
while state != goal_pos and len(path) < 100:
    action = max(Q[state], key=Q[state].get)
    state = move(state, action, grid_size)
    path.append(state)

print("Path:", path)







fig, ax = plt.subplots(figsize=(5, 5))

# Grid
for i in range(grid_size + 1):
    ax.axhline(i, color='gray', linewidth=0.5)
    ax.axvline(i, color='gray', linewidth=0.5)

# Obstacles
for (ox, oy) in obs:
    ax.fill_between([ox, ox+1], oy, oy+1, color='black')

# Start and goal
ax.fill_between([0, 1], 0, 1, color='blue', alpha=0.3)
ax.fill_between([grid_size-1, grid_size], grid_size-1, grid_size, color='green', alpha=0.3)
ax.text(0.5, 0.5, 'S', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(grid_size-0.5, grid_size-0.5, 'G', ha='center', va='center', fontsize=14, fontweight='bold')

# Path
px = [p[0] + 0.5 for p in path]
py = [p[1] + 0.5 for p in path]
ax.plot(px, py, 'ro-', linewidth=2)

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.invert_yaxis()
plt.show()