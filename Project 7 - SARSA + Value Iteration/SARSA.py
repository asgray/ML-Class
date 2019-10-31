# Anthony Gray
# Intro to Machine Learning Project 7
# SARSA Methods File

from random import uniform, choice
from utils import bresenham_line, find_s_prime, constrain_velocity

# main SARSA method, accepts blank MDP object, list of starting postions, and discount factor gamma -------------------------
def sarsa(mdp, starting_points, gamma):
    # find finish line
    stopping_points = mdp.track.finishes
    j = 0
    epsilon = 0.1
    # each episode starts at a starting point
    for point in starting_points:
        # identify points in range of the starting point to test
        trial_points = find_neighbors(point,mdp)
        i=0
        # for each trial point
        for p in trial_points:
            eta = 0.98
            # start at p
            loc = p 
            # check each velocity from p
            for vel in mdp.S:
                # find best acceleration
                a = epsilon_greedy(vel,loc,epsilon,mdp)
                terminal = False
                while not terminal:
                    # calculate R, Q(s,a), new velocity and s'
                    R = mdp.V.at[vel, loc]
                    Q, new_velocity, s_prime = find_Q(vel, loc, a, mdp)
                    # find a' based on s' and new velocity
                    a_prime = epsilon_greedy(new_velocity,s_prime,epsilon,mdp)
                    # find Q', x and y are dummy variables that are not used
                    Q_prime, x, y = find_Q(new_velocity, s_prime, a_prime, mdp)
                    # caculate Q update with Q, Q', R, gamma and eta
                    total_Q = Q + (eta*(R + (gamma*Q_prime) - Q))
                    # update V and policy tables
                    mdp.V.at[vel,loc] = total_Q
                    mdp.Pi.at[vel,loc] = a
                    # reset location to s' and action to a'
                    loc = s_prime
                    a = a_prime
                    vel = new_velocity
                    loc = s_prime
                    # annealing
                    eta = eta*0.85
                    i += 1
                    if loc in stopping_points or i>500:
                        terminal = True
        j += i
    print(j)
# end sarsa() ------------------------------------------------------------------------------------------------------------

# method to find all points within two steps of starting point -----------------------------------------------------------
def find_neighbors(point, mdp):
    # possible distances
    neighbor_change = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1),(-2,-2),(-2,0),(-2,2),(0,-2),(0,2),(2,-2),(2,0),(2,2))
    # include staring point
    neighbors = [point]
    # find and attach all s'
    for change in neighbor_change:
        neighbor = tuple([sum(x) for x in zip(point, change)])
        # make sure s' is a valid location on track
        if neighbor in mdp.L:
            neighbors.append(neighbor)
    return neighbors
# end find_neighbors() ---------------------------------------------------------------------------------------------------

# method takes a current location and velocity, applies an acceleration to the velocity and returns the new location -----
def find_Q(vel, loc, a, mdp):
    # add acceleration to current velocty, contrained
    new_velocity = constrain_velocity(vel,a)
    # calculate s_prime by checking where acceleration update would land
    s_prime = find_s_prime(loc, new_velocity, mdp.L)
    # find associated reward
    R = mdp.V.at[new_velocity, s_prime]
    return R, new_velocity, s_prime
# end find_Q() ------------------------------------------------------------------------------------------------------------

# method chooses action with highest payout with probability (1-epsilon) --------------------------------------------------
def epsilon_greedy(velocity, location, epsilon, mdp):
    action = None
    # take ranomd action epsilon% of the time
    if uniform(0,1) <= epsilon:
        action = choice(mdp.A)
    else:
        # iterate over all actions
        action = choice(mdp.A)
        best_Q = float('-inf')
        # check and save each Q value
        for a in mdp.A:
            # other variables are not used
            Q, new_velocity, s_prime = find_Q(velocity, location, a, mdp)
            if Q > best_Q:
                best_Q = Q
                action = a
            # print(location, velocity, a, new_velocity, s_prime, Q)
    return action
# end epsilon_greedy() ---------------------------------------------------------------------------------------------------