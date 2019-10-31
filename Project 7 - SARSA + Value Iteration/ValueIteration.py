# Anthony Gray
# Intro to Machine Learning Project 7
# Value Iteration Methods File

from utils import bresenham_line, find_s_prime, constrain_velocity

# method accpets MDP, trains value and policy tables --------------------------------------------------------------------
def value_iteration(mdp, gamma, epsilon):
    # iteration counter
    t = 0
    # transition probability
    T = 0.8
    # convergence flag
    convergence = False
    while not convergence:
        update = False
        # update counter
        t += 1
        # iterate over all locations
        for l in mdp.L:
            # iterate over all velocities for each location
            for s in mdp.S:
                # iterate over all actions
                best_a = ('*','*')
                best_Q = float('-inf')
                for a in mdp.A:
                    # add acceleration to current velocty, contrained
                    new_velocity = constrain_velocity(s,a)
                    # calculate s_prime by checking where acceleration update would land
                    s_prime = find_s_prime(l, new_velocity, mdp.L)
                    # find direct reward for landing taking action
                    R = mdp.V.at[s, s_prime]

                    # find V values by assuming optimal move from next step
                    V = find_V(s_prime, s, mdp.A, mdp.L, mdp.V)
                    old_Q = mdp.V.at[s,l]
                    Q = R + (gamma*T*V)
                    # if Q < -100:
                    #     Q = -100
                    # store reward value in Value table
                    mdp.V.at[s,l] = Q
                    # remember best action for each state
                    if Q > best_Q:
                        best_Q = Q
                        best_a = a
                    if abs(Q-old_Q) > epsilon:
                        update = True
                # store best action in policy table
                mdp.Pi.at[s,l] = best_a
        # print(t)
        if t>50:
            convergence = True
# end value_iteration() -----------------------------------------------------------------------------------------------

# method for checking the value of the best subsequent step -----------------------------------------------------------
def find_V(s, velocity, actions, locations, val_table):
    # value starts at 0
    V = 0
    # check each action
    for a in actions:
        # add acceleration to current velocty, contrained
        new_velocity = constrain_velocity(velocity,a)
        # calculate s_prime by checking where acceleration update would land
        s_prime = find_s_prime(s, new_velocity, locations)
        # update V with value of the best next possible move
        V += val_table.at[new_velocity, s_prime]
    return V
# end find_V() --------------------------------------------------------------------------------------------------------