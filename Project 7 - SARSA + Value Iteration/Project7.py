# Anthony Gray
# Intro to Machine Learning Project 7
# Main Project File

# libraries
import pickle

# Project Files
from utils import Agent, Track, MDP
import ValueIteration
import SARSA

# method displays track with SARSA start points shown -----------------------------------
def print_track_with_starts(track, starts):
        for s in starts:
                track.track[s[0]][s[1]] = '!'
        print(track)
# end print_track_with_starts() ---------------------------------------------------------

#----------------------------------------------------------------------------------------
# paired methods for training and storing MDPs for value iteration ----------------------
# train and save MDP
def train_and_save_VI_MDP(track_str, gamma, iters):
    test_track = Track(track_str+'.txt')
    test_mdp = MDP(test_track)
    ValueIteration.value_iteration(test_mdp, gamma = gamma, epsilon = 1)
    with open(f'pickles\{track_str}_{gamma}_VI_pickle_Iter{iters}','wb') as file:
        pickle.dump(test_mdp, file) 
    print('Training Complete')
# end train_and_save_VI_MDP() -----------------------------------------------------------
# # open MDP file
def recover_VI_MDP_file(track_str, gamma, iters):
    with open(f'pickles\Good Runs\{track_str}_{gamma}_VI_pickle_Iter{iters}','rb') as inp:
        trained_mdp = pickle.load(inp)
    return trained_mdp
# end recover_VI_MDP_file() -------------------------------------------------------------
#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
# paired methods for training and storing MDPs for SARSA --------------------------------
# train and save MDP
def train_and_save_SAR_MDP(track_str, starts, gamma):
    test_track = Track(track_str+'.txt')
    test_mdp = MDP(test_track, algorithm='SARSA')
    SARSA.sarsa(test_mdp, starts, gamma = gamma)
    with open(f'pickles\{track_str}_{gamma}_SAR_pickle','wb') as file:
        pickle.dump(test_mdp, file) 
    print('Training Complete')
# end train_and_save_SAR_MDP() ---------------------------------------------------------

# open MDP file ------------------------------------------------------------------------
def recover_SAR_MDP_file(track_str, gamma):
    with open(f'pickles\Good Runs\{track_str}_{gamma}_SAR_pickle','rb') as inp:
        trained_mdp = pickle.load(inp)
    return trained_mdp
# end recover_SAR_MDP_file() -----------------------------------------------------------
#---------------------------------------------------------------------------------------

# lists of SARSA start points for each track
O_track_start_points = [(14,2),(17,2),(19,3),(21,4),(22,5),(22,8),(22,11),(22,14),(22,17),(22,19),(20,20),(18,21),(16,21),(14,21),(12,21),(10,21),(8,21),(6,21),(4,21),(3,19),(2,18),(2,15),(2,12),(2,9),(2,6),(3,5),(4,4),(5,3),(8,3)]
L_track_start_points = [(3,34),(5,34),(7,33),(8,29),(8,24),(8,20),(8,15),(8,10),(8,5),(6,30),(7,27),(8,32),(9,31),(9,34),(9,26),(6,24),(9,22),(7,22),(6,20),(9,18),(7,18),(6,16),(9,14),(7,13),(6,12),(9,10),(6,8),(7,8),(9,6),(6,4),(9,2),(7,2)]
R_track_start_points = [(24,26),(22,25),(19,25),(16,24),(14,18),(11,13),(8,16),(6,18),(4,21),(2,19),(2,15),(2,11),(2,22),(2,8),(3,5),(4,5),(6,4),(9,5),(12,3),(14,3),(16,4),(18,4),(20,4),(23,3),(25,3)]
T_track_start_points = [(3,1),(4,2)]

track_str = 'T-track_0.05_SAR_pickle'
gamma = 0.01
iters = 50
# print(f'Training {track_str} with gamma = {gamma} and iters = {iters}')
# train_and_save_VI_MDP(track_str, gamma, iters)
# mdp = recover_VI_MDP_file(track_str,gamma,iters)

# train_and_save_SAR_MDP(track_str, L_track_start_points,gamma)
mdp = recover_SAR_MDP_file(track_str,gamma)

# mdp.track.reset = True
mdp.run_simulation(animate=False)

# print_track_with_starts(mdp.track,L_track_start_points)
