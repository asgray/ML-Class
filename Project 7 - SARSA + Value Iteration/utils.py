# Anthony Gray
# Intro to Machine Learning Project 7
# Class File

from random import randrange, choice
import numpy as np
import pandas as pd
import math

# display imports
import os
clear = lambda: os.system('cls')
from time import sleep as sleep


#************************************
# CLASSES
#************************************
# class to store Agent behaviors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Agent:
    # initialize velocity and locaton to (0,0) --------------
    def __init__(self):
        # Agent variables
        self.velocity = [0,0]
        self.location = (0,0)
    # end init() --------------------------------------------

    # method for accelerating -------------------------------
    def accelerate(self, action):
        # 20% of time, attempt fails
        if randrange(1,100) <= 20:
            pass
        else:
            # otherwise add acceleration to velocity
            self.velocity[0] += action[0]
            self.velocity[1] += action[1]
            # constrian velocity 
            self.constrain_velocity()
    # end accelerate() --------------------------------------
    
    # method keeps velocity values within range of -5 to 5 --
    def constrain_velocity(self):
        # for each velocity coordinate (only two)
        for n in range(len(self.velocity)):
            # extract value
            x = self.velocity[n]
            # if greater than 5, cap at 5
            if x > 5:
                self.velocity[n] = 5
            # if less than -5, increase to -5
            elif x < -5:
                self.velocity[n] = -5
    # end constrain_velocity() ------------------------------
    
    # list of valid accleeration actions and velocity states
    actions = ((-1, 0), (-1, -1), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))
    velocity_states = [(-5, -5), (-5, -4), (-5, -3), (-5, -2), (-5, -1), (-5, 0), (-5, 1), (-5, 2), (-5, 3), (-5, 4), (-5, 5), 
        (-4, -5), (-4, -4), (-4, -3), (-4, -2), (-4, -1), (-4, 0), (-4, 1), (-4, 2), (-4, 3), (-4, 4), (-4, 5), 
        (-3, -5), (-3, -4), (-3,-3), (-3, -2), (-3, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3), (-3, 4), (-3, 5), 
        (-2, -5), (-2, -4), (-2, -3), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3), (-2, 4), (-2, 5), 
        (-1, -5), (-1, -4), (-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4), (-1, 5), 
        (0, -5), (0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), 
        (1, -5), (1, -4), (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), 
        (2, -5), (2, -4), (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), 
        (3, -5), (3, -4), (3, -3), (3, -2), (3, -1), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), 
        (4, -5), (4, -4), (4, -3), (4, -2), (4, -1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), 
        (5, -5), (5, -4), (5, -3), (5, -2), (5, -1), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
# end Agent class ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# class to maintain Track data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class maintains grid of track values, valid starting and ending areas, and location of Agent
# also controls the movement of the Agent and what happens when it crashes
class Track:

    # initialize with filepath to track.txt, reset flag controls crash behavior ------------------------------------------------------------------------------
    def __init__(self, path, reset=False):
        # read in file line by line
        path = "C:\\Users\\Anthony\\Dropbox\\Serious\\School\\Classes\\current\\Intro to Machine Leanring - EN.605.649\\Projects\\Project 7\\tracks\\" + path
        with open(path) as f:
            self.track = f.readlines()
        # store dimensions
        self.dimensions = [int(s) for s in self.track[0].strip().split(',') if s.isdigit()]
        # split lines into lists of characters
        self.track = [list(x.strip()) for x in self.track[1:]]

        # save all valid start and ending locations
        self.starts = []
        self.finishes = []
        # also store all valid locations
        self.valid_locs = []
        for row in range(self.dimensions[0]):
            for col in range(self.dimensions[1]):
                if self.track[row][col] == 'S':
                    self.starts.append((row,col))
                if self.track[row][col] == 'F':
                    self.finishes.append((row,col))
                if self.track[row][col] == '.' or self.track[row][col] == 'F' or self.track[row][col] == 'S':
                    self.valid_locs.append((row,col))
        
        # initialize and agent, set starting location to random start space
        self.agent = Agent()
        self.agent.location = choice(self.starts)
        # store crash behavior, True resets location to start, False just stops the Agent
        self.reset = reset
    # end init() ----------------------------------------------------------------------------------------------------------------------------------------------

    # method for printing current track to console ------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        rep_str = ''
        # proceed through grid, building string of space symbols
        for r_id, row in enumerate(self.track):
            for s_id, space in enumerate(row):
                # if space is occupied by agent, use agent symbol
                if (r_id, s_id) == self.agent.location:
                    rep_str += '@'
                elif (r_id, s_id) in self.move_traces:
                    rep_str += 'x'
                # otherwise use track symbol
                else:  
                    rep_str += space
            # add line break
            rep_str += '\n'
        return rep_str
    # end repr() ----------------------------------------------------------------------------------------------------------------------------------------------

    # method for moving agent from one location to another ----------------------------------------------------------------------------------------------------
    def move(self, end):
        # use Bresenham's algorithm to find all possible spaces traversed during move
        spaces = bresenham_line(self.agent.location, end)
        # track agent's location throughout move
        ending_position = self.agent.location
        # flag for reaching destination
        reached_end = True
        # for each location in Bresenham line
        for space in spaces:
            # find symbol for that space
            space_type = self.track[space[0]][space[1]]
            # if obstacle, use crash behavior
            if space_type == '#':
                # if agent rests at start, move to starting space
                if self.reset:
                    self.agent.location = choice(self.starts)
                # either way, zero velocity
                self.agent.velocity = [0,0]
                # signal that end of move was not completed
                reached_end = False
                # halt move 
                break
            # if reached finish line
            if space_type == 'F':
                # update location
                ending_position = space
                # track has been completed
                self.at_finish_line = True
                # halt move
                break
            # if the move hasn't stopped, track agent location
            ending_position = space
        # if the move was completed, located agent at end of move
        if reached_end:
            self.agent.location = ending_position
            # record where agent has been
            self.move_traces.append(ending_position)
    # end move() ----------------------------------------------------------------------------------------------------------------------------------------------
    # flag for track completion
    at_finish_line = False
    # maintain list of agent's locations
    move_traces = []
# end Track class ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# class to maintain MDP data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# generates Value and Policy matrix based on Track object
class MDP:
    # initialize with a Track object ------------------------------------------------------------------------------
    def __init__(self, track, algorithm = 'VI'):
        self.track = track
        # extract actions
        self.A = self.track.agent.actions
        # extract possible velocities
        self.S = self.track.agent.velocity_states
        # extract possible locations
        self.L = self.track.valid_locs
        if algorithm == 'VI':
            # intialize Value matrix with locations and velocity states to -1
            self.V = pd.DataFrame(data= -0.001, index= self.S, columns= self.L, dtype=object)
        if algorithm == 'SARSA':
            # intialize Value matrix with locations and velocity states to -1
            self.V = pd.DataFrame(data= -0.0001, index= self.S, columns= self.L, dtype=float)
        # ending states in value matrix to 0
        for loc in self.track.finishes:
            self.V[loc] = 0
        # initialize Policy matrix with locations and velocity states
        self.Pi = pd.DataFrame(data='None', index=self.S, columns=self.L)   
    # end init() -------------------------------------------------------------------------------------------------

    # initialize with a Track object ------------------------------------------------------------------------------
    def run_simulation(self, animate=False):
        self.track.agent.location = choice(self.track.starts)
        finished = False
        i = 0
        # start loop, count timesteps
        while not finished:
            # print out track at each step
            if animate:
                clear()
                print(self.track)
                sleep(0.25)
            # look up appropriate action
            loc = self.track.agent.location
            vel = tuple(self.track.agent.velocity)
            if self.Pi.at[vel,loc] != 'None':
                action = self.Pi.at[vel,loc]
            else:
                action = choice(self.A)
            # apply action
            self.track.agent.accelerate(action)
            # find destination
            new_vel = tuple(self.track.agent.velocity)
            target_location = (loc[0]+new_vel[0], loc[1]+new_vel[1])
            # move to destination
            self.track.move(target_location)
            # update loop
            finished = self.track.at_finish_line
            if finished:
                print(f"Made it in {i} steps!")
            i +=1
        # print final track with trace
        print(self.track)
    # end run_simulation() -------------------------------------------------------------------------------------------
# end MDP class ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#************************************
# METHODS
#************************************
# method used Bresenham's algorithm to produce a list of coordinates between start and end points on the track ---------------------------------------------
def bresenham_line(start, end):
    # extract starting and ending coordiantes
    x0 = start[0]
    y0 = start[1]
    x1 = end[0]
    y1 = end[1]
    # calculate m values
    delta_x = x1 - x0
    delta_y = y1 - y0
 
    # determine if slope is too high to iterate over Xs
    high_m = abs(delta_y) > abs(delta_x)
    # swap Xs and Ys if need to iterate over Y vals for steep function
    if high_m:
        # simultaneuos variable reassignment
        x0, y0 = y0, x0
        x1, y1 = y1, x1
 
    # if negative vector, iterate in reverse
    negative_m = False
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        negative_m = True
 
    # recalculate m values
    delta_x = x1 - x0
    delta_y = y1 - y0
    # find error value
    e = int(delta_x / 2.0)
    # choose which direction it iterate Y values
    y_step = 1 if y0 < y1 else -1
 
    # iterate over X values bounding box generating points between start and end
    y = y0
    spaces = []
    for x in range(x0, x1 + 1):
        # find full coordinate in grid, reversed if iterating over Y values
        coord = (y, x) if high_m else (x, y)
        spaces.append(coord)
        # update error value
        e -= abs(delta_y)
        if e < 0:
            # change row
            y += y_step
            e += delta_x
    # if the coordinates were reversed, restore to true order
    if negative_m:
        spaces.reverse()
    return spaces
# end bresenham_line() -------------------------------------------------------------------------------------------------------------------------------------

# method for determining where velocity update would land ----------------------------------------------------------------
def find_s_prime(l,v,S):
    # find target location by applying velocity to location
    target = (l[0]+v[0], l[1]+v[1])
    # generate all points along line
    b_line = bresenham_line(l,target)
    # set location
    cur_step = l
    # check each step along line
    for b in b_line:
        # if valid step, make step
        if b in S:
            cur_step = b
        else:
            # otherwise stop and return final location
            return cur_step
    return cur_step
# end find_s_prime() ---------------------------------------------------------------------------------------------------

# method keeps velocity values within range of -5 to 5 -----------------------------------------------------------------
def constrain_velocity(velocity,a):
    # new_v = [velocity[0]+a[0], velocity[1]+a[1]]
    new_v = [sum(x) for x in zip(velocity,a)]
    # for each velocity coordinate (only two)
    for n in range(len(new_v)):
        # extract value
        x = new_v[n]
        # if greater than 5, cap at 5
        if x > 5:
            new_v[n] = 5
        # if less than -5, increase to -5
        elif x < -5:
            new_v[n] = -5
    return tuple(new_v)
# end constrain_velocity() ---------------------------------------------------------------------------------------------
