from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import random
import math
import os
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
_UNIT_SHIELD = features.SCREEN_FEATURES.unit_shields.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [0]
_THREAT_MATRIX_A_0 = numpy.load('best_threat_a_0.npy').item()
_THREAT_MATRIX_A_1 = numpy.load('best_threat_a_1.npy').item()
_MOVE_MATRIX_A_0 = numpy.load('best_move_a_0.npy').item()
_MOVE_MATRIX_A_1 = numpy.load('best_move_a_1.npy').item()
_NUM_POPULATION_SIZE = 1
_NUM_TESTS = 5

_ZERG_BANELINGS = 9
_ZERG_ZERGLINGS = 105
_TERRAN_MARINE = 48

def sigmoid(x):
  """x is a vector input"""
  new_x = numpy.zeros((x.shape))
  for item_1 in range(x.shape[0]):
    for item_2 in range(x.shape[1]):
      gamma = int(x[item_1][item_2])
      if gamma > 0:
        new_x[item_1][item_2] = 1 / (1 + math.exp(-gamma))
      else:
        new_x[item_1][item_2] = 1 - 1 / (1 + math.exp(gamma))
  return new_x

def distance(x1,y1,x2,y2):
  return math.sqrt((x1-x2)**2 +(y1-y2)**2)
  
class Attack_Zerg(base_agent.BaseAgent):
  """ simoutaneous run 10 candidate in 10 simulations, gather their score in score.txt file.
  then, use GA to find elite candidate, candidate mutation and crossover, further test these
  candidate in future simulations until a global optimal candidate is located"""
  counter1 = _NUM_POPULATION_SIZE - 1
  Stage_1 = True
  Stage_2 = False
  Stage_3 = False
  Stage_4 = False
  Attack = False
  Move = False
  Record = False
  score = 0
  Hostile_HP = 0
  Hostile_num = 0
  tests = _NUM_TESTS - 1
  Friendly_location = numpy.array([0.1,0.1])

  def step(self, obs):
    super(Attack_Zerg, self).step(obs)
    player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
    player_type = obs.observation["screen"][_UNIT_TYPE]
    max_x, max_y = player_relative.shape
    friendly_num = numpy.ceil((player_relative == _PLAYER_FRIENDLY).sum()/10)
    friendly_location = numpy.mean((player_relative == _PLAYER_FRIENDLY).nonzero(),axis = 1)
    hostile_num = numpy.ceil((player_relative == _PLAYER_HOSTILE).sum()/10)
    player_HP = obs.observation["screen"][_UNIT_HIT_POINTS]
    hostile_HP = 0
    for item_1 in range(player_relative.shape[0]):
      for item_2 in range(player_relative.shape[1]):
        if player_relative[item_1][item_2] == _PLAYER_HOSTILE:
          hostile_HP += player_HP[item_1][item_2]
    
    if friendly_num > 1 and hostile_num > 2:
      self.Stage_3 = False
      self.Stage_4 = False
      if hostile_HP == self.Hostile_HP:
        self.Hostile_num = hostile_num
        self.Friendly_location = friendly_location
        self.Stage_1 = True
        self.Stage_2 = False
      else:
        self.Stage_1 = False
        self.Stage_2 = True
    else:
      if (not self.Stage_3) and (not self.Stage_4):
        self.Stage_3 = True
        self.Stage_2 = False
        self.Stage_1 = False

    if self.Stage_1:
      self.Attack = True
      self.Move = False
    
    if self.Stage_2:
      if self.Attack and hostile_num < self.Hostile_num:
        self.Attack = False
        self.Move = True
        self.Friendly_location = friendly_location
      
      move_distance = distance(self.Friendly_location[0],self.Friendly_location[1],friendly_location[0],friendly_location[1])
      if self.Move and move_distance > 2.5:
        self.Hostile_num = hostile_num
        self.Move = False
        self.Attack = True
    
    if self.Stage_4:
      self.Record = False
      return actions.FunctionCall(_NO_OP, [])
      
    if self.Stage_3:
      self.Attack = False
      self.Move = False
      self.Record = True
      self.Stage_3 = False
      self.Stage_4 = True
    
    self.Hostile_HP = hostile_HP
    player_shield = obs.observation["screen"][_UNIT_SHIELD]
    friendly_info = numpy.array([[],[],[],[],[]])
    hostile_info = numpy.array([[],[],[],[],[]])
    distinct_unit = numpy.unique(player_type)
    distinct_unit = numpy.delete(distinct_unit,[0])
    for unit in distinct_unit:
      unit_y, unit_x = (player_type == unit).nonzero()
      if len(unit_y) > 10:
        select_index = random.sample(range(1,len(unit_y)),10)
        unit_y = unit_y [select_index]
        unit_x = unit_x [select_index]
      if unit == _ZERG_BANELINGS:
        unit_dps = (35-0)*numpy.ones(len(unit_y))
      elif unit == _ZERG_ZERGLINGS:
        unit_dps = (10-0)/0.497*numpy.ones(len(unit_y))
      elif unit == _TERRAN_MARINE:
        unit_dps = (6-0)/0.61*numpy.ones(len(unit_y))
      unit_relation = player_relative[unit_y,unit_x]
      unit_HP = player_HP[unit_y,unit_x]
      unit_shield = player_shield[unit_y,unit_x]
      unit_matrix = numpy.vstack(([[unit_y],[unit_x],[unit_dps],[unit_HP],[unit_shield]]))
      if all (unit_relation == _PLAYER_HOSTILE):
        hostile_info = numpy.hstack((hostile_info,unit_matrix))
      else:
        friendly_info = numpy.hstack((friendly_info,unit_matrix))
    threat = numpy.zeros(hostile_info.shape[1])
    temp_threat_matrix_a_0 = _THREAT_MATRIX_A_0[self.counter1]
    temp_threat_matrix_a_1 = _THREAT_MATRIX_A_1[self.counter1]
    
    for i in range(hostile_info.shape[1]):
      for j in range(friendly_info.shape[1]):
        friendly_dps = friendly_info[2][j]
        hostile_dps = hostile_info[2][i]
        unit_distance = distance(hostile_info[1][i],hostile_info[0][i],friendly_info[1][j],friendly_info[0][j])/max_y
        unit_health = hostile_info[3][i]+hostile_info[4][i]
        friendly_hostile_vector = numpy.array([friendly_dps,hostile_dps,unit_distance,unit_health])
        NN_level_1 = numpy.dot(friendly_hostile_vector,temp_threat_matrix_a_0) + temp_threat_matrix_a_1
        NN_level_active_1 = NN_level_1 * (NN_level_1 > 0)
        threat[i] += numpy.sum(NN_level_active_1)
    highest_threat = numpy.argmax(threat)
    high_threat_y, high_threat_x = hostile_info[:,highest_threat][:2]
    
    if self.Attack: 
      if _ATTACK_SCREEN in obs.observation["available_actions"]:
        target = [high_threat_x,high_threat_y]
        return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
      elif _SELECT_ARMY in obs.observation["available_actions"]:
        return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

    if self.Move:
      if _MOVE_SCREEN in obs.observation["available_actions"]: 
        """ use neural network to find location to move"""
        temp_move_matrix_a_0 = _MOVE_MATRIX_A_0[self.counter1]
        temp_move_matrix_a_1 = _MOVE_MATRIX_A_1[self.counter1]
        friendly_health = numpy.mean(friendly_info[3] + friendly_info[4])
        hostile_health = numpy.mean(hostile_info[3] + hostile_info[4])
        friendly_dps = numpy.mean(friendly_info[2])
        hostile_dps = numpy.mean(hostile_info[2])
        move_vector = numpy.array([hostile_health,friendly_health,hostile_dps,friendly_dps])
        NN_level_1 = numpy.dot(move_vector, temp_move_matrix_a_0) + temp_move_matrix_a_1 #a_0 4*2, a_1 1*2
        NN_decision = numpy.argmax(NN_level_1)
        
        if NN_decision == 0: 
          move_rate = 1 # retreat
        else:
          move_rate = 0 # hold position
        hostile_y_mean = numpy.mean(hostile_info[0])
        hostile_x_mean = numpy.mean(hostile_info[1])
        friendly_y_mean = numpy.mean(friendly_info[0])
        friendly_x_mean = numpy.mean(friendly_info[1])
        target_x = friendly_x_mean + numpy.sign(friendly_x_mean-hostile_x_mean) * 5 * move_rate
        target_y = friendly_y_mean + numpy.sign(friendly_y_mean-hostile_y_mean) * 5 * move_rate
        if target_x < 1:
          target_x = 1
        if target_x > 83:
          target_x = 83
        if target_y < 1:
          target_y = 1
        if target_y > 83:
          target_y = 83
        target = numpy.around([target_x,target_y])
        return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
      elif _SELECT_ARMY in obs.observation["available_actions"]:
        return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
      
    if self.Record:
      self.Stage_3 = False
      self.Stage_4 = True
      if friendly_num < hostile_num:
        score_combined = numpy.load('score.npy')
        score_combined[self.counter1][self.tests] = obs.observation["score_cumulative"][[0]]
        numpy.save('score',score_combined)
        #print("simulation {} score {}".format(self.counter1,score_combined[self.counter1]))
        print(obs.observation["score_cumulative"][[0]])
        if self.tests == 0:
          self.counter1 = self.counter1 - 1
          self.tests = _NUM_TESTS - 1
        else:
          self.tests += -1
        self.Hostile_HP = 0
        self.Hostile_num = 0
        self.Friendly_location = numpy.array([0,0])
  
        if self.counter1 < 0:
          quit()
      else:
        current_score = obs.observation["score_cumulative"][[0]]
        if self.score > current_score:
          score_combined = numpy.load('score.npy')
          score_combined[self.counter1][self.tests] = self.score
          numpy.save('score',score_combined)
          #print("simulation {} score {}".format(self.counter1,score_combined[self.counter1]))
          print(self.score)
          self.score = 0
          if self.tests == 0:
            self.counter1 = self.counter1 - 1
            self.tests = _NUM_TESTS - 1
          else:
            self.tests += -1
          self.Hostile_HP = 0
          self.Hostile_num = 0
          self.Friendly_location = numpy.array([0,0])
          if self.counter1 < 0:
            quit()
        else:
          self.score = current_score
      
    return actions.FunctionCall(_NO_OP, [])
