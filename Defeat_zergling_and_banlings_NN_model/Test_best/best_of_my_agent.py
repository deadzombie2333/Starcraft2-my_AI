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
_NUM_POPULATION_SIZE = 20

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
  Hostile_HP = 0
  Friendly_HP = 0
  Score = 0

  def step(self, obs):
    super(Attack_Zerg, self).step(obs)
    player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
    player_type = obs.observation["screen"][_UNIT_TYPE]
    max_x, max_y = player_relative.shape
    friendly_num = numpy.ceil((player_relative == _PLAYER_FRIENDLY).sum()/10)
    hostile_num = numpy.ceil((player_relative == _PLAYER_HOSTILE).sum()/10)
    player_HP = obs.observation["screen"][_UNIT_HIT_POINTS]
    hostile_HP = 0
    for item_1 in range(player_relative.shape[0]):
      for item_2 in range(player_relative.shape[1]):
        if player_relative[item_1][item_2] == _PLAYER_HOSTILE:
          hostile_HP += player_HP[item_1][item_2]
    
    if friendly_num > 1 and hostile_num > 1:
      self.Stage_3 = False
      self.Stage_4 = False
      if hostile_HP == self.Hostile_HP:
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

    self.Hostile_HP = hostile_HP
    
    if self.Stage_1:
      self.Attack = True
      self.Move = False
    
    if self.Stage_2:
      if self.Attack:
        self.Attack = False
        self.Move = True
      else:
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

    if self.Attack: 
      if _ATTACK_SCREEN in obs.observation["available_actions"]:
        temp_threat_matrix_a_0 = _THREAT_MATRIX_A_0[self.counter1]
        temp_threat_matrix_a_1 = _THREAT_MATRIX_A_1[self.counter1]
        player_shield = obs.observation["screen"][_UNIT_SHIELD]
        friendly_info = numpy.array([[],[],[],[],[]])
        hostile_info = numpy.array([[],[],[],[],[]])
        distinct_unit = numpy.unique(player_type)
        distinct_unit = numpy.delete(distinct_unit,[0])
        for unit in distinct_unit:
          unit_y, unit_x = (player_type == unit).nonzero()
          unit_vector = unit*numpy.ones(len(unit_y))
          unit_relation = player_relative[unit_y,unit_x]
          unit_HP = player_HP[unit_y,unit_x]
          unit_shield = player_shield[unit_y,unit_x]
          unit_matrix = numpy.vstack(([[unit_y],[unit_x],[unit_vector],[unit_HP],[unit_shield]]))
          if all (unit_relation == _PLAYER_HOSTILE):
            hostile_info = numpy.hstack((hostile_info,unit_matrix))
          else:
            friendly_info = numpy.hstack((friendly_info,unit_matrix))
        threat = numpy.zeros(hostile_info.shape[1])
        for i in range(hostile_info.shape[1]):
          for j in range(friendly_info.shape[1]):
            friendly_dps = (6-0)/0.61
            if hostile_info[2][i] == 9:
              hostile_dps = (35-0)
            else:
              hostile_dps = (10-0)/0.497
            unit_distance = (hostile_info[0][i]/max_y - friendly_info[0][j]/max_y)**2
            unit_distance += (hostile_info[1][i]/max_x - friendly_info[1][j]/max_x)**2
            unit_distance = math.sqrt(unit_distance)
            unit_health = hostile_info[3][i]+hostile_info[4][i]
            friendly_hostile_vector = numpy.array([friendly_dps,hostile_dps,unit_distance,unit_health])
            NN_level_1 = numpy.dot(friendly_hostile_vector,temp_threat_matrix_a_0) + temp_threat_matrix_a_1
            NN_level_active_1 = NN_level_1 * (NN_level_1 > 0)
            threat[i] += numpy.sum(NN_level_active_1)
        highest_threat = numpy.argmax(threat)
        target_y, target_x = hostile_info[:,highest_threat][:2]
        target = [target_x,target_y]
        return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
      elif _SELECT_ARMY in obs.observation["available_actions"]:
        return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

    if self.Move:
      if _MOVE_SCREEN in obs.observation["available_actions"]: 
        """ use neural network to find location to move"""
        temp_move_matrix_a_0 = _MOVE_MATRIX_A_0[self.counter1]
        temp_move_matrix_a_1 = _MOVE_MATRIX_A_1[self.counter1]
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
          unit_vector = unit*numpy.ones(len(unit_y))
          unit_relation = player_relative[unit_y,unit_x]
          unit_HP = player_HP[unit_y,unit_x]
          unit_shield = player_shield[unit_y,unit_x]
          unit_matrix = numpy.vstack(([[unit_y],[unit_x],[unit_vector],[unit_HP],[unit_shield]]))
          if all (unit_relation == _PLAYER_HOSTILE):
            hostile_info = numpy.hstack((hostile_info,unit_matrix))
          else:
            friendly_info = numpy.hstack((friendly_info,unit_matrix))
        friendly_health = numpy.mean(friendly_info[3] + friendly_info[4])
        hostile_health = numpy.mean(hostile_info[3] + hostile_info[4])
        hostile_y_mean = numpy.mean(hostile_info[0])
        hostile_x_mean = numpy.mean(hostile_info[1])
        friendly_y_mean = numpy.mean(friendly_info[0])
        friendly_x_mean = numpy.mean(friendly_info[1])
        move_vector = numpy.array([hostile_y_mean,hostile_x_mean,friendly_y_mean,friendly_x_mean,friendly_health,hostile_health])
        NN_level_1 = numpy.dot(move_vector, temp_move_matrix_a_0) + temp_move_matrix_a_1 #a_0 6*2, a_1 1*2
        NN_level_2 = sigmoid(NN_level_1)
        target = numpy.around(numpy.multiply(NN_level_2,numpy.array([max_x-2,max_y-2]))+numpy.array([1,1])).ravel()
        self.Hostile_HP = hostile_health
        return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
      elif _SELECT_ARMY in obs.observation["available_actions"]:
        return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
      
    if self.Record:
      self.Stage_3 = False
      self.Stage_4 = True
      if friendly_num < hostile_num:
        self.Score = self.Score + 5*(10 - hostile_num) - (9 - friendly_num)
        score_combined = numpy.load('score.npy')
        score_combined[self.counter1] = self.Score
        numpy.save('score',score_combined)
        print("simulation {} score {}".format(self.counter1,self.Score))
        self.counter1 = self.counter1 - 1
        self.Score = 0
        self.Hostile_HP = 0
      else:
        self.Score = self.Score + 5*(10 - hostile_num) - (9 - friendly_num)
      return actions.FunctionCall(_NO_OP, [])
