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

_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id 
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id 
_SELECT_POINT = actions.FUNCTIONS.select_point.id 
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id 
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id 
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_HARVEST = actions.FUNCTIONS.Harvest_Gather_screen.id
_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_SELECT_UNIT = features.SCREEN_FEATURES.selected.index

_GROUP_ACT_APPEND = 2
_GROUP_ACT_SET = 1
_GROUP_ACT_RECALL = 0

_MINIMAP_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

_PLAYER_FRIENDLY = 1
_IDLE_WORKER = 7
_PLAYER_NEUTRAL = 3  # beacon/minerals
_SUPPLY_MAX = 4
_ARMY_SUPPLY = 5
_WORKER_SUPPLY = 6
_MINERALS = [1] 
_ALL_TYPE = [2]


_NEUTRAL_MINERALFIELD = 341# size 8 * 8
_TERRAN_BARRACKS = 21 # size 12 * 12
_TERRAN_COMMANDCENTER = 18 # 18 * 18
_TERRAN_SUPPLYDEPOT = 19 # size 4 * 4
_TERRAN_SCV = 45 # 3 * 3
_SELECTED = 1

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [0]
_SCREEN = [0]
_MINIMAP = [1]
_SELECT_UNIT_ID = [5]
_SUPPLY_USED = [3]

_THRES_VECTOR = numpy.load("Thres_vector.npy").item()
_MATRIX_A_0 = numpy.load('Matrix_a_0.npy').item()
_MATRIX_A_1 = numpy.load('Matrix_a_1.npy').item()
_NUM_POPULATION_SIZE = 5

def sigmoid(x):
  """x is a vector input"""
  new_x = numpy.zeros((x.shape))
  for item_1 in range(x.shape[0]):
    gamma = x[item_1]
    if gamma > 0:
      new_x[item_1] = 1 / (1 + math.exp(-gamma))
    else:
      new_x[item_1] = 1 - 1 / (1 + math.exp(gamma))
  return new_x

class Build_Marine(base_agent.BaseAgent):
  """ simoutaneous run 10 candidate in 10 simulations, gather their score in score.txt file.
  then, use GA to find elite candidate, candidate mutation and crossover, further test these
  candidate in future simulations until a global optimal candidate is located"""
  command_hold = False
  select_barrack = False
  select_commandcenter = False
  select_SCV = False
  build_barrack = False
  build_supplydepot = False
  add_barrack_to_group = False
  barrack_grouped = False
  current_command = 0
  Score = 0
  depot_x = 5
  depot_y = 5
  barrack_x = 47
  barrack_y = 47
  barrack_num = 0
  COUNTER1 = _NUM_POPULATION_SIZE - 1
  Simu_init = False
  Simu_record = False
  pre_structure = 0
  
  def step(self, obs):
    super(Build_Marine, self).step(obs)
    Supply_used = obs.observation["player"][_SUPPLY_USED]
    Supply_max = obs.observation["player"][_SUPPLY_MAX]
    minerals = obs.observation["player"][_MINERALS]
    unit_type = obs.observation["screen"][_UNIT_TYPE]
    
    max_y, max_x = unit_type.shape
    distinct_unit = numpy.unique(unit_type)
    distinct_unit = numpy.delete(distinct_unit,[0])
    army_supply = obs.observation["player"][_ARMY_SUPPLY]
    worker_supply = obs.observation["player"][_WORKER_SUPPLY]
    collected_mineral = obs.observation["score_cumulative"][[7]]
    structure_value = obs.observation["score_cumulative"][[4]]
    
    if self.Score == 0 and not self.Simu_init:
      self.Simu_init = True

    if self.Simu_init and collected_mineral != 0:
      self.Score = [army_supply,worker_supply]
      self.Simu_record = True
      
    if self.Simu_init and self.Simu_record and collected_mineral == 0:
      structure_value = obs.observation["score_cumulative"][[4]] - 400
      #record to file
      score_combined = numpy.load('score.npy')
      score_combined[self.COUNTER1,:] = numpy.hstack(([self.Score,structure_value - self.pre_structure]))
      self.pre_structure = structure_value
      numpy.save('score',score_combined)
      print("simulation {} score {}".format(self.COUNTER1,score_combined[self.COUNTER1,:]))
      self.COUNTER1 += -1
      self.Score = 0
      self.Simu_init = False
      self.Simu_record = False
      self.command_hold = False
      self.select_barrack = False
      self.select_commandcenter = False
      self.select_SCV = False
      self.build_barrack = False
      self.build_supplydepot = False
      self.add_barrack_to_group = False
      self.barrack_grouped = False
      self.current_command = 0
      self.depot_x = 5
      self.depot_y = 5
      self.barrack_x = 47
      self.barrack_y = 47
      self.barrack_num = 0
      if self.COUNTER1 < 0:
        quit()

    minefield_y, minefield_x = (unit_type == _NEUTRAL_MINERALFIELD).nonzero()
    barrack_y, barrack_x = (unit_type == _TERRAN_BARRACKS).nonzero()
    barrack_obs = math.floor(len(barrack_y)/121)
    supplydepot_y, supplydepot_x = (unit_type == _TERRAN_SUPPLYDEPOT).nonzero()
    supply_obs = math.floor(len(supplydepot_y)/64)
    
    if barrack_obs > self.barrack_num:
      self.barrack_num = barrack_obs
      self.add_barrack_to_group = True
    
    if not self.command_hold:
      if self.current_command == 0:
        _vector_a_0 = _THRES_VECTOR[self.COUNTER1]
        _matrix_a_0 = _MATRIX_A_0[self.COUNTER1]
        _matrix_a_1 = _MATRIX_A_1[self.COUNTER1]
        #determine what to build (army_supply, worker_supply, mineral_mines, minerals, max_supply, barrack_number)
        info_vector = numpy.array([worker_supply/8, supply_obs/Supply_used, self.barrack_num/worker_supply, 50/minerals])
        Thres_vector = numpy.append(_vector_a_0,worker_supply)     # _vector_a_0 1*4
        NN_level_1 = numpy.dot(Thres_vector,_matrix_a_0) + _matrix_a_1 # _matrix_a_0 5*4 _matrix_a_1 1*4
        NN_level_active_1 = NN_level_1 * (NN_level_1 > 0) + 0.01
        NN_level_2 = numpy.divide(info_vector,NN_level_active_1)
        if barrack_y.any():
          NN_decision = numpy.argmin(NN_level_2)
        elif supplydepot_y.any():
          NN_decision = numpy.argmin(NN_level_2[:3])
        else:
          NN_decision = numpy.argmin(NN_level_2[:2])
        
        if NN_level_2[0][NN_decision] >= 1:
          self.current_command = 0
        else:
          self.current_command = NN_decision + 1
      else:
        self.current_command = 0

    if self.current_command == 0: #1, idle worker. 2, group barrack
      if self.add_barrack_to_group:
        if not self.command_hold:
          self.command_hold = True
          random_location = numpy.random.choice(len(barrack_y),1)
          ba_target = [barrack_x[random_location] ,barrack_y[random_location]]
          return actions.FunctionCall(_SELECT_POINT, [_ALL_TYPE, ba_target])
        else:
          select_y, select_x = (obs.observation["screen"][_SELECT_UNIT] == _SELECTED).nonzero()
          if select_y.any():
            select_type = unit_type[select_y[0]][select_x[0]]
          else:
            select_type = 0
          if select_type == _TERRAN_BARRACKS:
            self.command_hold = False
            self.barrack_grouped = True
            self.add_barrack_to_group = False
            return actions.FunctionCall(_CONTROL_GROUP, [[_GROUP_ACT_SET] , [3]]) 
          else:
            self.command_hold = True
            random_location = numpy.random.choice(len(barrack_y),1)
            ba_target = [barrack_x[random_location] ,barrack_y[random_location]]
            return actions.FunctionCall(_SELECT_POINT, [_ALL_TYPE, ba_target])

      elif obs.observation["player"][_IDLE_WORKER] > 0:
        if not self.command_hold:
          self.command_hold = True
          return actions.FunctionCall(_SELECT_IDLE_WORKER, [[2]])
        else:
          self.command_hold = False
          m_index = numpy.random.choice(len(minefield_y),1)
          return actions.FunctionCall(_HARVEST, [_SCREEN, [minefield_x[m_index],minefield_y[m_index]]])
      
      else:
        return actions.FunctionCall(_NO_OP, [])
      
    if self.current_command == 4: #build marine
      if Supply_used >= Supply_max:
        self.command_hold = False
        return actions.FunctionCall(_NO_OP, [])
      else :
        if _TRAIN_MARINE in obs.observation['available_actions']:
          self.command_hold = False
          return actions.FunctionCall(_TRAIN_MARINE, [_NOT_QUEUED])
        else:
          self.select_barrack = True
          self.command_hold = True
    
    if self.current_command == 1: #build SCV
      if Supply_used >= Supply_max:
        self.command_hold = False
        return actions.FunctionCall(_NO_OP, [])
      else:
        if _TRAIN_SCV in obs.observation['available_actions']:
          self.command_hold = False
          return actions.FunctionCall(_TRAIN_SCV, [_NOT_QUEUED])
        else:
          self.select_commandcenter = True
          self.command_hold = True
        
    if self.current_command == 2: #build supply depot
      if _BUILD_SUPPLYDEPOT in obs.observation['available_actions']:
        self.command_hold = False
        self.build_supplydepot = True
      else:
        self.select_SCV = True
        self.command_hold = True
        
    if self.current_command == 3: #build barrack
      if _TERRAN_SUPPLYDEPOT in unit_type:
        if  _BUILD_BARRACKS in obs.observation['available_actions']:
          self.command_hold = False
          self.build_barrack = True
        else:
          self.select_SCV = True
          self.command_hold = True
      else:
        return actions.FunctionCall(_NO_OP, [])
        
    if self.select_SCV:
      self.select_SCV = False
      if obs.observation["player"][_IDLE_WORKER] > 0:
        return actions.FunctionCall(_SELECT_IDLE_WORKER, [[2]])
      else:
        unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
        SCV_lineup = numpy.random.choice(len(unit_y),1)
        ss_target = [unit_x[SCV_lineup], unit_y[SCV_lineup]]
        return actions.FunctionCall(_SELECT_POINT, [_SCREEN, ss_target])
    
    if self.select_commandcenter:
      unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
      self.select_commandcenter = False
      if unit_y.any():
        cs_target = [int(unit_x.mean()), int(unit_y.mean())]
        return actions.FunctionCall(_SELECT_POINT, [_SCREEN, cs_target])
      else:
        self.command_hold = False
        return actions.FunctionCall(_NO_OP, [])
        
    if self.select_barrack:
      self.select_barrack = False
      if self.barrack_grouped:
        return actions.FunctionCall(_CONTROL_GROUP, [[_GROUP_ACT_RECALL] ,[3]])
      else:
        if barrack_y.any():
          random_location = numpy.random.choice(len(barrack_y),1)
          bs_target = [barrack_x[random_location] ,barrack_y[random_location]]
          return actions.FunctionCall(_SELECT_POINT, [_SCREEN, bs_target])
        else:
          self.command_hold = False
          return actions.FunctionCall(_NO_OP, [])
        
    if self.build_supplydepot:
      self.build_supplydepot = False
      distance = 6
      chance = 10
      while True:
        s_target_x = int(self.depot_x + numpy.random.choice([-1,0,1],1)*distance)
        s_target_y = int(self.depot_y + numpy.random.choice([-1,0,1],1)*distance)
        within_map = (0 < s_target_x < max_x-4) and (0 < s_target_y < max_y-4)
        buildings = [_NEUTRAL_MINERALFIELD, _TERRAN_BARRACKS, _TERRAN_COMMANDCENTER, _TERRAN_SUPPLYDEPOT]
        area = unit_type[s_target_y:s_target_y + 6][s_target_x:s_target_x + 6]
        space_available = not any(x in area for x in buildings)
        within_mineral_field = (min(minefield_y) < s_target_y < max(minefield_y)) and (min(minefield_x) < s_target_x < 11+max(minefield_x))
        chance += -1
        if within_map and space_available and not within_mineral_field:
          self.depot_x = s_target_x
          self.depot_y = s_target_y
          s_target = numpy.array([s_target_x,s_target_y])
          break
        if chance == 0:
          distance += 1
          chance = 10
      print("build supply depot at {}".format(s_target))
      return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_SCREEN, s_target])
        
    if self.build_barrack:
      self.build_barrack = False
      distance = 13
      chance = 10
      while True:
        b_target_x = int(self.barrack_x + numpy.random.choice([-1,0,1],1)*distance)
        b_target_y = int(self.barrack_y + numpy.random.choice([-1,0,1],1)*distance)
        within_map = (0 < b_target_x < max_x-11) and (0 < b_target_y < max_y-11)
        buildings = [_NEUTRAL_MINERALFIELD, _TERRAN_BARRACKS, _TERRAN_COMMANDCENTER, _TERRAN_SUPPLYDEPOT]
        area = unit_type[b_target_y:b_target_y + 13][b_target_x:b_target_x + 13]
        space_available = not any(x in area for x in buildings)
        within_mineral_field = (min(minefield_y) < b_target_y < max(minefield_y)) and (min(minefield_x) < b_target_x < 11+max(minefield_x))
        chance += -1
        if within_map and space_available and not within_mineral_field:
          self.barrack_x = b_target_x
          self.barrack_y = b_target_y
          b_target = numpy.array([b_target_x,b_target_y])
          break
        if chance == 0:
          distance += 1
          chance = 10
      print("build barrack at {}".format(b_target))
      return actions.FunctionCall(_BUILD_BARRACKS, [_SCREEN, b_target])

