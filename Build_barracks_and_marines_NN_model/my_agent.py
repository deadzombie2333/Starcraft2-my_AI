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
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

_SELECT_UNIT = features.SCREEN_FEATURES.selected.index
_MINIMAP_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

_GROUP_ACT_APPEND = 2
_GROUP_ACT_SET = 1
_GROUP_ACT_RECALL = 0
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
_TERRAN_COMMANDCENTER = 18 # 16 * 16
_TERRAN_SUPPLYDEPOT = 19 # size 8 * 8
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
_NUM_POPULATION_SIZE = 8
_NUM_MINERALMINES = 8
_NUM_TESTS = 2

class Build_Marine(base_agent.BaseAgent):
  """ simoutaneous run 10 candidate in 10 simulations, gather their score in score.txt file.
  then, use GA to find elite candidate, candidate mutation and crossover, further test these
  candidate in future simulations until a global optimal candidate is located"""
  command_hold = False
  sub_command = 0
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
  depot_num = 0
  barrack_x = 47
  barrack_y = 47
  barrack_num = 0
  COUNTER1 = _NUM_POPULATION_SIZE - 1
  Simu_init = False
  Simu_record = False
  tests = _NUM_TESTS - 1
  
  def step(self, obs):
    super(Build_Marine, self).step(obs)
    Supply_used = obs.observation["player"][_SUPPLY_USED]
    Supply_max = obs.observation["player"][_SUPPLY_MAX]
    minerals = obs.observation["player"][_MINERALS]
    unit_type = obs.observation["screen"][_UNIT_TYPE]
    
    max_y, max_x = unit_type.shape
    army_supply = obs.observation["player"][_ARMY_SUPPLY]
    worker_supply = obs.observation["player"][_WORKER_SUPPLY]
    collected_mineral = obs.observation["score_cumulative"][[7]]
    
    if self.Score == 0 and not self.Simu_init:
      self.Simu_init = True

    if self.Simu_init and collected_mineral != 0:
      self.Score = obs.observation["score_cumulative"][[0]]
      self.Simu_record = True
      
    if self.Simu_init and self.Simu_record and collected_mineral == 0:
      #record to file
      score_combined = numpy.load('score.npy')
      score_combined[self.COUNTER1][self.tests] = self.Score
      numpy.save('score',score_combined)
      if self.tests == 0:
        print("simulation {} score {}".format(self.COUNTER1 + 1,score_combined[self.COUNTER1,:]))
        self.COUNTER1 = self.COUNTER1 - 1
        self.tests = _NUM_TESTS - 1
      else:
        self.tests += -1
      if self.COUNTER1 < 0:
        quit()
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
      self.sub_command = 0
      self.depot_x = 5
      self.depot_y = 5
      self.depot_num = 0
      self.barrack_x = 47
      self.barrack_y = 47
      self.barrack_num = 0
      return actions.FunctionCall(_NO_OP, [])

    minefield_y, minefield_x = (unit_type == _NEUTRAL_MINERALFIELD).nonzero()
    barrack_y, barrack_x = (unit_type == _TERRAN_BARRACKS).nonzero()
    barrack_obs = math.floor(len(barrack_y)/121)
    supplydepot_y, supplydepot_x = (unit_type == _TERRAN_SUPPLYDEPOT).nonzero()
    depot_obs = math.floor(len(supplydepot_y)/64)

    if barrack_obs > self.barrack_num and self.current_command == 3:
      self.command_hold = False
      self.barrack_num = barrack_obs
      self.add_barrack_to_group = True
    
    if depot_obs > self.depot_num and self.current_command == 2:
      self.command_hold = False
      self.depot_num = depot_obs
    
    if not self.command_hold:
      if self.current_command == 0:
        _vector_a_0 = _THRES_VECTOR[self.COUNTER1]
        #determine what to build (army_supply, worker_supply, minerals, max_supply, barrack_number)
        info_vector = numpy.array([worker_supply/_NUM_MINERALMINES, (15 + depot_obs * 8 - Supply_used)/(self.barrack_num + 1), (self.barrack_num + 1)/worker_supply])
        NN_level_2 = numpy.divide(info_vector,_vector_a_0)# _vector_a_0 1*4
        if depot_obs > 0:
          NN_decision = numpy.argmin(NN_level_2[0])
        else:
          NN_decision = numpy.argmin(NN_level_2[0][:2])
        if NN_level_2[0][NN_decision] < 1:
          self.current_command = NN_decision + 1
        else:  
          if barrack_obs > 1:
            self.current_command = 4
          else:
            self.current_command = 0
      else:
        self.current_command = 0
    
    if self.current_command == 0: 
      if self.add_barrack_to_group:
        if not self.command_hold:
          self.sub_command = 1
          self.command_hold = True
          random_location = numpy.random.choice(len(barrack_y),1)
          ba_target = [barrack_x[random_location] ,barrack_y[random_location]]
          return actions.FunctionCall(_SELECT_POINT, [_ALL_TYPE, ba_target])
        elif self.sub_command == 1:
          select_y, select_x = (obs.observation["screen"][_SELECT_UNIT] == _SELECTED).nonzero()
          if select_y.any():
            select_type = unit_type[select_y[0]][select_x[0]]
          else:
            select_type = 0
          if select_type == _TERRAN_BARRACKS:
            self.sub_command = 0
            self.command_hold = False
            self.barrack_grouped = True
            self.add_barrack_to_group = False
            return actions.FunctionCall(_CONTROL_GROUP, [[_GROUP_ACT_SET] , [3]]) 
          else:
            self.command_hold = True
            random_location = numpy.random.choice(len(barrack_y),1)
            ba_target = [barrack_x[random_location] ,barrack_y[random_location]]
            return actions.FunctionCall(_SELECT_POINT, [_ALL_TYPE, ba_target])

      if obs.observation["player"][_IDLE_WORKER] > 0:
        if not self.command_hold:
          self.sub_command = 2
          self.command_hold = True
          return actions.FunctionCall(_SELECT_IDLE_WORKER, [[2]])
        elif self.sub_command == 2:
          self.sub_command = 0
          self.command_hold = False
          m_index = numpy.random.choice(len(minefield_y),1)
          return actions.FunctionCall(_HARVEST, [_SCREEN, [minefield_x[m_index],minefield_y[m_index]]])
      
      if _SELECT_ARMY in obs.observation['available_actions']:
        if self.sub_command == 3: 
          if (_MOVE_SCREEN in obs.observation['available_actions']):
            self.command_hold = False
            self.sub_command = 0
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, numpy.array([1,1])])
          else:
            return actions.FunctionCall(_NO_OP, [])
          
        if numpy.random.uniform(low = 0, high = 1) < 0.3 and self.sub_command == 0:
          if not self.command_hold:
            self.sub_command = 3
            self.command_hold = True
            return actions.FunctionCall(_SELECT_ARMY, [[1]])
      
      if self.sub_command == 0:
        return actions.FunctionCall(_NO_OP, [])
    
    if self.current_command == 1: #build SCV
      if Supply_used < Supply_max  and _TERRAN_SCV not in obs.observation["build_queue"]:
        if _TRAIN_SCV in obs.observation['available_actions']:
          if minerals >= 50:
            self.command_hold = False
            return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])
          else:
            return actions.FunctionCall(_NO_OP, [])
        else:
          self.select_commandcenter = True
          self.command_hold = True
      else:
        self.command_hold = False
        return actions.FunctionCall(_NO_OP, [])
        
    if self.current_command == 2: #build supply depot
      if _BUILD_SUPPLYDEPOT in obs.observation['available_actions']:
        if minerals >= 100:
          self.build_supplydepot = True
        else:
          return actions.FunctionCall(_NO_OP, [])
      else:
        self.select_SCV = True
        self.command_hold = True
        
    if self.current_command == 3: #build barrack
      if _BUILD_BARRACKS in obs.observation['available_actions']:
        if minerals >= 150:
          self.build_barrack = True
        else:
          return actions.FunctionCall(_NO_OP, [])
      else:
        self.select_SCV = True
        self.command_hold = True
        
    if self.current_command == 4: #build marine
      if Supply_used < Supply_max:
        if _TRAIN_MARINE in obs.observation['available_actions']:
          if minerals >= 50:
            self.command_hold = False
            self.add_barrack_to_group = True
            return actions.FunctionCall(_TRAIN_MARINE, [_NOT_QUEUED])
          else:
            return actions.FunctionCall(_NO_OP, [])
        else:
          self.select_barrack = True
          self.command_hold = True
      else:
        self.command_hold = False
        return actions.FunctionCall(_NO_OP, [])
    
    if self.select_SCV: #select SCV
      self.select_SCV = False
      if obs.observation["player"][_IDLE_WORKER] > 0:
        return actions.FunctionCall(_SELECT_IDLE_WORKER, [[2]])
      else:
        unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
        SCV_lineup = numpy.random.choice(len(unit_y),1)
        ss_target = [unit_x[SCV_lineup], unit_y[SCV_lineup]]
        return actions.FunctionCall(_SELECT_POINT, [_SCREEN, ss_target])
    
    if self.select_commandcenter: #select command center
      unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
      self.select_commandcenter = False
      if unit_y.any():
        cs_target = [int(unit_x.mean()), int(unit_y.mean())]
        return actions.FunctionCall(_SELECT_POINT, [_SCREEN, cs_target])
      else:
        self.command_hold = False
        return actions.FunctionCall(_NO_OP, [])
        
    if self.select_barrack: #select barrack
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
        
    if self.build_supplydepot: #build supplydepot
      self.build_supplydepot = False
      distance = 9
      chance = 10
      while True:
        s_target_x = int(self.depot_x + numpy.random.choice([-1,0,1],1)*distance)
        s_target_y = int(self.depot_y + numpy.random.choice([-1,0,1],1)*distance)
        within_map = (2 < s_target_x < max_x-3) and (2 < s_target_y < max_y*1/2)
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
      return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_SCREEN, s_target])
        
    if self.build_barrack: #build barrack
      self.build_barrack = False
      distance = 14
      chance = 10
      while True:
        b_target_x = int(self.barrack_x + numpy.random.choice([-1,0,1],1)*distance)
        b_target_y = int(self.barrack_y + numpy.random.choice([-1,0,1],1)*distance)
        within_map = (2 < b_target_x < max_x-7) and (max_y*1/3 < b_target_y < max_y-7)
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
      return actions.FunctionCall(_BUILD_BARRACKS, [_SCREEN, b_target])
    
    return actions.FunctionCall(_NO_OP, [])

