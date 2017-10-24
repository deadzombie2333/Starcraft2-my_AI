import numpy 
import os
import re
import struct

_NUM_POPULATION_SIZE = 1
_NUM_TESTS = 5
#generate a for loop for multiple generation of genetic algorithm

Threat_matrix_a_0 = numpy.load('threat_a_0.npy').item()
Threat_matrix_a_1 = numpy.load('threat_a_1.npy').item()
Move_matrix_a_0 = numpy.load('move_a_0.npy').item()
Move_matrix_a_1 = numpy.load('move_a_1.npy').item()

Score = numpy.load('score.npy')

good_index = numpy.argmax(numpy.mean(Score,axis=1))
Best_Threat_matrix_a_0 = {}
Best_Threat_matrix_a_1 = {}
Best_Move_matrix_a_0 = {}
Best_Move_matrix_a_1 = {}

for item in range(_NUM_POPULATION_SIZE):
  Best_Threat_matrix_a_0 [item] = Threat_matrix_a_0 [good_index]
  Best_Threat_matrix_a_1 [item] = Threat_matrix_a_1 [good_index]
  Best_Move_matrix_a_0 [item] = Move_matrix_a_0 [good_index]
  Best_Move_matrix_a_1 [item] = Move_matrix_a_1 [good_index]

Best_score = numpy.zeros((_NUM_POPULATION_SIZE,_NUM_TESTS))
numpy.save('best_NN_score',Best_score)

numpy.save('best_threat_a_0',Best_Threat_matrix_a_0)
numpy.save('best_threat_a_1',Best_Threat_matrix_a_1)
numpy.save('best_move_a_0',Best_Move_matrix_a_0)
numpy.save('best_move_a_1',Best_Move_matrix_a_1)

os.system("python -m pysc2.bin.agent --map DefeatZerglingsAndBanelings --agent pysc2.agents.Test_lings_model.best_of_my_agent.Attack_Zerg")

#get score for each candidate
Best_score = numpy.load('best_NN_score.npy')
Average = numpy.mean(Best_score)
STD = numpy.std(Best_score)
print("Best NN model yet have mean score {} and STD {}".format(Average,STD))
