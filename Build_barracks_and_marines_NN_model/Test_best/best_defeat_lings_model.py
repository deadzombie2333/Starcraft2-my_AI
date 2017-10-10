import numpy 
import os
import re
import struct

_NUM_POPULATION_SIZE = 20
#generate a for loop for multiple generation of genetic algorithm

Threat_matrix_a_0 = numpy.load('threat_a_0.npy').item()
Threat_matrix_a_1 = numpy.load('threat_a_1.npy').item()
Move_matrix_a_0 = numpy.load('move_a_0.npy').item()
Move_matrix_a_1 = numpy.load('move_a_1.npy').item()

Score = numpy.load('score.npy')

good_index = numpy.argmax(Score)

Best_Threat_matrix_a_0 = {}
Best_Threat_matrix_a_1 = {}
Best_Move_matrix_a_0 = {}
Best_Move_matrix_a_1 = {}

for item in range(_NUM_POPULATION_SIZE):
  Best_Threat_matrix_a_0 [item] = Threat_matrix_a_0 [good_index]
  Best_Threat_matrix_a_1 [item] = Threat_matrix_a_1 [good_index]
  Best_Move_matrix_a_0 [item] = Move_matrix_a_0 [good_index]
  Best_Move_matrix_a_1 [item] = Move_matrix_a_1 [good_index]

numpy.save('best_threat_a_0',Best_Threat_matrix_a_0)
numpy.save('best_threat_a_1',Best_Threat_matrix_a_1)
numpy.save('best_move_a_0',Best_Move_matrix_a_0)
numpy.save('best_move_a_1',Best_Move_matrix_a_1)

os.system("python -m pysc2.bin.agent \
  --map DefeatZerglingsAndBanelings \
  --agent pysc2.agents.Defeat_zergling_and_banlings_NN_model.my_agent.Attack_Zerg")

#get score for each candidate
score = numpy.load('score.npy')
Average = numpy.mean(score)
STD = numyp.std(score)
print("Best NN model yet have mean score {} and STD {}".format(Average,STD))
