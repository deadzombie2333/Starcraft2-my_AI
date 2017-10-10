import numpy 
import os
import re
import struct

_NUM_POPULATION_SIZE = 5
#generate a for loop for multiple generation of genetic algorithm

Thres_vector = numpy.load('Thres_vector.npy').item()
matrix_a_0 = numpy.load('Matrix_a_0.npy').item()
matrix_a_1 = numpy.load('Matrix_a_1.npy').item()

Score = numpy.load('score.npy')

good_index = numpy.argmax(Score[:,0])

Best_Thres_vector = {}
Best_matrix_a_0 = {}
Best_matrix_a_1 = {}

for item in range(_NUM_POPULATION_SIZE):
  Best_Thres_vector [item] = Thres_vector [good_index]
  Best_matrix_a_0 [item] = matrix_a_0 [good_index]
  Best_matrix_a_1 [item] = matrix_a_1 [good_index]
 
 
numpy.save('best_Thres_vector',Best_Thres_vector)
numpy.save('best_Matrix_a_0',Best_matrix_a_0)
numpy.save('best_Matrix_a_1',Best_matrix_a_1)

os.system("python -m pysc2.bin.agent \
	--map BuildMarines \
	--agent pysc2.agents.Build_barracks_and_marines_NN_model.my_agent.Build_Marine")

#get score for each candidate
score = numpy.load('score.npy')
Average = numpy.mean(score[:,0])
STD = numpy.std(score[:,0])
print("Best NN model yet have mean score {} and STD {}".format(Average,STD))
