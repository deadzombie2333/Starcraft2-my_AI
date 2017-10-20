import numpy 
import os
import re
import struct

_NUM_POPULATION_SIZE = 3
#generate a for loop for multiple generation of genetic algorithm

Thres_vector = numpy.load('Thres_vector.npy').item()

Score = numpy.load('score.npy')
score = numpy.mean(Score,axis=1)

good_index = numpy.argmax(score)

quit()
Best_Thres_vector = {}

for item in range(_NUM_POPULATION_SIZE):
  Best_Thres_vector [item] = Thres_vector [good_index]

Score = numpy.zeros((_NUM_POPULATION_SIZE,1))
numpy.save('score',Score)# target: average: 133 138 Max: 133 142
 
numpy.save('best_Thres_vector',Best_Thres_vector)

os.system("python -m pysc2.bin.agent \
	--map BuildMarines \
	--agent pysc2.agents.Build_barracks_and_marines_NN_model.Test_best.best_of_my_agent.Build_Marine")

#get score for each candidate
score = numpy.load('score.npy')
Average = numpy.mean(score[:,0])
STD = numpy.std(score[:,0])
print("Best NN model yet have mean score {} and STD {}".format(Average,STD))
