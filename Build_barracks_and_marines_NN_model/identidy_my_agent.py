import numpy 
import os
import re
import struct

_NUM_POPULATION_SIZE = 8
_NUM_CROSSOVER_RATE = 0.2
_NUM_MUTATION_RATE = 0.05
_NUM_TESTS = 2
#generate a for loop for multiple generation of genetic algorithm

def num_normalize(num,f_length,r_length):
  num = float(num)
  if num == 0:
    return '+' + '0' * f_length + '.' + '0' * r_length
  else:
    if num < 0:
      num = - num
    str_0 = '+'
      
    Num = str(num)
    Num = Num.split('.')
    Num[0] = '0' * (f_length - len(Num[0])) + Num[0]
    if r_length >= len(Num[1]):
      Num[1] = Num[1] + '0' * (r_length - len(Num[1]))
    else:
      Num[1] = Num[1][:r_length]
    new_num = str_0 + Num[0] + '.' + Num[1]
    return new_num

def Cross_over_fun(num_a,num_b):
  front_length = 1
  rear_length = 4
  digit_a = num_normalize(num_a,front_length,rear_length)
  digit_b = num_normalize(num_b,front_length,rear_length)
  location_1 = int(numpy.random.choice(front_length + rear_length,1) + 1)
  new_digit_a = digit_a[:location_1] + digit_b[location_1:]
  new_digit_b = digit_b[:location_1] + digit_a[location_1:]
  new_a = float(new_digit_a)
  new_b = float(new_digit_b)
  return (new_a,new_b)

def Mutation_fun(num_a):
  front_length = 1
  rear_length = 4
  digit_a = num_normalize(num_a,front_length,rear_length)
  index_a = list(range(len(digit_a)))
  index_a.pop(0)
  index_a.pop(front_length)
  location_1 = int(numpy.random.choice(index_a,size = 1))
  substitution = ' '.join(map(str,numpy.random.randint(0,9,size=1)))
  if location_1 == len(digit_a):
    new_digit_a = digit_a[:location_1] +  substitution
  else:
    new_digit_a = digit_a[:location_1] + substitution + digit_a[location_1+1:]
  return new_digit_a


#find a library of candidate for genetic algorithm based on past generation
def next_generation( matrix, good_index, bad_index, crossover_rate, mutation_rate):
  size_0, size_1 = matrix[0].shape
  vector_size = size_0 * size_1

  non_elite = numpy.zeros((1,vector_size))
  bad_candidate = numpy.zeros((1,vector_size))

  for item in range(len(matrix.keys())):
    combined_vector = matrix[item].reshape(1,vector_size)
    if item == good_index:
      elite = combined_vector
    elif item in bad_index:
      bad_candidate = numpy.append(bad_candidate,combined_vector,axis = 0)
    else:
      non_elite = numpy.append(non_elite,combined_vector,axis = 0)
  
  combined_matrix = non_elite[1:]
  bad_candidate = bad_candidate[1:]
  combined_index = numpy.arange(combined_matrix.shape[0])
  numpy.random.shuffle(combined_index)
  if len(bad_index) == 0:
    combined_index = combined_index[:-1]
  combined_matrix = numpy.append(combined_matrix[combined_index],elite,axis = 0)
  
  for cross_over_i in range(combined_matrix.shape[0]-1):
    for cross_over_j in range(combined_matrix.shape[1]):
      if numpy.random.uniform(0,1) < crossover_rate:
        combined_matrix[cross_over_i][cross_over_j], combined_matrix[cross_over_i+1][cross_over_j] = \
        Cross_over_fun(combined_matrix[cross_over_i][cross_over_j],combined_matrix[cross_over_i+1][cross_over_j])
        
  for mutation_i in range(combined_matrix.shape[0]):
    for mutation_j in range(combined_matrix.shape[1]):
      if numpy.random.uniform(0,1) < mutation_rate:
        combined_matrix[mutation_i][mutation_j] = Mutation_fun(combined_matrix[mutation_i][mutation_j])  
  
  combined_matrix = numpy.append(combined_matrix,elite,axis = 0)
  
  if len(bad_index) != 0:

    bad_candidate_min = numpy.amin(bad_candidate,axis = 0)
    bad_candidate_max = numpy.amax(bad_candidate,axis = 0)
    
    good_candidate = numpy.append(non_elite,elite,axis = 0)[1:]
    
    good_candidate_min = numpy.amin(good_candidate,axis = 0)
    good_candidate_max = numpy.amax(good_candidate,axis = 0)
    
    standard1 = good_candidate_min > bad_candidate_max
    standard2 = good_candidate_max < bad_candidate_min
    
    random_min = numpy.zeros(bad_candidate_min.shape)
    random_max = 3*numpy.ones(bad_candidate_min.shape)
    for item in range(len(standard1)):
      if standard1[item]:
        random_min[item] = bad_candidate_max[item]
    
    for item in range(len(standard2)):
      if standard2[item]:
        random_max[item] = bad_candidate_min[item]
    
    candidate_num = combined_matrix.shape[0]
    while candidate_num < _NUM_POPULATION_SIZE:
      new_candidate = numpy.zeros((1,bad_candidate_min.shape[0]))
      for iii in range(bad_candidate_min.shape[0]):
        new_candidate[0][iii] = numpy.random.uniform(low = random_min[iii], high = random_max[iii],size = (1,1))
      combined_matrix = numpy.append(combined_matrix,new_candidate,axis = 0)
      candidate_num = combined_matrix.shape[0]
  new_matrix = {}
  
  Full_matrix = numpy.load('Full_Matrix.npy')
  Full_matrix = numpy.append(Full_matrix,combined_matrix,axis = 0)
  numpy.save('Full_Matrix',Full_matrix)
  print(numpy.flipud(combined_matrix))
  
  for lines in range(combined_matrix.shape[0]):
    new_matrix[lines] = combined_matrix[lines].reshape(size_0,size_1)
  return new_matrix

for i in range(5):
  try: 
    Thres_vector = numpy.load('Thres_vector.npy').item()
    
    Score = numpy.load('score.npy')
    score = numpy.mean(Score,axis=1)
    summed_score = numpy.load('summed_score.npy')
    
    if numpy.amax(score) > numpy.amax(summed_score):
      good_index = numpy.argmax(score)
    else:
      random_location = numpy.random.uniform(0,1,1) * sum(score)
      for j in range(score.shape[0]):
        if sum(score[:j+1]) > random_location:
          good_index = j
          break
    
    bad_index = numpy.where(score == 0)[0]
    
    Thres_vector = next_generation(Thres_vector, good_index, bad_index, _NUM_CROSSOVER_RATE, _NUM_MUTATION_RATE)
 
  except FileNotFoundError:
    Thres_vector = {}
    for item in range(_NUM_POPULATION_SIZE):
      Thres_vector [item] = numpy.random.uniform(low = 0, high = 3,size = (1,3))
    Score = numpy.zeros((_NUM_POPULATION_SIZE,_NUM_TESTS))
    Summed_Score = numpy.zeros((1,_NUM_POPULATION_SIZE))
    Full_matrix = numpy.zeros((1,3))
    numpy.save('Full_Matrix',Full_matrix)
    numpy.save('summed_score',Summed_Score)
    numpy.save('score',Score)# target: average: 133 138 Max: 133 142
    
  numpy.save('Thres_vector',Thres_vector)

  os.system("python -m pysc2.bin.agent \
	--map BuildMarines \
	--agent pysc2.agents.Build_barracks_and_marines_NN_model.my_agent.Build_Marine")

  #get score for each candidate
  Score = numpy.load('score.npy')
  score = numpy.mean(Score,axis=1)
  Summed_Score = numpy.load('summed_score.npy')
  Summed_Score = numpy.vstack((Summed_Score,score))
  numpy.save('summed_score',Summed_Score)
  print("average score {}".format(score))
  print("trials {}, highest score {}".format(Summed_Score.shape[0] - 1,numpy.amax(score))) 
