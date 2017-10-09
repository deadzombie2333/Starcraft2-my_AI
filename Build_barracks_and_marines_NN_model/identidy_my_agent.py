import numpy 
import os
import re
import struct

_NUM_POPULATION_SIZE = 5
_NUM_CROSSOVER_RATE = 0.2
_NUM_MUTATION_RATE = 0.05
_NUM_MAX_GENERATION = 1
#generate a for loop for multiple generation of genetic algorithm

def num_normalize(num,f_length,r_length):
  num = float(num)
  if num == 0:
    return '+' + '0' * f_length + '.' + '0' * r_length
  else:
    if num > 0:
      str_0 = '+'
    else: 
      str_0 = '-'
      num = - num
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
  front_length = 3
  rear_length = 4
  if num_a > 1000 or num_b > 1000:
    print(num_a)
    print(num_b)
    quit()
  digit_a = num_normalize(num_a,front_length,rear_length)
  digit_b = num_normalize(num_b,front_length,rear_length)
  location_1 = int(numpy.random.randint(0,len(digit_a),size=1))
  new_digit_a = digit_a[:location_1] + digit_b[location_1:]
  new_digit_b = digit_b[:location_1] + digit_a[location_1:]
  new_a = float(new_digit_a)
  new_b = float(new_digit_b)
  if new_a > 1000 or new_b > 1000:
    print(new_a)
    print(new_b)
    quit()
    new_a = numpy.random.uniform(low = -1, high = 1)
    new_b = numpy.random.uniform(low = -1, high = 1)
  return (new_a,new_b)

def Mutation_fun(num_a):
  front_length = 3
  rear_length = 4
  if num_a > 1000:
    print(num_a)
    quit()
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
  if numpy.random.uniform(0,1) < 0.5:
    new_a = float(new_digit_a)
  else:
    new_a = -float(new_digit_a)
  if new_a > 1000:
    print(new_a)
    quit()
    new_a = numpy.random.uniform(low = -1, high = 1)
  return new_a


#find a library of candidate for genetic algorithm based on past generation
def next_generation( matrix, good_index, crossover_rate, mutation_rate):
  size_0, size_1 = matrix[0].shape
  vector_size = size_0 * size_1

  non_elite = numpy.zeros((1,vector_size))

  for item in range(len(matrix.keys())):
    combined_vector = matrix[item].reshape(1,vector_size)
    if item == good_index:
      elite = combined_vector
    else:
      non_elite = numpy.vstack((non_elite,combined_vector))
  non_elite = non_elite[1:]
  combined_matrix = non_elite
  combined_index = numpy.arange(combined_matrix.shape[0])
  numpy.random.shuffle(combined_index)
  combined_matrix = combined_matrix[combined_index]
  
  for cross_over_i in range(combined_matrix.shape[0]-2):
    for cross_over_j in range(combined_matrix.shape[1]-1):
      if numpy.random.uniform(0,1) < crossover_rate:
        combined_matrix[cross_over_i][cross_over_j], combined_matrix[cross_over_i+1][cross_over_j] = \
        Cross_over_fun(combined_matrix[cross_over_i][cross_over_j],combined_matrix[cross_over_i+1][cross_over_j])
        
  for mutation_i in range(combined_matrix.shape[0]-1):
    for mutation_j in range(combined_matrix.shape[1]-1):
      if numpy.random.uniform(0,1) < mutation_rate:
        combined_matrix[mutation_i][mutation_j] = Mutation_fun(combined_matrix[mutation_i][mutation_j])  
  combined_matrix = numpy.vstack((elite,combined_matrix))
  new_matrix = {}
  
  if combined_matrix.shape[0] != _NUM_POPULATION_SIZE:
    quit()

  for lines in range(combined_matrix.shape[0]):
    new_matrix[lines] = combined_matrix[lines].reshape(size_0,size_1)
  return new_matrix

for trials in range(_NUM_MAX_GENERATION):
  try: 
    Thres_vector = numpy.load('Thres_vector.npy').item()
    matrix_a_0 = numpy.load('Matrix_a_0.npy').item()
    matrix_a_1 = numpy.load('Matrix_a_1.npy').item()
    
    Score = numpy.load('score.npy')
    
    if sum(Score[0,:]) == len(Score[0,:]) * Score[0][0]:
      if sum(Score[1,:]) == len(Score[1,:]) * Score[1][0]:
        if sum(Score[2,:]) == len(Score[2,:]) * Score[2][0]:
          good_index = numpy.random.choice(len(Score),1)
        else:
          score = Score[2,:]
      else:
        score = Score[1,:]
    else:
      score = Score[0,:]
    
    try:
      random_location = numpy.random.uniform(0,1,1) * sum(score)
      for j in range(score.shape[0]):
        if sum(score[:j+1]) > random_location:
          good_index = j
          break
    except NameError:
      pass
    
    Thres_vector = next_generation(Thres_vector, good_index, _NUM_CROSSOVER_RATE, _NUM_MUTATION_RATE)
    matrix_a_0 = next_generation(matrix_a_0, good_index, _NUM_CROSSOVER_RATE, _NUM_MUTATION_RATE)
    matrix_a_1 = next_generation(matrix_a_1, good_index, _NUM_CROSSOVER_RATE, _NUM_MUTATION_RATE)
 
  except FileNotFoundError:
    Thres_vector = {}
    matrix_a_0 = {}
    matrix_a_1 = {}
    for item in range(_NUM_POPULATION_SIZE):
      Thres_vector [item] = numpy.random.uniform(low = -1, high = 1,size = (1,4))
      matrix_a_0 [item] = numpy.random.uniform(low = -1, high = 1,size = (5,4))
      matrix_a_1 [item] = numpy.random.uniform(low = -1, high = 1,size = (1,4))
    Score = numpy.zeros((_NUM_POPULATION_SIZE,3))
    
  numpy.save('Thres_vector',Thres_vector)
  numpy.save('Matrix_a_0',matrix_a_0)
  numpy.save('Matrix_a_1',matrix_a_1)
  numpy.save('score',Score)

  os.system("python -m pysc2.bin.agent \
	--map BuildMarines \
	--agent pysc2.agents.Build_barracks_and_marines_NN_model.my_agent.Build_Marine")

  #get score for each candidate
  score = numpy.load('score.npy')
  print(score)
