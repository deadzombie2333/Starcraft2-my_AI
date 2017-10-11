import numpy 
import os
import re
import struct

_NUM_POPULATION_SIZE = 10
_NUM_CROSSOVER_RATE = 0.2
_NUM_MUTATION_RATE = 0.05
_NUM_TESTS = 10
trials = 0
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
  front_length = 2
  rear_length = 4
  digit_a = num_normalize(num_a,front_length,rear_length)
  digit_b = num_normalize(num_b,front_length,rear_length)
  location_1 = int(numpy.random.randint(0,len(digit_a),size=1))
  new_digit_a = digit_a[:location_1] + digit_b[location_1:]
  new_digit_b = digit_b[:location_1] + digit_a[location_1:]
  new_a = float(new_digit_a)
  new_b = float(new_digit_b)
  return (new_a,new_b)

def Mutation_fun(num_a):
  front_length = 2
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
  if numpy.random.uniform(0,1) < 0.2:
    new_a = float(new_digit_a)
  else:
    new_a = -float(new_digit_a)
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

  for lines in range(combined_matrix.shape[0]):
    new_matrix[lines] = combined_matrix[lines].reshape(size_0,size_1)
  return new_matrix

while True:
  try: 
    Threat_matrix_a_0 = numpy.load('threat_a_0.npy').item()
    Threat_matrix_a_1 = numpy.load('threat_a_1.npy').item()
    Move_matrix_a_0 = numpy.load('move_a_0.npy').item()
    Move_matrix_a_1 = numpy.load('move_a_1.npy').item()
    
    Score = numpy.load('score.npy')
    score = numpy.mean(Score,axis=1)
    score_mean = numpy.mean(score)
    score_std = numpy.std(score)
    if score_mean > 150 and score_std < 20:
      break
    random_location = numpy.random.uniform(0,1,1) * sum(score)
    for j in range(score.shape[0]):
      if sum(score[:j+1]) > random_location:
        good_index = j
        break
    
    Threat_matrix_a_0 = next_generation(Threat_matrix_a_0, good_index, _NUM_CROSSOVER_RATE, _NUM_MUTATION_RATE)
    Threat_matrix_a_1 = next_generation(Threat_matrix_a_1, good_index, _NUM_CROSSOVER_RATE, _NUM_MUTATION_RATE)
    Move_matrix_a_0 = next_generation(Move_matrix_a_0, good_index, _NUM_CROSSOVER_RATE, _NUM_MUTATION_RATE)
    Move_matrix_a_1 = next_generation(Move_matrix_a_1, good_index, _NUM_CROSSOVER_RATE, _NUM_MUTATION_RATE)
 
  except FileNotFoundError:
    Threat_matrix_a_0 = {}
    Threat_matrix_a_1 = {}
    Move_matrix_a_0 = {}
    Move_matrix_a_1 = {}
    overall_score = numpy.zeros((1,_NUM_POPULATION_SIZE))
    for item in range(_NUM_POPULATION_SIZE):
      Threat_matrix_a_0 [item] = numpy.random.uniform(low = -1, high = 1,size = (4,4))
      Threat_matrix_a_1 [item] = numpy.random.uniform(low = -1, high = 1,size = (1,4))
      Move_matrix_a_0 [item] = numpy.random.uniform(low = -1, high = 1,size = (4,2))
      Move_matrix_a_1 [item] = numpy.random.uniform(low = -1, high = 1,size = (1,2))
    Score = numpy.zeros((_NUM_POPULATION_SIZE,_NUM_TESTS))
    numpy.save('score',Score)
    numpy.save('score_summation',overall_score)

  numpy.save('threat_a_0',Threat_matrix_a_0)
  numpy.save('threat_a_1',Threat_matrix_a_1)
  numpy.save('move_a_0',Move_matrix_a_0)
  numpy.save('move_a_1',Move_matrix_a_1)
  
  os.system("python -m pysc2.bin.agent \
    --map DefeatZerglingsAndBanelings \
    --agent pysc2.agents.Defeat_zergling_and_banlings_NN_model.my_agent.Attack_Zerg")

  #get score for each candidate
  Score = numpy.load('score.npy')
  score = numpy.mean(Score,axis=1)
  overall_score = numpy.load('score_summation.npy')
  overall_score = numpy.append(overall_score,[numpy.transpose(score)],axis = 0)
  print("average score {}".format(score))
  trials += 1
  numpy.save('score_summation',overall_score)
  print("trials {}, highest score {}".format(trials,numpy.amax(score)))
