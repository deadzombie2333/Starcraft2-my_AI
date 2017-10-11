# Starcraft2-my_AI

# simple AI for starcraft 2 minigame based on pysc2:

In this small project, two mini games are considered for now. One is defeat zerglings and banelings and the other is build marine. This two mini games, on my opinion, is good representative of tactic part and strategy part of Starcraft 2. 

Tactic part of Starcraft 2 works on winning a local battle versus enemy use limited amount of army. Key to this local victory is based on optimal use of friendly unit advantage such as longer attack range, higher moving speed, higher damage versus specific unit etc. It is most of time refer to as “Micro” in Starcraft 2. Strategy part of Starcraft 2 works on winning in a “bigger picture”. By optimal design of building order to possess advantage in army size or dominance in army technique (such as use banshee, a high damage anti-ground unit versus roach, a ground unit with no anti-air). This part can be refer to as “Macro” in Starcraft 2. Two selected mini game represent the “Micro” and “Macro” aspect of Starcraft 2 which is very challenging.

In my project, I attempt to design agent for both mini games. Goal of this project is, for both mini game, achieve the highest score possible and ultimately, design a neural network based agent that performs as good as human player. The AI structure of the agent will be hybrid which his based on both machine learning and rules basis. 

This project is built on python 3.6.1 and numpy. During my coding, I referred to the project by https://github.com/chris-chris/pysc2-examples and https://github.com/skjb/pysc2-tutorial. 

# Training of NN:

NN in this project is trained use Evolution Algorithm (EA). The basic idea of EA come from evolution of natural species where the strongest one have higher opportunity to reproduce. 

To initiate EA for NN training, we first find generation 0 which is the initial generation. A library of 20 candidates (number of populations) is generated at first while all parameters in each candidate is generated randomly between -1 and 1. The saved library (.npy file) is then imported into the agent where for each candidate, its score is recorded as described in previous section. After all candidates are tested, all recorded score will be stored in .npy file.

Before the next generation (generation 1) started, EA first read all scores from score file. Based on the score of all candidate, roulette method (higher score candidate will have bigger chance to be selected) is adopted to select an elite candidate that will survive this generation where the rest of the generation will go through a cross-over and mutation process to generate the rest of next generation. 

In order to demonstrate the cross-over and mutation of parent candidate (candidate from generation 0) to generate children candidate (candidate to be, in generation 1), I will use two arbitrary candidates as example. Let the parent candidate be: “1.097” and “0.472”. Two candidates first transferred into a normalized form: “001.0970” and “000.4720”. Over some cross-over rate, some section of parent candidate will switch to generate two children candidates, such as: “001.0420” and “000.4970”. Over some mutation rate, some number of single parent will also mutate to generate one child candidate, such as: “001.3970”. Cross-over and mutation are major searching method of EA algorithm.

After cross-over and mutation, next generation (generation 1) can be generated use previous generation (generation 0) based on the score. After this, next generation will be imported into the agent to generate its score.

# Current Progress:

Currently, agent design for mini game attack zerglings and baneling are successful. The agent score is getting close to human score and far superior to the agent published in the paper. Detail information regarding agent and AI structure is within its separate folder. Design of agent for mini game build marines is just initiated. The agent itself can only produce several marines and it is still far away from human scores. 
