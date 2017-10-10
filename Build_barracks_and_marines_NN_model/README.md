# Mini game, build marines:

In mini game build marine, agent is attempt to build as many marine as possible with only 8 minefields and one command center. This simulate the “Macro” in Starcraft 2 where player have to balance between many factor such as timing of build supply depot, total worker built and number of barracks build. Based on my understanding, success of this mini game rely on certain factors: 1, build exact amount of worker such that mineral mining is maximized; 2, in exacting time, build supply depot such that there are enough supply to build marine; 3, build exact amount of barracks such that minerals harvested can support marine production with small amount of mineral reserve. In brief, is to design a build order based on current mineral and supply such that marine build is in high speed and continuous.

# AI architecture:

In this AI design, I attempt to design a threshold system which quantify the desired ratio or parameter of ideal system. Based on current status such as minerals, worker supply, maximum supply etc., threshold system compare the ratio with the intended threshold of ideal system. It then conclude which ratio is the farthest away from ideal and use the corresponding build to compensate for this shortness. If all ratios are larger than ideal threshold, the system is considered in optimal and no operation will be necessary.

This threshold is still simple attempt for now. Further research on it performance will be analyzed. 

