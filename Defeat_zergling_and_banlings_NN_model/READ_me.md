# Mini game, defeat zerglings and banglings:

In mini game Defeat zerglings and banelings, friendly units are terran marines while hostile unit are zerg zerglings and banelings. Based on knowledge, banelings are melee (short range) unit which have higher damage to group of light armor unit such as marines (one can also observe this by play this mini game for several rounds). Therefore, it is important to select the optimal attacking target of friendly unit and take advantage of attack range of marines. A full of AI architecture of this mini game tactic would be first acknowledge the advantage of attack range of friendly unit with disadvantage of attack damage (DPS). Then, based on this, a “hit and run” tactic is designed to limit hostile unit damage (only part of hostile unit can attack friendly unit at the same time). In my project, I bypassed the first decision part and went straight to marine “hit and run” micro. 

# Ai architecture:

In my AI design, two design questions are considered, one is what unit to attack when attacking, and another is where to move when moving. Two neural networks are designed to answer this question. 
When attacking, I attempt to design a NN that calculate the highest threat of all enemy unit toward friendly unit and the highest threat unit should be eliminated first. In my design, I take each friendly-hostile unit as one pair, input both unit damage per second versus the other, distance between each other and the health of friendly unit as input for my NN. NN then calculate the threat of the selected hostile unit toward the selected friendly unit and add it to threat index of the selected hostile unit. After looping over all hostile unit and friendly unit, hostile units will have their summed threat index which is then compared. Hostile unit with highest threat will be selected as priority attacking unit for friendly unit. 

When moving, NN is used to calculate the location or direction to move. There are generally two conditions in “hit and run” real game play: one is hostile units are advancing toward friendly unit so that the unit should move away from hostile unit and the other is hostile unit are retreating where friendly unit should chase. Although, in this mini game, hostile units are in auto attack (attack no matter what) so only attack and move away property is trained in this project. Location of move is calculated based on multiple factor: general location of friendly unit, general location of hostile unit, friendly unit health and hostile unit health. A (x,y) location is calculated use NN to direct the friendly the location of friendly unit move.

# Stages of combats and NN scores:

“Hit and Run” is designed (instead of trained) in this agent. Before the first engagement of hostile unit, friendly unit will only have attack command to ensure the initiation of combat. This stage is named Stage_1 in my code. After the first contact of friendly unit and hostile unit (damage done on hostile unit), the agent turns to Stage_2 where friendly unit will take “Attack-Move-Attack-Move” as the “Hit and Run” indicated. After some round of combat, either friendly unit or hostile unit will be very low (only one unit left). As soon as there are only one friendly unit or one hostile unit left (Stage_3), the agent will record the current score by calculating the number of hostile unit and friendly unit on the field. If friendly is expected to win (friendly unit have more hit point), the score will add up. If friendly is expected to loss, the score will be stored as the score of introduced NN. After the score is recorded, the agent turns to Stage_4 where the agent wait for the combat to finish and ready for map reset. 

# Result and discussion:

Based on the current NN structure, only some observation result is available. I will include some result as the project proceed.

First of all, the agent can run successfully on my PC. Early generation have pretty lower score as expected. After several generations, highest score of current generation increases. Randomness in hostile unit location will influence the outcome of NN as well as the line-up of lings (banelings at front or back). Recorded score generally lower than score displayed (individual episode score intended). Highest recorded score for single episode is around 200 which is still lower than the NN proposed in original publication. 

# Current Issue:

Game speed is still too fast for current NN training. I attempted to change the game speed of mini game but not working very well. During combating, CPU is at full capacity which influence the agent control. Currently, the agent works better with smaller amount of marine since it requires lower computation. I’ll attempt to modify my code for an efficient computation.

Score recorded is still inaccurate. Because, in some situation, map reset will automatically activate as soon as the last marine or zergling dies. Therefore, calculating based on current observed friendly and hostile unit is wrong (most of the time lower than paper proposed score). I’ll attempt to use the accumulative score property later to see if this improves.

Currently, the model is still under-trained. For same candidate, its episode score variation is still too high. I believe this should be resolve after the fix of game speed and score recording. 
