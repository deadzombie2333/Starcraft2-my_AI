import numpy 
import os
import re
import struct

os.system("python -m pysc2.bin.agent \
	--map BuildMarines \
	--agent pysc2.agents.Build_barracks_and_marines_NN_model.my_agent.Build_Marine")
