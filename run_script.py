import os
import numpy as np

subjects = np.loadtxt('HCP_list', dtype=int)

for sub in subjects:

	command = 'qsub -V -l mem_free=5G -M kaihwang -j y -o /home/despoB/kaihwang/tmp/ -e /home/despoB/kaihwang/tmp/ -N g%s /home/despoB/kaihwang/bin/ThaBha/ThaBha.py %s' %(sub, sub)
	os.system(command)