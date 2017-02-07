#!/usr/local/anaconda/bin/python
# analyze HCP data to examine PC-Q and PC-Behavior correlations

import numpy as np
from igraph import Graph, ADJ_UNDIRECTED, VertexClustering
import bct
import os 
import sys
import pandas as pd
import glob


datapath = '/home/despoB/connectome-thalamus/NotBackedUp/HCP_Matrices/'
Conditions = ['EMOTION', 'GAMBLING', 'MOTOR', 'SOCIAL', 'LANGUAGE', 'WM', 'RELATIONAL', 'REST1', 'REST2']
Directions = ['_LR', '_RL']
Atlases =['_Gordon_plus_Morel', '_Gordon_plus_Thalamus_WTA']


def matrix_to_igraph(matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=False,mst=False):
	'''use igrpah function to build graphs'''
	matrix = threshold(matrix,cost,binary,check_tri,interpolation,normalize,mst)
	g = Graph.Weighted_Adjacency(matrix.tolist(),mode=ADJ_UNDIRECTED,attr="weight")
	print 'Matrix converted to graph with density of: ' + str(g.density())
	if np.diff([cost,g.density()])[0] > .005:
		print 'Density not %s! Did you want: ' %(cost)+ str(g.density()) + ' ?' 
	return g


def threshold(matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=False,mst=False):
	'''threshold function from Maxwell'''
	matrix[np.isnan(matrix)] = 0.0
	matrix[matrix<0.0] = 0.0
	np.fill_diagonal(matrix,0.0)
	c_cost_int = 100-(cost*100)
	if check_tri == True:
		if np.sum(np.triu(matrix)) == 0.0 or np.sum(np.tril(matrix)) == 0.0:
			c_cost_int = 100.-((cost/2.)*100.)
	if c_cost_int > 0:
		if mst == False:
			matrix[matrix<np.percentile(matrix,c_cost_int,interpolation=interpolation)] = 0.
		else:
			matrix = np.tril(matrix,-1)
			mst = minimum_spanning_tree(matrix*-1)*-1
			mst = mst.toarray()
			mst = mst.transpose() + mst
			matrix = matrix.transpose() + matrix
			a = matrix<np.percentile(matrix,c_cost_int,interpolation=interpolation)
			b = mst==0.0
			matrix[(matrix<np.percentile(matrix,c_cost_int,interpolation=interpolation)) & (mst==0.0)] = 0.
	if binary == True:
		matrix[matrix>0] = 1
	if normalize == True:
		matrix = matrix/np.sum(matrix)
	return matrix


def run_graph(sub):
	'''run graph analysis, get PC/WMD/Q'''
	for condition in Conditions:
		for atlas in Atlases:			
			
			matrix = []
			for direction in Directions:
				fn = datapath + str(sub) +'_' + condition + direction + atlas + '.corrmat'			
				try:	
					matrix += [np.loadtxt(fn)]
				except:
					break
			
			PCs = []
			WMDs = []
			Qs = []		
			if np.shape(matrix)[0] > 1:			
				matrix = np.sum(matrix, axis=0)/np.shape(matrix)[0]
				
				for c in np.arange(0.05, 0.16, 0.01):			
					mat_th = threshold(matrix.copy(), c)	
					graph = matrix_to_igraph(matrix.copy(), c)
					i = graph.community_infomap(edge_weights='weight')
					CI = np.asarray(i.membership)+1
					PCs += [bct.participation_coef(mat_th, CI)]
					WMDs += [bct.module_degree_zscore(mat_th, CI)]
					Qs += [i.modularity]

				fn = datapath + str(sub) +'_' + condition  + atlas + '.PC'
				np.save(fn, PCs)
				fn = datapath + str(sub) +'_' + condition  + atlas + '.WMD'
				np.save(fn, WMDs)
				fn = datapath + str(sub) +'_' + condition  + atlas + '.Q'
				np.save(fn, Qs)



def compile_dataframe(condition):
	''' load PC/Q/WMD across subjects, for now ave across costs'''
	Subjects = np.loadtxt('HCP_list', dtype=int)
	df = pd.DataFrame(columns = ['Subject', 'Condition', 'Atlas', 'ROI', 'Q', 'PC', 'WMD', 'Performance'])
	bdf = pd.read_csv('/home/despoB/mb3152/dynamic_mod/os_behavior_data.csv')
	for sub in Subjects:
		
		for atlas in Atlases:
			try:
				## data
				tmp_df = pd.DataFrame()
				fn = datapath + str(sub) +'_' + condition  + atlas + '.PC.npy'
				PC = np.mean(np.load(fn), axis=0)
				fn = datapath + str(sub) +'_' + condition  + atlas + '.WMD.npy'
				WMD = np.mean(np.load(fn), axis =0)
				fn = datapath + str(sub) +'_' + condition  + atlas + '.Q.npy'
				Q = np.mean(np.load(fn), axis =0)

				tmp_df['PC'] = PC
				tmp_df['Q'] = [Q]*len(PC)
				tmp_df['WMD'] = WMD
				tmp_df['Subject'] = np.int(sub)
				tmp_df['Atlas'] = atlas[1:]
				tmp_df['ROI'] = np.arange(1,len(PC)+1, dtype=int)
				tmp_df['Condition'] = condition
				
				## behav
				files = glob.glob('/home/despoB/mb3152/scanner_performance_data/%s_tfMRI_*%s*_Stats.csv' %(str(sub),condition))
				performance = []
				for f in files:
					behav_df = pd.read_csv(f)
					if condition == 'WM':
						t_performance = np.mean(behav_df['Value'][[24,27,30,33]])  #average accuracy
					if condition == 'RELATIONAL':
						t_performance = np.mean([behav_df['Value'][0],behav_df['Value'][1]])
					if condition == 'LANGUAGE':
						t_performance = np.mean([behav_df['Value'][2],behav_df['Value'][5]])  #need to further normalize by age for language task
						s1 = bdf['ReadEng_AgeAdj'][bdf.Subject == int(sub)] 
						s2 = bdf['PicVocab_AgeAdj'][bdf.Subject == int(sub)]
						t_performance = np.nanmean([t_performance,s1,s2])
					if condition == 'SOCIAL':
						t_performance = np.mean([behav_df['Value'][0],behav_df['Value'][5]]) #accuracy
					performance.append(t_performance)
				
				tmp_df['Performance']= np.mean(performance)
				df = df.append(tmp_df,  ignore_index=True)

			except:
				continue	

	return df






if __name__ == "__main__":


	######### run graph metics
	#for sub in subjects:
	#sub = 100307
	#sub =  sys.argv[1] #sys.stdin.read().strip('\n')
	#run_graph(sub)
	
	######### compaile dataframe
	for condition in Conditions:
		df = compile_dataframe(condition)
		fn = 'Data/graph_%s_df.csv' %condition
		df.to_csv(fn)	
	#df = compile_dataframe('WM')			


	##### look coorrelation
	PCcorr =[]
	for i in np.arange(1,np.max(df['ROI'])+1):
		PC = df[(df['Atlas']=='Gordon_plus_Thalamus_WTA') & (df['ROI']==i)]['PC'].values
		behav = df[(df['Atlas']=='Gordon_plus_Thalamus_WTA') & (df['ROI']==i)]['Performance'].values	
		PCcorr += [np.corrcoef(PC[~np.isnan(behav)],behav[~np.isnan(behav)])[0,1]]





	