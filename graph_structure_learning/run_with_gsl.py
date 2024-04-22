"""
    Run our GNN-GSL model.
        - Much of this code can be taken from main.py, but
        we will have to modify the running process so that it
        takes the 'readout' output from the trainer and feeds
        it into the GSL pipeline.
"""

import json

from xyz_processor import *
from molecular_similarity_graph import *
import numpy as np
from scripts.main import Runner
# from graph_structure_learner import GraphLearner, Model, ModelHandler


# data = []
# processor = None
# n = 700

"""
    Testing molecule data processing
"""

# with open('data/QM9.json') as f:
#     data = json.load(f)
#     data = data[:n]
#     processor = MoleculeProcessor(molList=data)
#     processor.processMolObjects()
#     processor.computeMetrics()

# print("PROCESSED MOLS: ", processor.processedMols)
# print(processor.similarityMap)
# print(type(processor.similarityMap))
# f.close()

"""
    Testing molecular similarity graph generation 
"""

# Generate random embeddings for testing purposes.
# The jth vector in this array corresponds to the
# embedding of molecule j.
# embeddings = np.array([np.random.standard_normal(5) for i in range(n)])
# threshold = 0.14
# simGraph = MolecularSimilarityGraph(
#     similarityMap=processor.similarityMap,
#     tanimotoThreshold=threshold,
#     embeddings=embeddings)

# simGraph.toGraphData()
# print(simGraph.initialAdj)
# print(len(simGraph.initialAdj[0]))
# print(len(simGraph.initialAdj.T[0]))
# simGraph.visualize(n)


class Model:
    """
      1. Process the data into mol objects (xyz_processor.py)
      2. Construct the MSG (A^0)
      3. Feed A^0 into STGNN
    """

    def __init__(self, num_mols, raw_data, tanimoto_threshold=0.12):
        self.tanimoto_threshold = tanimoto_threshold
        self.num_moles = num_mols
        self.raw_data = raw_data
        self.data = []
        self.processor = None
        with open(self.raw_data) as f:
            self.data = json.load(f)
            self.data = self.data[:self.num_moles]
            processor = MoleculeProcessor(molList=self.data)
        f.close()
        self.processor.processMolObjects()
        self.processor.computeMetrics()
        self.sim_map = self.processor.similarityMap
        self.sim_graph_data = None
        self.A_0 = None

    def _getEmbeddings(self, config='configs/torchmd_config.yml'):
        _, embeddings = Runner()(config)
        return embeddings

    def _constructMSG(self, embeddings_mat):
        simGraph = MolecularSimilarityGraph(
            similarityMap=self.processor.similarityMap,
            tanimotoThreshold=self.tanimoto_threshold,
            embeddings=embeddings_mat)

        simGraph.toGraphData()
        self.sim_graph_data = simGraph.graphData
        self.A_0 = simGraph.initialAdj

    def computeInitAdj(self, config='configs/torchmd_config.yml'):
        embeddings_mat = self._getEmbeddings(config=config)
        self._constructMSG(embeddings_mat=embeddings_mat)

    def predict(self, config):
        if self.A_0 != None:
            pass
        else:
            print("No initial adjacency matrix computed")
