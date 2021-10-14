import numpy as np
import torch 
import math

#These two functions may not be needed 
class Frontier():
    def __init__(self, map_size, map_depth, device, num_obs, occupancy_map):
        self.map_size = map_size
        self.ch = map_depth
        self.device = device
        self.num_obs = num_obs-1
        self.occupancy_map = occupancy_map
        self.count = 0
        self.obstacles = set()
        self.emptymax = 0.51
        self.emptymin = 0.49

    def getCellValue(self, cell):
        (_, x, y, _) = cell 
        return self.occupancy_map[self.num_obs][x][y]

    def setCellValue(self, cell, cellValue):
        (_, x, y, _) = cell 
        self.occupancy_map[self.num_obs][x][y] = cellValue
        
    def valueisEmpty(self, cell):
        cellValue = self.getCellValue(cell)
        #if cellValue > self.emptymin and cellValue < self.emptymax:
        if cellValue == 0.5:
            return True
        else:
            return False
    
    def getSetWithCellValue(self, set, value):
        for i in range(0, self.map_size):
            for j in range(0, self.map_size):
                cell = (self.num_obs, j, i, self.ch)
                cellType = self.getCellValue(cell)
                if cellType == value:
                    set.add(cell)

    def getSetWithCellValueObstacle(self, set):
        for i in range(0, self.map_size):
            for j in range(0, self.map_size):
                cell = (self.num_obs, j, i, self.ch)
                #if self.getCellValue(cell) > self.emptymax:
                if self.getCellValue(cell) == 1:
                    set.add(cell)                  

    def getSetWithCellValueEmpty(self, set):
        for i in range(0, self.map_size):
            for j in range(0, self.map_size):
                cell = (self.num_obs, j, i, self.ch)
                #if self.getCellValue(cell) > self.emptymin and self.getCellValue(cell) < self.emptymax:
                if self.getCellValue(cell) == 0.5:
                    set.add(cell)
                    
    def isWithinMap(self, cell):
        [_, x, y, _] = cell
        if x >= 0 and x <= self.map_size - 1 and y >= 0 and y <= self.map_size -1:
            return True
        else:
            return False
        
    def getNeighbors(self, cell, includeObstacles=False):
        neighborList = []
        [_, x, y, _] = cell 
        neighborCells = [(self.num_obs, x+i,y+j, self.ch) for i in [-1,0,1] for j in [-1,0,1] if (i != 0 or j != 0) ]
        for neighborCell in neighborCells:
            if includeObstacles:
                if self.isWithinMap(neighborCell):
                    neighborList.append(neighborCell)
            else:
               # if self.isWithinMap(neighborCell) and self.getCellValue(neighborCell) < self.emptymax:
                if self.isWithinMap(neighborCell) and self.getCellValue(neighborCell) != 1:
                    neighborList.append(neighborCell)
        return neighborList

    def isAdjacent(self, cell):
        if self.isWithinMap(cell) and self.valueisEmpty(cell):
            return True
        else: 
            return False

    #TODO: clean up the code 
    def getAdjNeighbors(self,cell):
        neighborList = []
        (num_obs, x, y, ch) = cell

        if self.isAdjacent((num_obs, x+1, y, ch)):
            neighborList.append((num_obs, x+1, y, ch))
        if self.isAdjacent((num_obs, x-1, y, ch)) :
            neighborList.append((num_obs,x-1,y, ch))
        if self.isAdjacent((num_obs,x, y-1, ch)) : 
            neighborList.append((num_obs,x,y-1,ch))
        if self.isAdjacent((num_obs,x, y+1,ch)): 
            neighborList.append((num_obs,x,y+1,ch))
        return neighborList
    
    def getNeighborValuePairs(self, cell):
        neighbors = self.getNeighbors(cell, True)
        neighborValuePairs = []
        for neighbor in neighbors:
            neighborValuePair = (neighbor, self.getCellValue(neighbor))
            neighborValuePairs.append(neighborValuePair)
        return neighborValuePairs

    def getHeuristic(self, cellA, cellB):
        (_, Ax, Ay, _) = cellA
        (_, Bx, By, _) = cellB
        return math.sqrt((Ax - Bx) ** 2 + (Ay - By) ** 2)

    def expandObstacles(self):
        newObstacles = set()
        if not self.obstacles: 
            self.getSetWithCellValueObstacle(self.obstacles)
        
        for obstaclecell in self.obstacles: 
            neighbors = self.getNeighbors(obstaclecell)
            for neighbor in neighbors: 
                self.setCellValue(neighbor, 1)
                newObstacles.add(neighbor)
        
    
    def expandclusters(self, cell, cluster, visited):
        if cell in visited: 
            return
        visited.add(cell)
        neighbors = self.getNeighbors(cell)
        belongToCluster = False
        clusterCandidates = []
        [self.num_obs, x, y, self.ch] = cell 
        for neighbor in neighbors: 
            neighborValue = self.getCellValue(neighbor)
            [self.num_obs, Nx, Ny, self.ch] = neighbor
            if neighborValue == 0:
            #if neighborValue <= self.emptymin: 
                if (Nx - x == 0 or Ny - y == 0):
                    belongToCluster = True
            elif neighborValue == 0.5:
            #elif neighborValue > self.emptymin and neighborValue < self.emptymax: 
                clusterCandidates.append(neighbor)
        if belongToCluster:
            cluster.append(cell)
            for clusterCandidate in clusterCandidates: 
                self.expandclusters(clusterCandidate, cluster,visited)
    
    def calculate_centroid(self, cluster):
        # print("cluster: ", cluster)
        centroid = cluster[0]
        minimum_sum = self.map_size*len(cluster) 
        sum_val = 0
        for cell in cluster: 
            for othercell in cluster: 
                if cell != othercell:
                    sum_val += self.getHeuristic(cell, othercell)
            if sum_val < minimum_sum:
                minimum_sum = sum_val
                centroid = cell 
            sum_val = 0
        return centroid 


    def calculate_clusters(self, clusters):
        emptyset = set()
        self.getSetWithCellValueEmpty(emptyset)
        
        visited = set()
        new_visited = []

        for cell in emptyset: 
            cluster = []
            self.expandclusters(cell, cluster, visited)
            if (len(cluster) > 3):
                clusters.append(cluster)
        clusters.sort(key = lambda tup: len(tup), reverse=True)




