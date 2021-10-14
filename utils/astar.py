from queue import PriorityQueue
import torch 
from  utils.frontier import Frontier
import numpy as np
import math

def valueisEmpty(map, cell):
    cellValue = getCellValue(map, cell)
    if cellValue > 0.49 and cellValue < 0.51:
        return True
    else:
        return False

def isWithinMap(cell, map_size):
    [_, x, y, _] = cell
    if x >= 0 and x <= map_size - 1 and y >= 0 and y <= map_size -1:
        return True
    else:
        return False

def isAdjacent(map, cell, map_size):
    if isWithinMap(cell, map_size) and valueisEmpty(map, cell):
        return True
    else: 
        return False

def getAdjNeighbors(map, cell, map_size):
    neighborList = []
    (num_obs, x, y, ch) = cell

    if isAdjacent(map, (num_obs, x+1, y, ch), map_size):
        neighborList.append((num_obs, x+1, y, ch))
    if isAdjacent(map, (num_obs, x-1, y, ch), map_size) :
        neighborList.append((num_obs,x-1,y, ch))
    if isAdjacent(map, (num_obs,x, y-1, ch), map_size) : 
        neighborList.append((num_obs,x,y-1,ch))
    if isAdjacent(map, (num_obs,x, y+1,ch), map_size): 
        neighborList.append((num_obs,x,y+1,ch))
    return neighborList
    

def getDistance(start, target):
    (_, Ax, Ay, _) = start
    (_, Bx, By, _) = target
    return math.sqrt((Ax - Bx) ** 2 + (Ay - By) ** 2)

def getCellValue(map, cell):
    (_, x, y, _) = cell 
    return map[0][x][y]

def setCellValue(map, cell, cellValue):
    (_, x, y, _) = cell 
    map[0][x][y] = cellValue

def astar_algorithm(map, start, end, num_obs, map_size, occupancy_map_channels, device):
    start = (num_obs-1, start[0], start[1], occupancy_map_channels)
    #print("===============start: ", start, "  end: ", end)

    count = 0
    empty_set = PriorityQueue()
    empty_set.put((0,count,start))
    prev_grid =  {} #keep track of the last node it came from
    g_score_grid_init = torch.ones(num_obs, map_size, map_size, occupancy_map_channels, device = device)*np.inf
    f_score_grid_init = torch.ones(num_obs, map_size, map_size, occupancy_map_channels, device = device)*np.inf
    h_val_init = torch.ones(num_obs, map_size, map_size, occupancy_map_channels, device = device)*np.inf

    setCellValue(g_score_grid_init, start, 0)   
    setCellValue(f_score_grid_init, start, getDistance(start,end))
    
    # create empty_set_hash to keep track of empty_set
    # PriorityQueue does not have a datastructure that checks if a value exists in the set
    empty_set_hash = set()
    empty_set_hash.add(start)
    #debugging purpose: 
    considered_cells = set()
    considered_cells.add(start)

    path = []
    foundPath = False
    while not empty_set.empty() and foundPath == False:
        current = empty_set.get()[2]
        empty_set_hash.remove(current)
        if current == end:
            #print("found the path!")
            foundPath = True
            while (current in prev_grid) and (current != start) :
                path.append(current)
                current = prev_grid[current]

        neighbors = getAdjNeighbors(map, current, map_size)
        
        for neighbor in neighbors:
            considered_cells.add(neighbor)
            temp_g_score = getCellValue(g_score_grid_init, current) + 1 
            if temp_g_score < getCellValue(g_score_grid_init, neighbor):
                prev_grid[neighbor] = current
                setCellValue(g_score_grid_init, neighbor, temp_g_score) 
                setCellValue(f_score_grid_init, neighbor, temp_g_score + getDistance(neighbor, end))
                setCellValue(h_val_init, neighbor,  getDistance(neighbor, end))
                if neighbor not in empty_set_hash:
                    count += 1
                    v = getCellValue(f_score_grid_init, neighbor)
                    empty_set.put((v, count, neighbor))
                    empty_set_hash.add(neighbor)
    
    if foundPath == False: 
        path = []
        #find the smallest f score cell 
        minidx = torch.argmin(h_val_init)
        a = minidx//map_size 
        b = minidx%map_size 
        mincell = (0, a.item(),b.item(), 1)
        current = mincell
        while (mincell in prev_grid) and (current != start):
            path.append(current)
            current = prev_grid[current]

        if len(path) != 0: 
            foundPath = True
           # print("found the closest exploration point")


 #   print("smallest f score index: ", mincell , " value: ", getCellValue(f_score_grid_init, mincell))
 #   print("end index: ", end, "value: ", getCellValue(f_score_grid_init, end))


    return considered_cells, path, foundPath
                  

