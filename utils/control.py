import numpy as np 
import math

def getDistance(start, target):
    Ax, Ay = start
    Bx, By = target
    return math.sqrt((Ax - Bx) ** 2 + (Ay - By) ** 2)

def normalize_to_pi(rad):
    angle = rad% (np.pi * 2)

    if angle > np.pi : 
        angle = angle - (np.pi * 2 )
    return angle

def normalize_to_2pi(rad): 
    angle = rad % (np.pi *2)
    return angle 


def get_action(start, waypoints, agentRot, prev_pos, path_found):
    action = 5 
    collision = 0 

    if (len(waypoints) <2 ):
        path_found = False
        waypoints = []
        prev_pos = (0,0)

    if path_found: 

        if (start == prev_pos):
            #print("****************************COLLISION !!!!*************")
            waypoints = []
            path_found = False
            collision = 1

        else: 

            next_target = (waypoints[-2][1], waypoints[-2][2])
            target = (waypoints[-1][1], waypoints[-1][2])

            agent_to_next_target = getDistance(start,next_target )
            agent_to_target = getDistance(start, target) 
            target_to_next_target = getDistance(target, next_target)
            
            if ( agent_to_next_target <= target_to_next_target or agent_to_target <= 1):
                waypoints = waypoints[:-1]
                target = (waypoints[-1][1], waypoints[-1][2])
                
            angleToGlobalPos = normalize_to_pi(math.atan2(target[1] - start[1], start[0] - target[0]))
            agentRot = normalize_to_pi(-agentRot)

            angle_diff = (normalize_to_2pi(angleToGlobalPos) - normalize_to_2pi(agentRot))

            #print("A: ", start, ", ", agentRot, " T: ", target, "," , angleToGlobalPos, "AD: ", AD)

            if (abs(normalize_to_pi(angle_diff)) > 0.50 ):
                if (angle_diff <0 ):
                    if (abs(angle_diff) < 180):
                        action = 2
                    else: 
                        action =  3
                else: 
                    if (abs(angle_diff) < 180):
                        action =  3
                    else: 
                        action = 2 
            else: 
                # print("=========================================go forward")
                action =  1
                prev_pos = start
    else: 
        prev_pos = (0,0)
        waypoints = []
        path_found = False
    return action, prev_pos, waypoints, path_found, collision


