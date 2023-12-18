# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 22:34:17 2023

@author: arun2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.colors import ListedColormap
from collections import namedtuple
import random
import math

class PerceptionMapper:
  def __init__(self, image, resolution):
    self.map = self.initialiseMap(image)
    # height, width
    self.size = self.map.shape
    self.defaultResolution = resolution

  def initialiseMap(self, testImage):
    env = np.ones(testImage.shape)
    for i in range(testImage.shape[0]):
      for j in range(testImage.shape[1]):
        if testImage[i][j] > 125:
          env[i][j] = 0
    # print(env.map[0][0])
    return env

class TreeNode:
   def __init__(self, data):
    self.data = data
    self.xg = data[0]
    self.yg = data[1]
    self.theta = data[2]
    self.vy = data[3]
    self.r_dot = data[4]
    self.parent = None
    self.children = []

class RapidlyExploringRandomTrees():
  def __init__(self, start, goal, stepsize, grid):
    #self.start = start
    #self.goal = goal
    self.start_node = TreeNode(start)
    self.goal_node = TreeNode(goal)
    self.stepsize = stepsize
    self.grid = grid
    #self.xg = start[0]
    #self.yg = start[1]
    #self.theta = start[2]
    #self.vy = start[3]
    #self.r_dot = start[4]
    #self.parent = None
    #self.child = []
    #self.nodes = []
    self.near_node = None
    self.path_distance = 0
    self.numWaypoints = 0
    self.Waypoints = []
    #self.x_goal = goal[0]
    #self.y_goal = goal[1]
    self.m = 1500
    self.vx = 20
    self.Lf = 1.3
    self.Lr = 1.7
    self.Cf = 20000
    self.Cr = 20000
    self.Iz = 6000
    self.min_steer = -np.pi/6
    self.max_steer = np.pi/6
    self.neardist = np.inf

  #def addNode(self, node):
    #self.nodes.append(node)

  def addChildNode(self, data):
    if (data[0] == self.goal_node.xg):
    #goal_distance = self.dist(data[0], self.goal_node.xg, data[1], self.goal_node.yg)
    #if goal_distance < self.stepsize:
      self.near_node.children.append(self.goal_node)
      self.goal_node.parent = self.near_node
    else:
      tempNode = TreeNode(data)
      self.near_node.children.append(tempNode)
      tempNode.parent = self.near_node

  #def output(self):
    #print("Nodes:", self.nodes)

  def randomConfiguration(self):
    x = random.randint(1, self.grid.shape[1])
    y = random.randint(1, self.grid.shape[0])
    theta = random.random()*2*np.pi
    Xrand = [x, y, theta, 0.0, 0.0]
    return Xrand

  def dist(self, x1, x2, y1, y2):
    del_x = x1 - x2
    del_y = y1 - y2
    dist = np.sqrt((del_x)**2 + (del_y)**2)
    return dist

  def nearestneighbor(self, root, Xrand):
    #near = root
    self.neardist = np.inf
    if not root:
      return
    dist = self.dist(root.xg, Xrand[0], root.yg, Xrand[1])
    if dist<self.neardist:
      self.near_node = root
      self.neardist = dist
      print('Nearest Node = ', [self.near_node.xg, self.near_node.yg])

    for child in root.children:
      self.nearestneighbor(child, Xrand)
    pass
    #print("Root:",self.nodes[0])
    #self.near_output(self.nodes[0])
    #self.near_output(Xrand)
    #print("Random point inside loop:", Xrand)
    #for node in self.nodes:
      #if self.dist(node[0], Xrand[0], node[1], Xrand[1]) < self.dist(near[0], Xrand[0], near[1], Xrand[1]):
        #near = node
        #print("Near for every iteration",near)
    #print("NearestNode:", near)
    #return near

  #def near_output(self,x):
    #print(x)

  def dynamic_car_model(self, state, delta_f):
    xg = state.xg
    yg = state.yg
    theta = state.theta
    vy = state.vy
    yaw_rate = state.r_dot
    xg = float(xg)
    yg = float(yg)
    theta = float(theta)
    vy = float(vy)
    yaw_rate = float(yaw_rate)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_delta = np.cos(delta_f)

    A = -(self.Cf*cos_delta + self.Cr)/(self.m*self.vx)
    B = (-self.Lf*self.Cf*cos_delta + self.Lr*self.Cr)/(self.m*self.vx) - self.vx
    C = (-self.Lf*self.Cf*cos_delta + self.Lr*self.Cr)/(self.Iz*self.vx)
    D = -(self.Lf*self.Lf*self.Cf*cos_delta + self.Lr*self.Lr*self.Cr)/(self.Iz*self.vx)
    E = (self.Cf*cos_delta)/self.m
    F = (self.Lf*self.Cf*cos_delta)/self.Iz

    xg_dot = self.vx*cos_t - vy*sin_t
    yg_dot = self.vx*sin_t + vy*cos_t
    theta_dot = yaw_rate
    vy_dot = A*vy + B*yaw_rate + E*delta_f
    r_dot = C*vy + D*yaw_rate + F*delta_f

    dydt = np.array([float(xg_dot), float(yg_dot), float(theta_dot), float(vy_dot), float(r_dot)])
    return dydt

  def new_state(self, state, delta_f):
    k1 = self.dynamic_car_model(state, delta_f)
    new_k1 = TreeNode(k1)
    new_k1_xg = new_k1.xg/2
    new_k1_yg = new_k1.yg/2
    new_k1_theta = new_k1.theta/2
    new_k1_vy = new_k1.vy/2
    new_k1_r_dot = new_k1.r_dot/2
    modified_k1 = [new_k1_xg, new_k1_yg, new_k1_theta, new_k1_vy, new_k1_r_dot]
    used_k2_xg = state.xg + new_k1_xg
    used_k2_yg = state.yg + new_k1_yg
    used_k2_theta = state.theta + new_k1_theta
    used_k2_vy = state.vy + new_k1_vy
    used_k2_r_dot = state.r_dot + new_k1_r_dot
    used_k2 = [used_k2_xg, used_k2_yg, used_k2_theta, used_k2_vy, used_k2_r_dot]
    main_k2_intergration = TreeNode(used_k2)
    k2 = self.dynamic_car_model(main_k2_intergration, delta_f)
    new_k2 = TreeNode(k2)
    new_k2_xg = new_k2.xg/2
    new_k2_yg = new_k2.yg/2
    new_k2_theta = new_k2.theta/2
    new_k2_vy = new_k2.vy/2
    new_k2_r_dot = new_k2.r_dot/2
    modified_k2 = [new_k2_xg, new_k2_yg, new_k2_theta, new_k2_vy, new_k2_r_dot]
    used_k3_xg = state.xg + new_k2_xg
    used_k3_yg = state.yg + new_k2_yg
    used_k3_theta = state.theta + new_k2_theta
    used_k3_vy = state.vy + new_k2_vy
    used_k3_r_dot = state.r_dot + new_k2_r_dot
    used_k3 = [used_k3_xg, used_k3_yg, used_k3_theta, used_k3_vy, used_k3_r_dot]
    main_k3_intergration = TreeNode(used_k3)
    k3 = self.dynamic_car_model(main_k3_intergration, delta_f)
    new_k3 = TreeNode(k3)
    used_k4_xg = state.xg + new_k3.xg
    used_k4_yg = state.yg + new_k3.yg
    used_k4_theta = state.theta + new_k3.theta
    used_k4_vy = state.vy + new_k3.vy
    used_k4_r_dot = state.r_dot + new_k3.r_dot
    used_k4 = [used_k4_xg, used_k4_yg, used_k4_theta, used_k4_vy, used_k4_r_dot]
    main_k4_intergration = TreeNode(used_k4)
    k4 = self.dynamic_car_model(main_k4_intergration, delta_f)
    new_k4 = TreeNode(k4)
    delta_t = 0.2
    new_state_main_xg = state.xg + (new_k1.xg + 2* new_k2.xg + 2* new_k3.xg + new_k4.xg)*delta_t/6
    new_state_main_yg = state.yg + (new_k1.yg + 2* new_k2.yg + 2* new_k3.yg + new_k4.yg)*delta_t/6
    new_state_main_theta = state.theta + (new_k1.theta + 2* new_k2.theta + 2* new_k3.theta + new_k4.theta)*delta_t/6
    new_state_main_vy = state.vy + (new_k1.vy + 2* new_k2.vy + 2* new_k3.vy + new_k4.vy)*delta_t/6
    new_state_main_r_dot = state.r_dot + (new_k1.r_dot + 2* new_k2.r_dot + 2* new_k3.r_dot + new_k4.r_dot)*delta_t/6
    new_state_main = [new_state_main_xg, new_state_main_yg, new_state_main_theta, new_state_main_vy, new_state_main_r_dot]
    new_state_main = TreeNode(new_state_main)
    state_new = new_state_main
    state_new = [float(state_new.xg), float(state_new.yg), float(state_new.theta), float(state_new.vy), float(state_new.r_dot)]
    return state_new

  def selectInput(self, Xrand, Xnear):
    delta_f = self.min_steer
    bestDistance = np.inf
    bestNode = None
    while delta_f < self.max_steer:
      del_input = delta_f
      Xnew = self.new_state(Xnear, del_input)
      distance = self.dist(Xnew[0], Xrand[0], Xnew[1], Xrand[1])
      if distance < bestDistance:
        bestState = Xnew
        bestDelta = del_input
        bestDistance = distance
      delta_f += np.pi/60
    return bestDelta

  def unitVector(self, locationStart, locationEnd):
    v = np.array([locationEnd[0]-locationStart.xg, locationEnd[1] - locationStart.yg])
    u_hat = v/np.linalg.norm(v)
    return u_hat

  def stepsizeCover(self, Xnear, Xnew):
    uni_hat = self.unitVector(Xnear, Xnew)
    offset = self.stepsize * uni_hat
    point = np.array([Xnear.xg + offset[0], Xnear.yg + offset[1], Xnew[2], Xnew[3], Xnew[4]])
    if point[0] >= self.grid.shape[1]:
      point[0] = self.grid.shape[1]
    if point[1] >= self.grid.shape[0]:
      point[1] = self.grid.shape[0]
    return point

  def isObstacle(self, Xnear, new_Xnew):
    if self.grid[round(Xnear.yg), round(Xnear.xg)] == 1:
      return True
    if self.grid[round(new_Xnew[1]), round(new_Xnew[0])] == 1:
      return True
    u_hat = self.unitVector(Xnear, new_Xnew)
    testpoint = np.array([0.0, 0.0])
    for i in range(self.stepsize):
      testpoint[0] = Xnear.xg + i*u_hat[0]
      testpoint[1] = Xnear.yg + i*u_hat[1]
      if self.grid[round(testpoint[1]),round(testpoint[0])] == 1:
          return True
    return False

  def goalFound(self, point):
    if self.dist(self.goal_node.xg, point[0], self.goal_node.yg, point[1]) <= self.stepsize:
      return True
    pass

  def Path(self, goal_n):
    if goal_n.xg == self.start_node.xg:
      return
    self.numWaypoints += 1
    currentPoint = np.array([goal_n.xg, goal_n.yg])
    self.Waypoints.insert(0, currentPoint)
    self.path_distance+=self.stepsize
    self.Path(goal_n.parent) 
    
if __name__ == '__main__':
  testImage1 = img.imread('tb3_house_map.pgm')
  env1 = PerceptionMapper(testImage1, 1)
  map1 = env1.map

  #start = [2.0, 2.0, 2.0, 0.0, 0.0]
  goal = [70.0, 330.0, 0.0, 0.0, 0.0]
  start = [218.0, 184.0, 0.6, 0.0, 0.0]
  #start2 = [300.0, 46.0, 3.2, 0.0, 0.0]
  #start3 = [173.0, 88.0, 2.1, 0.0, 0.0]
  #start4 = [268.0, 190.0, 0.88, 0.0, 0.0]
  stepsize = 10
  grid = map1
  goalRegion = plt.Circle((goal[0], goal[1]), stepsize, color = 'b', fill = False)
  fig = plt.figure("RRT Algorithm")
  #ax = fig.add_subplots(111)
  plt.imshow(grid, cmap = 'binary')
  plt.plot(start[0], start[1], 'ro')
  plt.plot(goal[0], goal[1], 'bo')
  ax = fig.gca()
  ax.add_patch(goalRegion)
  plt.xlabel('X-axis $(m)$')
  plt.ylabel('Y-axis $(m)$')
  rrt = RapidlyExploringRandomTrees(start, goal, stepsize, grid)
  #rrt.addNode(start)
  #rrt.addNode(start1)
  #rrt.addNode(start2)
  #rrt.addNode(start3)
  #rrt.addNode(start4)
  #rrt.output()

  for i in range(200000):
    Xrand = rrt.randomConfiguration()
    print('Xrand:', Xrand)
    #rrt.addNode(Xrand)
    #n_node = rrt.nearestneighbor(rrt.start_node,Xrand)
    rrt.nearestneighbor(rrt.start_node,Xrand)
    #print("NearestNode:", rrt.near_node)
    #rrt.addNode(n_node)
    u = rrt.selectInput(Xrand, rrt.near_node)
    print(u)
    Xnew = rrt.new_state(rrt.near_node, u)
    print("Xnew is:",Xnew)
    new_Xnew = rrt.stepsizeCover(rrt.near_node, Xnew)
    print("New Xnew is:", new_Xnew)
    bool = rrt.isObstacle(rrt.near_node, new_Xnew)
    print(bool)
    if bool == False:
      rrt.addChildNode(new_Xnew)
      plt.pause(0.10)
      plt.plot([rrt.near_node.xg, new_Xnew[0]], [rrt.near_node.yg, new_Xnew[1]], 'go', linestyle='--')
      if(rrt.goalFound(new_Xnew)):
        rrt.addChildNode(goal)
        print("Goal found!")
        break
      
  rrt.Path(rrt.goal_node)
  rrt.Waypoints.insert(0, start)
  print("Number of Waypoints:", rrt.numWaypoints)
  print("Path Distance:", rrt.path_distance)
  print("Waypoints:", rrt.Waypoints)
  
  for i in range(len(rrt.Waypoints)-1):
    plt.plot([rrt.Waypoints[i][0], rrt.Waypoints[i+1][0]],[rrt.Waypoints[i][1], rrt.Waypoints[i+1][1]], 'ro', linestyle="--")
    plt.pause(0.10)