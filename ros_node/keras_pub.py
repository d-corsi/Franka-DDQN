#!/usr/bin/env python
# license removed for brevity
from __future__ import print_function

import rospy
from std_msgs.msg import String
from std_msgs.msg import Header
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA

import numpy as np
import math
import os

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, InputLayer, Dropout

'''
def normalized_state(deg_vec):
  normalizedState = []
  for deg in (deg_vec):
    normalizedState.append(deg / 100.0)
  return normalizedState
'''

def normalized_state(deg_vec):
  minRange =  [-50,   -10,   -50,   -95,   -50,    0,   -50]
  maxRange =  [ 50,    90,    50,    -5,    50,   95,    50]

  normalizedState = []
  for i in range(7):
    deg = deg_vec[i]
    space = math.fabs(maxRange[i] - minRange[i]) / 2.0
    delay = (maxRange[i] - space)
    normalizedState.append((deg - delay) / space)

    if(max(normalizedState) > 1 or min(normalizedState) < -1):
      print ("ERROR: INPUT OUT OF RANGE state " + str(max(normalizedState)) + ", " + str(min(normalizedState)))

  return normalizedState

def deg2rad(deg_vec):
    rad_vec = []
    for i in range(9):
      rad_vec.append(math.radians(deg_vec[i]))
    return rad_vec

def compute_new_state(state, action):
  minRange =  [-50,   -10,   -50,   -95,   -50,    0,   -50]
  maxRange =  [ 50,    90,    50,    -5,    50,   95,    50]

  step = 2

  if ( (action%2) == 0 and (state[int(action/2)] + step ) < maxRange[int(action/2)] ):
    state[int(action/2)] += step

  if ( (action%2) == 1 and (state[int(action/2)] - step ) > minRange[int(action/2)] ):
    state[int(action/2)] -= step

  return state

def generate_target():

  t1 = np.random.randint(-50, 50)
  t2 = np.random.randint(-10, 90)
  t3 = np.random.randint(-50, 50)
  t4 = np.random.randint(-95, -5)
  t5 = np.random.randint(-50, 50)
  t6 = np.random.randint(  0, 95)
  t7 = np.random.randint(-50, 50)

  target_angle = [t1, t2, t3, t4, t5, t6, t7]

  x, y, z = endEffectorPos( target_angle )
  return [x, y, z]

def marker_spawn(marker_publisher, target, radius, color, marker_id):
    robotMarker = Marker()
    robotMarker.header.frame_id = 'panda_link0'
    robotMarker.header.stamp    = rospy.get_rostime()
    robotMarker.type = 2 # sphere

    robotMarker.id = marker_id

    robotMarker.pose.position.x = target[0]
    robotMarker.pose.position.y = target[1]
    robotMarker.pose.position.z = target[2]

    robotMarker.scale.x = radius
    robotMarker.scale.y = radius
    robotMarker.scale.z = radius

    robotMarker.color.r = color[0]
    robotMarker.color.g = color[1]
    robotMarker.color.b = color[2]
    robotMarker.color.a = 1.0

    marker_publisher.publish(robotMarker)

def endEffectorPos(joints):
  t1 = joints[0]
  t2 = joints[1]
  t3 = joints[2]
  t4 = joints[3]
  t5 = joints[4]
  t6 = joints[5]
  t7 = joints[6]

  x = (33*math.cos((math.pi*t5)/180)*(math.cos((math.pi*t4)/180)*(math.sin((math.pi*t1)/180)*math.sin((math.pi*t3)/180) - math.cos((math.pi*t1)/180)*math.cos((math.pi*t2)/180)*math.cos((math.pi*t3)/180)) - math.cos((math.pi*t1)/180)*math.sin((math.pi*t2)/180)*math.sin((math.pi*t4)/180)))/400 - (11*math.sin((math.pi*t7)/180)*(math.sin((math.pi*t5)/180)*(math.cos((math.pi*t4)/180)*(math.sin((math.pi*t1)/180)*math.sin((math.pi*t3)/180) - math.cos((math.pi*t1)/180)*math.cos((math.pi*t2)/180)*math.cos((math.pi*t3)/180)) - math.cos((math.pi*t1)/180)*math.sin((math.pi*t2)/180)*math.sin((math.pi*t4)/180)) - math.cos((math.pi*t5)/180)*(math.cos((math.pi*t3)/180)*math.sin((math.pi*t1)/180) + math.cos((math.pi*t1)/180)*math.cos((math.pi*t2)/180)*math.sin((math.pi*t3)/180))))/125 + (79*math.cos((math.pi*t1)/180)*math.sin((math.pi*t2)/180))/250 + (11*math.cos((math.pi*t7)/180)*(math.sin((math.pi*t6)/180)*(math.sin((math.pi*t4)/180)*(math.sin((math.pi*t1)/180)*math.sin((math.pi*t3)/180) - math.cos((math.pi*t1)/180)*math.cos((math.pi*t2)/180)*math.cos((math.pi*t3)/180)) + math.cos((math.pi*t1)/180)*math.cos((math.pi*t4)/180)*math.sin((math.pi*t2)/180)) - math.cos((math.pi*t6)/180)*(math.cos((math.pi*t5)/180)*(math.cos((math.pi*t4)/180)*(math.sin((math.pi*t1)/180)*math.sin((math.pi*t3)/180) - math.cos((math.pi*t1)/180)*math.cos((math.pi*t2)/180)*math.cos((math.pi*t3)/180)) - math.cos((math.pi*t1)/180)*math.sin((math.pi*t2)/180)*math.sin((math.pi*t4)/180)) + math.sin((math.pi*t5)/180)*(math.cos((math.pi*t3)/180)*math.sin((math.pi*t1)/180) + math.cos((math.pi*t1)/180)*math.cos((math.pi*t2)/180)*math.sin((math.pi*t3)/180)))))/125 - (33*math.cos((math.pi*t4)/180)*(math.sin((math.pi*t1)/180)*math.sin((math.pi*t3)/180) - math.cos((math.pi*t1)/180)*math.cos((math.pi*t2)/180)*math.cos((math.pi*t3)/180)))/400 + (48*math.sin((math.pi*t4)/180)*(math.sin((math.pi*t1)/180)*math.sin((math.pi*t3)/180) - math.cos((math.pi*t1)/180)*math.cos((math.pi*t2)/180)*math.cos((math.pi*t3)/180)))/125 + (33*math.sin((math.pi*t5)/180)*(math.cos((math.pi*t3)/180)*math.sin((math.pi*t1)/180) + math.cos((math.pi*t1)/180)*math.cos((math.pi*t2)/180)*math.sin((math.pi*t3)/180)))/400 + (48*math.cos((math.pi*t1)/180)*math.cos((math.pi*t4)/180)*math.sin((math.pi*t2)/180))/125 + (33*math.cos((math.pi*t1)/180)*math.sin((math.pi*t2)/180)*math.sin((math.pi*t4)/180))/400
  y = (11*math.sin((math.pi*t7)/180)*(math.sin((math.pi*t5)/180)*(math.cos((math.pi*t4)/180)*(math.cos((math.pi*t1)/180)*math.sin((math.pi*t3)/180) + math.cos((math.pi*t2)/180)*math.cos((math.pi*t3)/180)*math.sin((math.pi*t1)/180)) + math.sin((math.pi*t1)/180)*math.sin((math.pi*t2)/180)*math.sin((math.pi*t4)/180)) - math.cos((math.pi*t5)/180)*(math.cos((math.pi*t1)/180)*math.cos((math.pi*t3)/180) - math.cos((math.pi*t2)/180)*math.sin((math.pi*t1)/180)*math.sin((math.pi*t3)/180))))/125 - (33*math.cos((math.pi*t5)/180)*(math.cos((math.pi*t4)/180)*(math.cos((math.pi*t1)/180)*math.sin((math.pi*t3)/180) + math.cos((math.pi*t2)/180)*math.cos((math.pi*t3)/180)*math.sin((math.pi*t1)/180)) + math.sin((math.pi*t1)/180)*math.sin((math.pi*t2)/180)*math.sin((math.pi*t4)/180)))/400 + (79*math.sin((math.pi*t1)/180)*math.sin((math.pi*t2)/180))/250 + (33*math.cos((math.pi*t4)/180)*(math.cos((math.pi*t1)/180)*math.sin((math.pi*t3)/180) + math.cos((math.pi*t2)/180)*math.cos((math.pi*t3)/180)*math.sin((math.pi*t1)/180)))/400 - (48*math.sin((math.pi*t4)/180)*(math.cos((math.pi*t1)/180)*math.sin((math.pi*t3)/180) + math.cos((math.pi*t2)/180)*math.cos((math.pi*t3)/180)*math.sin((math.pi*t1)/180)))/125 - (33*math.sin((math.pi*t5)/180)*(math.cos((math.pi*t1)/180)*math.cos((math.pi*t3)/180) - math.cos((math.pi*t2)/180)*math.sin((math.pi*t1)/180)*math.sin((math.pi*t3)/180)))/400 - (11*math.cos((math.pi*t7)/180)*(math.sin((math.pi*t6)/180)*(math.sin((math.pi*t4)/180)*(math.cos((math.pi*t1)/180)*math.sin((math.pi*t3)/180) + math.cos((math.pi*t2)/180)*math.cos((math.pi*t3)/180)*math.sin((math.pi*t1)/180)) - math.cos((math.pi*t4)/180)*math.sin((math.pi*t1)/180)*math.sin((math.pi*t2)/180)) - math.cos((math.pi*t6)/180)*(math.cos((math.pi*t5)/180)*(math.cos((math.pi*t4)/180)*(math.cos((math.pi*t1)/180)*math.sin((math.pi*t3)/180) + math.cos((math.pi*t2)/180)*math.cos((math.pi*t3)/180)*math.sin((math.pi*t1)/180)) + math.sin((math.pi*t1)/180)*math.sin((math.pi*t2)/180)*math.sin((math.pi*t4)/180)) + math.sin((math.pi*t5)/180)*(math.cos((math.pi*t1)/180)*math.cos((math.pi*t3)/180) - math.cos((math.pi*t2)/180)*math.sin((math.pi*t1)/180)*math.sin((math.pi*t3)/180)))))/125 + (48*math.cos((math.pi*t4)/180)*math.sin((math.pi*t1)/180)*math.sin((math.pi*t2)/180))/125 + (33*math.sin((math.pi*t1)/180)*math.sin((math.pi*t2)/180)*math.sin((math.pi*t4)/180))/400
  z = (79*math.cos((math.pi*t2)/180))/250 + (11*math.sin((math.pi*t7)/180)*(math.sin((math.pi*t5)/180)*(math.cos((math.pi*t2)/180)*math.sin((math.pi*t4)/180) - math.cos((math.pi*t3)/180)*math.cos((math.pi*t4)/180)*math.sin((math.pi*t2)/180)) - math.cos((math.pi*t5)/180)*math.sin((math.pi*t2)/180)*math.sin((math.pi*t3)/180)))/125 + (11*math.cos((math.pi*t7)/180)*(math.cos((math.pi*t6)/180)*(math.cos((math.pi*t5)/180)*(math.cos((math.pi*t2)/180)*math.sin((math.pi*t4)/180) - math.cos((math.pi*t3)/180)*math.cos((math.pi*t4)/180)*math.sin((math.pi*t2)/180)) + math.sin((math.pi*t2)/180)*math.sin((math.pi*t3)/180)*math.sin((math.pi*t5)/180)) + math.sin((math.pi*t6)/180)*(math.cos((math.pi*t2)/180)*math.cos((math.pi*t4)/180) + math.cos((math.pi*t3)/180)*math.sin((math.pi*t2)/180)*math.sin((math.pi*t4)/180))))/125 + (48*math.cos((math.pi*t2)/180)*math.cos((math.pi*t4)/180))/125 + (33*math.cos((math.pi*t2)/180)*math.sin((math.pi*t4)/180))/400 - (33*math.cos((math.pi*t5)/180)*(math.cos((math.pi*t2)/180)*math.sin((math.pi*t4)/180) - math.cos((math.pi*t3)/180)*math.cos((math.pi*t4)/180)*math.sin((math.pi*t2)/180)))/400 - (33*math.cos((math.pi*t3)/180)*math.cos((math.pi*t4)/180)*math.sin((math.pi*t2)/180))/400 + (48*math.cos((math.pi*t3)/180)*math.sin((math.pi*t2)/180)*math.sin((math.pi*t4)/180))/125 - (33*math.sin((math.pi*t2)/180)*math.sin((math.pi*t3)/180)*math.sin((math.pi*t5)/180))/400 + 0.333

  return x, y, z

def load_model():
  model = Sequential()
  model.add(Dense(100, input_shape = (10, ), activation='tanh'))
  model.add(Dense(100, activation = 'tanh'))
  model.add(Dense(100, activation = 'tanh'))
  model.add(Dense(14, activation = 'linear'))
  model.compile(loss='mse', optimizer='adam', metrics=['mae'])

  model.load_weights("backup.h5")

  return model

def talker():
    marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=10)


    pub = rospy.Publisher('joint_states', JointState, queue_size=10)
    rospy.init_node('joint_state_publisher')
    rate = rospy.Rate(30) # 10hz (1 slow - 100 fast)
    model = load_model()

    success = 0

    for sample in range (3):

      state_deg = [0, 40, 0, -40, 0, 0, 0, 0, 0]
      target = generate_target()

      target_fake = [0.24996699759338026, -0.42281411738967556, 0.5220085954218298]

      target_middle = [0.37450022166722785, -0.3545424758397752, 0.6195838860925136]
      target_base = [0.6193646599546465, 0.02470997975543915, 0.0014545226912595077]

      target = target_middle

      #while not rospy.is_shutdown():
      for _ in range(150):
        input_layer = np.concatenate((normalized_state(state_deg[0:7]), target))
        action = np.argmax(model.predict(np.array([input_layer])))
        state_deg = compute_new_state(state_deg, action)
        state_deg[5] = 90

        state_rad = deg2rad(state_deg)

        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
        msg.position = state_rad
        msg.velocity = []
        msg.effort = []

        #rospy.loginfo(msg)
        pub.publish(msg)

        marker_spawn(marker_publisher, target, 0.05, [1, 0, 0], 0)
        #marker_spawn(marker_publisher, target_fake, 0.08, [1, 0, 0], 0)


        x, y, z = endEffectorPos( state_deg[0:7] )
        distance = math.sqrt( math.pow((x-target[0]), 2) + math.pow((y-target[1]), 2) + math.pow((z-target[2]), 2) )
        if(distance < 0.05):
          success += 1
          break

        rate.sleep()

      print ("Success :" + str(success) + "/" + str(sample + 1))
      print (target)
      

if __name__ == '__main__':
  try:
    talker()
  except rospy.ROSInterruptException:
    pass