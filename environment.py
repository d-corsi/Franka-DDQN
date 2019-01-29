import math
import numpy as np

class Environment:

    def __init__(self, timeout, step, error):
        self.min_range =  [-50, -10, -50, -95, -50, 90, 0]
        self.max_range =  [ 50, 90, 50, -5, 50, 90, 0]

        self.state = np.array([0, 40, 0, -40, 0, 90, 0])

        self.step = step
        self.timeout = timeout
        self.error = error

    def reset_random(self):
        self.counter = 0
        self.state = np.array([0, 40, 0, -40, 0, 90, 0])

        t1 = np.random.randint(-50, 50)
        t2 = np.random.randint(-10, 90)
        t3 = np.random.randint(-50, 50)
        t4 = np.random.randint(-95, -5)
        t5 = np.random.randint(-50, 50)
        t6 = 90
        t7 = 0

        t1 -= (t1 % 5)
        t2 -= (t2 % 5)
        t3 -= (t3 % 5)
        t4 -= (t4 % 5)
        t5 -= (t5 % 5)

        target_angle = [t1, t2, t3, t4, t5, t6, t7]
        x, y, z = self.ee_pos(target_angle)
        self.target = [x, y, z]

        return np.concatenate((self.norm_state(), self.target ))

    def reset(self, target_angle):
        self.counter = 0
        x, y, z = self.ee_pos( target_angle )
        self.target = [x, y, z]

        return np.concatenate((self.norm_state(), self.target ))

    def perform_action(self, action):
        done = False
        self.counter += 1

        if((action%2) == 0 and (self.state[int(action/2)] + self.step) < self.max_range[int(action/2)]):
            self.state[int(action/2)] += self.step

        if ((action%2) == 1 and (self.state[int(action/2)] - self.step) > self.min_range[int(action/2)]):
            self.state[int(action/2)] -= self.step

        x, y, z = self.ee_pos(self.state)
        distance = math.sqrt(math.pow((x-self.target[0]), 2) + math.pow((y-self.target[1]), 2) + math.pow((z-self.target[2]), 2))

        reward = 0

        # Done if the Panda pushes outside its workspace
        if ((action%2) == 0 and (self.state[int(action/2)] + self.step) > self.max_range[int(action/2)]):
            reward = -1
            done = True

        if ((action%2) == 1 and (self.state[int(action/2)] - self.step) < self.min_range[int(action/2)]):
            reward = -1
            done = True


        if(self.counter >= self.timeout):
            reward = -1
            done = True

        if(distance < self.error):
            reward = 1
            done = True

        return np.concatenate((self.norm_state(), self.target)), reward, done, self.counter

    def norm_state(self):
        normalized_state = []
        for i in range(5):
            #normalized_state.append(self.state[i] / 100)
            deg = self.state[i]
            space = math.fabs(self.max_range[i] - self.min_range[i]) / 2.0
            offset = (self.max_range[i] - space)
            normalized_state.append((deg - offset) / space)

        if(max(normalized_state) > 1 or min(normalized_state) < -1):
            print ("ERROR: INPUT OUT OF RANGE state " + str(max(normalized_state)) + ", " + str(min(normalized_state)))

        return np.array(normalized_state)

    def ee_pos(self, joints):
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
