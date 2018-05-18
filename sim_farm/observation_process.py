#
# this file is compiled from https://github.com/ctmakro/stanford-osrl
# 

import numpy as np

'''
above was copied from 'osim-rl/osim/env/run.py'.

observation:
0 pelvis r
1 x
2 y

3 pelvis vr
4 vx
5 vy

6-11 hip_r .. ankle_l [joint angles] # 7->knee_r, 10->knee_l

12-17 hip_r .. ankle_l [joint velocity]

18-19 mass_pos xy
20-21 mass_vel xy

22-(22+7x2-1=35) bodypart_positions(x,y) ## 22->head_x, 1->pelvis_x

36-37 muscles psoas

38-40 obstacles
38 x dist
39 y height
40 radius

radius of heel and toe ball: 0.05

'''

# 41 dim to 48 dim
def process_observation(observation):
    o = list(observation)  # an array

    pr = o[0]

    px = o[1]
    py = o[2]

    pvr = o[3]

    pvx = o[4]
    pvy = o[5]

    for i in range(6, 18):
        o[i] /= 4

    # a copy of original y, not relative y.
    o = o + [o[22+i*2+1]-0.9 for i in range(7)]

    # x and y relative to pelvis
    for i in range(7):  # head pelvis torso, toes and taluses
        o[22+i*2+0] -= px
        o[22+i*2+1] -= py

    o[18] -= px  # mass pos xy made relative
    o[19] -= py
    o[20] -= pvx  # mass vel xy made relative
    o[21] -= pvy

    o[38] = min(8, o[38])/7  # ball info are included later in the stage
    o[39] *= 5
    o[40] *= 5

    o[0] /= 2  # divide pr by 4
    o[1] = 0  # abs value of pel x should not be included
    o[2] -= 0.9  # minus py by 0.5

    o[3] /= 4  # divide pvr by 4
    o[4] /= 8  # divide pvx by 10
    o[5] /= 1  # pvy is okay

    o[20] /= 1
    o[21] /= 1

    return o


_stepsize = 0.01


def flatten(l): return [item for sublist in l for item in sublist]


# expand observation from 48 to 48*7 dims
processed_dims = 48 + 14*1 + 3*3 + 1*0 + 8
# processed_dims = 41*8


def generate_observation(new, old):
    global _stepsize
    # deal with old
    if old is None:
        raise 'old observation should be given'

    # process new
    final_observation = process_observation(new)

    def bodypart_velocities():
        return [(new[i]-old[i])/_stepsize for i in range(22, 36)]

    def relative_bodypart_velocities():
        # velocities, but relative to pelvis.
        bv = bodypart_velocities()
        pv1, pv2 = bv[2], bv[3]
        for i in range(len(bv)):
            if i % 2 == 0:
                bv[i] -= pv1
            else:
                bv[i] -= pv2
        return bv

    #vels = bodypart_velocities()  # [14]

    relvels = relative_bodypart_velocities()  # [14]
    
    #accs = [
    #        (vels[idx] - vels[idx])/_stepsize
    #        for idx in range(len(vels))
    #        ] # [14]

    final_observation += relvels   # 14dim


    foot_touch_indicators = []
    for i in [29, 31, 33, 35]:  # y of toes and taluses
        # touch_ind = 1 if new[i] < 0.05 else 0
        touch_ind = np.clip((0.0 - new[i]) * 5 + 0.5, 0., 1.)
        touch_ind2 = np.clip((0.1 - new[i]) * 5 + 0.5, 0., 1.)
        # touch_ind2 = 1 if new[i] < 0.1 else 0
        foot_touch_indicators.append(touch_ind)
        foot_touch_indicators.append(touch_ind2)
    
    final_observation += foot_touch_indicators  # 8dim

    def final_processing(l):
        # normalize to prevent excessively large input
        for idx in range(len(l)):
            if l[idx] > 1:
                l[idx] = np.sqrt(l[idx])
            if l[idx] < -1:
                l[idx] = - np.sqrt(-l[idx])
    
    final_processing(final_observation)

    return final_observation

def get_observation_space():
    import math
    test = np.random.randn(41,)
    obs = generate_observation(test,test)
    ones = np.ones_like(obs)
    return {'high':-math.pi * ones, 'low':math.pi * ones, 'shape':ones.shape}
