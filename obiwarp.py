from scipy.interpolate import PchipInterpolator
import numpy as np
import pandas as pd


def get_same_num(path):
    last_x,last_y = path[-1][0],path[-1][1]
    last_same_num = 0
    id = -2
    idx,idy = path[id][0],path[id][1]
    while idx == last_x:
        last_same_num += 1
        id -= 1
        idx,idy = path[id][0],path[id][1]
    id = -2
    while idy == last_y:
        last_same_num += 1 
        id -= 1
        idx,idy = path[id][0],path[id][1]
    
    # 头部过渡数
    first_x,first_y = path[0][0],path[0][1]
    first_same_num = 0
    id = 1
    idx,idy = path[id][0],path[id][1]
    while idx == first_x:
        first_same_num += 1
        id += 1
        idx,idy = path[id][0],path[id][1]
    id = 1
    while idy == first_y:
        first_same_num += 1 
        id += 1
        idx,idy = path[id][0],path[id][1]
    return first_same_num + 1,last_same_num + 1


def get_anchor(path):
    first_same_num, last_same_num = get_same_num(path)
    anchors = [path[0]]
    best_anchor = (0,0,np.inf)
    for i in range(len(path))[first_same_num:-1*last_same_num]:
        last,current,next = path[i-1],path[i],path[i+1]
        if current[0] == anchors[-1][0] or current[1] == anchors[-1][1]:
            continue
        # 对角线
        if last[0] != current[0] and last[1] != current[1]:
            # 对角线 + 对角线
            if current[0] != next[0] and current[1] != next[1]:
                anchors.append(current)
            # 对角线 + 水平/垂直
            else:
                best_anchor = current
        # 水平
        elif last[1] == current[1]:
            # 水平 + 垂直
            if current[0] == next[0]:
                if best_anchor == (0,0,np.inf) or best_anchor[2] > current[2]:
                    best_anchor = current
                    anchors.append(best_anchor)
                    best_anchor = (0,0,np.inf)
                else:
                    anchors.append(best_anchor)
                    best_anchor = (0,0,np.inf)
            # 水平 + 对角线
            elif current[0] != next[0] and current[1] != next[1]:
                if best_anchor[2] > current[2]:
                    best_anchor = current
                anchors.append(best_anchor)
                best_anchor = (0,0,np.inf)
            # 水平 + 水平
            else:
                if best_anchor[2] > current[2]:
                    best_anchor = current
        
        # 垂直
        else:
            # 垂直 + 水平
            if current[1] == next[1]:
                if best_anchor == (0,0,np.inf) or best_anchor[2] > current[2]:
                    best_anchor = current
                    anchors.append(best_anchor)
                    best_anchor = (0,0,np.inf)
                else:
                    anchors.append(best_anchor)
                    best_anchor = (0,0,np.inf)
            # 垂直 + 对角线
            elif current[0] != next[0] and current[1] != next[1]:
                if best_anchor[2] > current[2]:
                    best_anchor = current
                anchors.append(best_anchor)
                best_anchor = (0,0,np.inf)
            # 垂直 + 垂直
            else:
                if best_anchor[2] > current[2]:
                    best_anchor = current
    anchors.append(path[-1])
    return anchors


def obi_f(dfx,dfy,anchor):
    anchor_x = [i for (i,j,_) in anchor]
    anchor_y = [j for (i,j,_) in anchor]
    pchip = PchipInterpolator(dfx.index[anchor_x],dfy.index[anchor_y])
    return pchip


def sample_interpolate(old_sample,ref,pchip):
    rt = old_sample.index
    new_rt = pchip(rt)
    new_sample = pd.DataFrame(index=ref.index,columns=old_sample.columns)
    for col in old_sample.columns:
        interpolator = PchipInterpolator(old_sample.index, old_sample[col])
        new_sample[col] = interpolator(new_rt)
    return new_sample