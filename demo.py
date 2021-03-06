import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import json
import os

from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
#hand_estimation = Hand('model/hand_pose_model.pth')


def wjson(root_path,img_path):
    test_image = os.path.join(root_path,img_path)
    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    h,w,c = oriImg.shape
    '''
    for i in range(len(candidate)):
        x, y = candidate[i][0:2]
        print('candidate: ',int(x),int(y))
    '''
    res = {'shapes':[],
                'imagePath':img_path,
                'flags':{},
                'version':'4.5.4',
                'imageData':None,
                'imageWidth':w,
                'imageHeight':h
    }
    for i in range(18):
            for n in range(len(subset)):
                index = int(subset[n][i])
                if index == -1:
                    continue
                x, y = candidate[index][0:2]
                temp_dic = {'shape_type': 'point', 
                                        'points': [], 
                                        'flags': {}, 
                                        'group_id': n, 
                                        'label': str(i)
                }
                temp_dic['points'].append([int(x),int(y)])
                res['shapes'].append(temp_dic)
                #print('第',n,'个人的第',i,'个关节点的坐标：',int(x),int(y))
    #print('subset: ',subset)
    res_json = json.dumps(res)
    wight_json = open(os.path.join(root_path,img_path[:-4]+'.json'), 'w')
    wight_json.write(res_json)
    wight_json.close()
    print(img_path,'ok!!!')
#plt.imshow(canvas[:, :, [2, 1, 0]])
#plt.axis('off')
#plt.show()
root_path = './data/tennis/'
files = os.listdir(root_path)
files.sort()
for file in files:
    wjson(root_path,file)
