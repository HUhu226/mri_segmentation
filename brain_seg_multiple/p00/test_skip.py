'''
对训练好的U-Net模型进行测试，并计算在测试集上的Dice系数
'''

#coding:utf8
from UNet import *
from sets import *
import torch as t
from tqdm import tqdm
import numpy
import time
import os

from PIL import Image


import os
import numpy as np

# 语义分割结果保存路径
segmentation_result_path = '/tmp/code/brain_seg/p00/segmentation_result/'
# 如果路径不存在，则创建它
os.makedirs(segmentation_result_path, exist_ok=True)

# 检查模型参数文件的存储路径
check_ing_path = '/tmp/code/brain_seg/p00/check/pth/'        #模型参数文件的存储路径

# 模型测试计数器和步长
start_index = 0
step = 2

# 加载预测记录
file_record = '/tmp/code/brain_seg/p00/dice/prediction_record_w.npy'
if not os.path.exists(file_record):
    prediction_record = []  # 创建一个新的记录列表
else:
    # 加载已有的记录
    prediction_record = np.load(file_record, allow_pickle=True)
    prediction_record = prediction_record.tolist()  # 转换为Python列表
     


if 1 > 0:

    val_dice = []                                      #用于储存计算结果
    val_std = []


    check_ing_path = check_ing_path + 'weight/'

    check_list = sorted(os.listdir(check_ing_path),key=lambda x: os.path.getmtime(os.path.join(check_ing_path, x)))
    #最后修改时间从旧到新
    #check_list.reverse() 
    
    print('Check-length: ',len(check_list))
    read_list = os.listdir(check_ing_path)
    mean_dice_list = []
    std_dice_list = []
    #model_feature = getattr(models, 'unet_3dd')() 
    #model_feature.cuda()
    
    for index, checkname in enumerate(check_list):
        
        print(index, checkname)
        
        #if checkname != 'un' and checkname != '01':
        if checkname not in [item[0] for item in prediction_record]:
            model = U_Net() 
            #model.eval()
            model.load_state_dict(torch.load(check_ing_path+checkname))
            model.eval()

            if opt.use_gpu: model.cuda()


            if 1 > 0:

                #os.mkdir('userhome/ye/2021_07/NET06/05/NET01/baseline00/00/check/dice_time_train/'+checkname+'/')

                testpath = '/tmp/code/brain_seg/data00/test/0010/'
                folderlist = os.listdir(testpath)
                #folderlist.extend(list(np.load('/userhome/ye/DATA_ALL/jidayi_naobaizi/filename/data/2.npy')))
                #folderlist.extend(list(np.load('/userhome/ye/DATA_ALL/jidayi_naobaizi/filename/data/3.npy')))
                #folderlist.extend(list(np.load('/userhome/ye/DATA_ALL/jidayi_naobaizi/filename/data/4.npy'))) 


                WT_dice = []
                label_list = [ 'WM', 'GM', 'CSF', 'Bone', 'Scalp', 'Eye Balls', 'Compact Bone', 'Spongy Bone', 'Blood', 'Muscle' ]
                # Initialize a list to store Dice scores for each label
                label_dice_scores = [[] for _ in range(10)]
                

                for foldername in folderlist:
                    
                    print(f'Testing... File Name: {testpath}{foldername}')
                    
                    data = np.load(testpath+foldername)                    
                    #print(data.shape)
                    
                    #vector = np.zeros((2,data.shape[1],data.shape[2],data.shape[3]))
                    #vector = data[0:2,:,:,:].astype(float)


                    #gh = np.zeros((4,160,160,6))

                    tru = data[1,:,:,:]  #2改成1
                    

                    prob = np.zeros((11,data.shape[1],data.shape[2],data.shape[3]))   #改成11
                    flag = np.zeros((2,data.shape[1],data.shape[2],data.shape[3]))


                    g = 0
                    s0 = 32 
                    s1 = 48
                    ss = 128
                    sss = 192
                    
                    #data 2,W,H,L
                    
                    ###
                    ###
                    ###
                    fast_stride = 20
                    
                    #####?切片分段
                    for iii in tqdm(range(50), desc = 'Processing Slices', ncols = 100):
                        if iii*fast_stride < data.shape[1]:
                            if fast_stride*(iii+1) <= data.shape[1]:
                                vector = data[0:1,fast_stride*iii:fast_stride*(iii+1),:,:].astype(float)  #改成0：1
                                z_start = fast_stride*iii
                                z_end = fast_stride*(iii+1)
                            else:
                                vector = data[0:1,fast_stride*iii:data.shape[1],:,:].astype(float)  #同理
                                z_start = fast_stride*iii
                                z_end = data.shape[1]                           
                            
                            vectorr = vector.transpose(1,0,2,3)
                            # B C W H
                            ### VECTORR: L', 1, W, H
                            ##below OUT OF DATE----
                            ### DATA: 2, W, H, L     e.g. (2, 512, 512, 160)
                            ### W: SS H: SSS    S0 S1 步长
                            ### IMG L', 1, W', H'
                            ## ------
                            for i in range(50): ## W
                                for ii in range(50): ## H
                                    if g+s0*i+ss < data.shape[2]-g: ## W
                                        if g+s1*ii+sss < data.shape[3]-g: ## H
                                            img_out = vectorr[:,:,g+s0*i:g+s0*i+ss,g+s1*ii:g+s1*ii+sss] #做剪裁
                                            img = torch.from_numpy(img_out).float()
                                            with torch.no_grad():
                                                input = t.autograd.Variable(img)
                                            if True: input = input.cuda()

                                            #input B C=1 W H
                                            #print(input.shape)
                                            
                                            score = model(input)
                                            score = torch.nn.Softmax(dim=1)(score).squeeze().detach().cpu().numpy()
                                            ##？SoftMax 归一化标注
                                            prob[:,z_start:z_end,g+s0*i:g+s0*i+ss,g+s1*ii:g+s1*ii+sss] = prob[:,z_start:z_end,g+s0*i:g+s0*i+ss,g+s1*ii:g+s1*ii+sss] + score.transpose(1,0,2,3)
                                            #确保通道数出现在第一个位置
                                            #flag[:,g+s0*i:g+s0*i+ss,g+s1*ii:g+s1*ii+sss,z_start:z_end] = flag[:,g+s0*i:g+s0*i+ss,g+s1*ii:g+s1*ii+sss,z_start:z_end] + 1

                    label = np.argmax((prob).astype(float),axis=0) 
                    pre = label
                    
                    print(np.shape(pre))
                    np.save(f'/tmp/code/brain_seg/p00/seg_r1/{foldername}_{checkname}', pre)
                    
                    
                    #for i in range(pre.shape[1]):
                    #    image = pre[:, i, :, :]
                    #    image = np.argmax(image, axis=0)
                    #    image = Image.fromarray(image.astype(np.uint8))
                    #    image.save(f'/tmp/code/brain_seg/p00/segmentation_result/{foldername}_{i}.png')
                    
                    
                    #print(np.sum(tru==1),np.sum(pre==1),np.max(pre))
                    ###################################
                    ###################################
                    ###################################

                    #os.mkdir('userhome/ye/2021_07/NET06/05/NET01/baseline00/00/check/dice_time_train/'+checkname+'/'+fodername+'/')
                    #np.save('userhome/ye/2021_07/NET06/05/NET01/baseline00/00/check/dice_time_train/'+checkname+'/'+fodername+'/'+'flag.npy',flag)
                    #np.save('userhome/ye/2021_07/NET06/05/NET01/baseline00/00/check/dice_time_train/'+checkname+'/'+fodername+'/'+'prob.npy',prob)
                    #np.save('userhome/ye/2021_07/NET06/05/NET01/baseline00/00/check/dice_time_train/'+checkname+'/'+fodername+'/'+'data.npy',data)
                    #print(flag.shape,prob.shape,data.shape)


                    preg = pre
                    trug = tru

                    pre = np.zeros(preg.shape)
                    tru = np.zeros(trug.shape)
                    
                    
                    

                    #？检索preg是iiii的地方 pre=iiii
                    print('Calculating Dice Coefficient...')
                    
                    for iiii in range(1,11):                    
                        pre[preg==iiii] = iiii
                        tru[trug==iiii] = iiii
                    
                        a1 = np.sum(pre==iiii)
                        a2 = np.sum(tru==iiii)
                        a3 = np.sum(np.logical_and(pre == iiii, tru == iiii))
                        #print(a1,a2,a3)
                        
                        if a1+a2 > 0:
                            WT_Dice = (2.0*a3)/(a1 + a2)
                            WT_dice.append(WT_Dice)
                            
                            label_dice_scores[iiii-1].append(WT_Dice)
                            
                            label_name = label_list[iiii-1]
                            print(f'Label NO.{iiii:2}: {label_name:<12} | Dice: {WT_Dice:.5f}')
                            
                    print('-'*40)
                            


                for idx, label in enumerate(label_list):
                    mean_dice = np.mean(label_dice_scores[idx])
                    std_dice = np.std(label_dice_scores[idx])
                    print(f'Label{idx+1:2}: {label:<12} | Mean Dice: {mean_dice:.4f} | Std Dice: {std_dice:.4f}')
                    # 存储每个标签的平均 Dice 分数
                    prediction_result = mean_dice
                    prediction_record.append((checkname, label, prediction_result))
                    
            np.save(file_record, prediction_record)
                



            
print('over!')
#while 1 > 0:
#    a = 1