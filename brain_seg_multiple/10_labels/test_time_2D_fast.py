from UNet import *
from sets import *
import torch as t
from tqdm import tqdm
import numpy
import time
import os
from config import dataset_folder, label_folder
#模型参数文件的存储路径
#检查是否有之前保存的.npy文件，如果有，则加载
file_record = f'/code/CZQ/202311/code/10_labels/{label_folder}/result/prediction_record.npy'
if not os.path.exists(file_record):
    prediction_record = []  # 创建一个新的记录列表
else:
    # 加载已有的记录
    prediction_record = np.load(file_record, allow_pickle=True)
    prediction_record = prediction_record.tolist()  # 转换为Python列表
#result_path = f'/code/CZQ/202311/code/10_labels/{label_folder}/result/{time.strftime("%m%d_%H:%M:%S")}_dice.npy'
check_ing_path = f'/code/CZQ/202311/code/10_labels/{label_folder}/check/pth/'  
if 1 > 0:
    val_dice = []                                      #用于储存计算结果
    val_std = []
    #check_ing_path = check_ing_path + 'pth/'
    check_list = sorted(os.listdir(check_ing_path),key=lambda x: os.path.getmtime(os.path.join(check_ing_path, x)))
    #最后修改时间从旧到新
    #check_list.reverse() 
    print('Check-length: ',len(check_list))
    read_list = os.listdir(check_ing_path)
    for index, checkname in enumerate(check_list):
        print(index, checkname)
        if checkname not in [item[0] for item in prediction_record]:   ##继续加载
            model = U_Net() 
            #model.eval()
            model.load_state_dict(torch.load(check_ing_path+checkname))
            model.eval()
            if opt.use_gpu: model.cuda()
            if 1 > 0:
                testpath = '/code/CZQ/202311/data/test/00/'
                folderlist = os.listdir(testpath)
                WT_dice = []
                dice_score = []
                count = 0
                label_name = label_folder.split('_')[1]
                for fodername in folderlist:
                    count += 1
                    print('-'*30+f'{testpath}{fodername}'+'-'*30)
                    #print(f'Testing... File Name: {testpath}{fodername}')
                    print(f'Count: {count} Label is {label_name}')
                    data = np.load(testpath+fodername)                    
                    #print(data.shape)
                    #vector = np.zeros((2,data.shape[1],data.shape[2],data.shape[3]))
                    #vector = data[0:2,:,:,:].astype(float)
                    tru = data[1,:,:,:]  #2-->1
                    tru[tru != int(dataset_folder)] = 0
                    tru[tru == int(dataset_folder)] = 1
                    prob = np.zeros((2,data.shape[1],data.shape[2],data.shape[3]))   #2
                    flag = np.zeros((2,data.shape[1],data.shape[2],data.shape[3]))
                    g = 0
                    s0 = 32 
                    s1 = 48
                    ss = 128
                    sss = 192
                    fast_stride = 20     
                    #####切片分段
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
                                            ##SoftMax 归一化标注
                                            prob[:,z_start:z_end,g+s0*i:g+s0*i+ss,g+s1*ii:g+s1*ii+sss] = prob[:,z_start:z_end,g+s0*i:g+s0*i+ss,g+s1*ii:g+s1*ii+sss] + score.transpose(1,0,2,3)
                    label = np.argmax((prob).astype(float),axis=0) 
                    pre = label                                                    
                    preg = pre
                    trug = tru
                    pre = np.zeros(preg.shape)
                    tru = np.zeros(trug.shape)               
                    #检索preg是iiii的地方 pre=iiii
                    #print('Calculating...')
                    #for iiii in range(1,11):                    
                    pre[preg==1] = 1
                    tru[trug==1] = 1
                    a1 = np.sum(pre==1)
                    a2 = np.sum(tru==1)
                    a3 = np.sum(np.logical_and(pre == 1, tru == 1))
                    #print(a1,a2,a3)
                    if a1+a2 > 0:
                        WT_Dice = (2.0*a3)/(a1 + a2)
                        WT_dice.append(WT_Dice)
                        print(f'Dice: {WT_Dice:.5f}')
                        print('-'*30+'*'*42+'-'*30)
                print('*'*80)    
                ### mean
                mean_dice = np.mean(WT_dice)                
                std_dice = np.std(WT_dice)
                print(f'Mean Dice: {mean_dice:.4f} | Std Dice: {std_dice:.4f}')
                prediction_result = mean_dice
                prediction_record.append((checkname, prediction_result))
                np.save(file_record, prediction_record)
np.save(file_record, prediction_record)
# 最后，可以使用prediction_record列表来获取已经预测的.pth文件和对应的结果
for checkname, result in prediction_record:
    print(f"Predicted {checkname} with result: {result}")
# 如果需要在下一次运行时继续预测，只需再次运行上述的遍历.pth文件的循环，并记录新的预测结果。
# 最后，保存最新的预测记录为.npy文件，以备下一次运行