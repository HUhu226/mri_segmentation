#coding:utf8
from UNet import *
from sets import *
from dataset_2d1 import Brats17
from torch.utils.data import DataLoader
import torch as t
from tqdm import tqdm
import numpy
import time

############################################################################
def val(model,dataloader):         
    '''
    计算模型在验证集上的准确率等信息
    '''
    model.eval()                                 #eval模式
    #val_meter=AverageMeter()
    val_losses, dcs = [], []
    #criterion = t.nn.CrossEntropyLoss()
    for ii, data in enumerate(dataloader):      
        input, label = data                      #data拆解为输入数据和标签
        val_input = Variable(input.cuda())       #input包含当前批次输入数据，封装到Variable对象中？用于自动梯度计算？
        val_label = Variable(label.cuda())       #接着调用.cuda()将val_input移动到GPU
        if opt.use_gpu:
            val_input = val_input.cuda()         #如果检查是否在GPU上计算为真 移动到GPU
            val_label = val_label.cuda()         #？？冗余的操作
            model = model.cuda()
        outputs=model(val_input)                 #使用模型model执行向前传播
        pred = outputs.data.max(1)[1].cpu().numpy().squeeze()    #对模型的output进行处理，获取每一个像素点的预测类别
        #.data - 获取张量中的数据部分
        #.max(1) - 对tensor的操作，找到每个像素点在不同类别上的最大值  .max(1)的结果中索引[1]
        #.cpu() - 移回CPU .numpy() - 转换为numpy数组 
        #.squeeze()删除张量中维度大小为1的维度
        gt = val_label.data.cpu().numpy().squeeze()              #标签数据中获取类别信息

        #print(pred.shape)
        #print(gt.shape)
        for i in range(gt.shape[0]):                             #遍历验证集中的每个样本
            #print(i)
            dc,val_loss=calc_dice(gt[i,:,:,:],pred[i,:,:,:])     #每次循环中，使用calc_dice计算Dice系数
            dcs.append(dc)                                       #dc添加到dcs列表
            val_losses.append(val_loss)
        #for gt_, pred_ in zip(gt, pred):
            #gts.append(gt_)
            #preds.append(pred_)
    #score,cc,acc=scores(gts,preds,n_class=classes)
    model.train()
    return np.mean(dcs),np.mean(val_losses)
############################################################################




############################################################################
print('train:')
lr = 0.0001 #opt.lr 
batch_size = 10
print('batch_size:',batch_size,'lr:',lr)

plt_list = []     #空列表，存储loss

model = U_Net()   #初始化一个U-Net模型实例，来源于UNet.py
model.load_state_dict(t.load('/tmp/code/brain_seg/p00/check/pth/00/0_4444_0_0.0001_10_0426_20:43:40.pth'))

if opt.use_gpu: 
    model.cuda()
train_data=Brats17(opt.train_data_root,train=True)                        #
val_data=Brats17(opt.train_data_root,train=False,val=True)                #创建一个Brats17数据集的实例，用于验证数据的加载
val_dataloader = DataLoader(val_data,4,shuffle=False,num_workers=opt.num_workers)
#数据加载器加载验证数据集，val_data包含验证数据的数据集对象，4-batchsize，不洗牌=不改变数据顺序，指定用与加载数据的工作线程数量


# 定义类别权重，将背景类别的权重设置为0
class_weights = [1, 1, 1, 2, 1, 2, 1, 2, 2, 3, 5]  # 类别权重，label=9 和 label=10 的权重较高

# 转换为 torch.Tensor 对象
class_weights_tensor = t.FloatTensor(class_weights)

# 将权重移到 GPU（如果使用 GPU 的话）
if opt.use_gpu:
    class_weights_tensor = class_weights_tensor.cuda()

# 定义交叉熵损失函数并传入权重
criterion = t.nn.CrossEntropyLoss(weight=class_weights_tensor)

#可以选择不同类型的损失函数 criterion 计算loss function
#criterion = get_loss_criterion('DiceLoss')
#weight = torch.FloatTensor([1, 2, 2, 2, 2])#[1, 10, 1, 5, 1]
#criterion = t.nn.CrossEntropyLoss()#weight=weight
#criterion = get_loss_criterion(config)
#criterion = FocalLoss2d()
#criterion = DiceLoss()
#criterion = FocalLoss(5)
#criterion = DiceLoss2D()
if opt.use_gpu:                                    #将损失函数转移到GPU上计算
    criterion = criterion.cuda()

loss_meter=AverageMeter()                          #跟踪记录训练损失
previous_loss = 1e+20

train_dataloader = DataLoader(train_data,batch_size = batch_size,shuffle=True,num_workers=opt.num_workers)
optimizer = t.optim.Adam(model.parameters(),lr = lr,weight_decay = opt.weight_decay)
#Adam 优化器

# train
for epoch in range(opt.max_epoch):
        
    loss_meter.reset()                          #用于重置损失值的平均计量工具
    #confusion_matrix.reset()
    
    for ii,(data,label) in tqdm(enumerate(train_dataloader),total=len(train_data), desc=f'Epoch{epoch}'):    #嵌套循环
        #train_dataloader是之前创建的数据加载器，将数据分成batch
        #data输入数据，label对应标签，每次enumerate都load一批数据，enumerate返回一个计数器ii和对应的data label
        #tqdm是一个进度条库，显示进度，total=len(train_data)指定了总迭代次数，这里为整个train_data
        
        #print(data.shape,label.shape)
        # train model 
        input = Variable(data)                  #这里将输入数据和目标标签封装成PyTorch的variable对象，用于自动求导和损失计算
        target = Variable(label)                
            
        if opt.use_gpu:
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()                   #清零优化器的梯度缓冲区
        # model = model.cuda()
        score = model(input)                    #使用模型向前传播，得到预测值，通常为score、logits
        #print('aa')
        #print(score.shape)
        #print(target.shape)
        loss = criterion(score,target)          #计算损失值，使用criterion计算score和target差异

        loss.backward()                         #反向传播损失
        optimizer.step()

        # meters update and visualize
        #loss_meter.add(loss.data[0])
        loss_meter.update(loss.item())

        if ii%5==1:
            plt_list.append(loss_meter.val)
        if ii%50==1:
            print('train-loss-avg:', loss_meter.avg,'train-loss-each:', loss_meter.val)
            
    #if epoch%2==1:
    #if 1 > 0:
        if ii%200==1:
        #if 1 > 0:
            #acc,val_loss = val(model,val_dataloader)
            acc = 0
            val_loss = 0
            
            ##prefix定义保存模型参数文件和损失值文件的路径前缀，str转换为字符串，
            
            #if acc > pre_acc:
            #prefix = 'checkpoints/' + str(loss_meter.avg)+'_'+str(loss_meter.val) + '_'
            prefix = '/tmp/code/brain_seg/p00/check/pth/weight/' + str(acc)+'_4444_'+str(val_loss) + '_'+str(lr)+'_'+str(batch_size)+'_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')     #将当前的日期和时间以字符串形式作为文件名的一部分
            t.save(model.state_dict(), name)                       #保存当前模型参数到文件name中
            
            ##定义损失值变化曲线文件的路径和名称，将损失值的变化曲线保存在name1中以便可视化
            name1 = time.strftime('/tmp/code/brain_seg/p00/check/plt/weight/' + '%m%d_%H:%M:%S.npy')   
            numpy.save(name1, plt_list)
            
        #if ii%100=7
            #numpy.save(time.strftime('checkpoints/plt/' + str(loss_meter) + '__%m%d_%H:%M:%S.npy'), loss_meter)
            #model.save(str(acc)+'_'+str(val_loss))
            #pre_acc=acc
            #vis.plot('dice-coefficent',acc)
            #vis.plot('val_loss',val_loss)

    # update learning rate
    #if loss_meter.avg > previous_loss:          
    #   lr = lr * opt.lr_decay
            # \u7b2c\u4e8c\u79cd\u964d\u4f4e\u5b66\u4e60\u7387\u7684\u65b9\u6cd5:\u4e0d\u4f1a\u6709moment\u7b49\u4fe1\u606f\u7684\u4e22\u5931
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = lr        
    print('old:','batch_size:',batch_size,'lr:',lr)
    #batch_size = numpy.load('/userhome/GUOXUTAO/2019_00/NET00/checkpoints/config/batch_size.npy')
    #lr = numpy.load('/userhome/GUOXUTAO/2019_00/NET00/checkpoints/config/lr.npy')
    print('new:','batch_size:',batch_size,'lr:',lr)
    #train_dataloader = DataLoader(train_data,batch_size,shuffle=True,num_workers=opt.num_workers)
    #optimizer = t.optim.Adam(model.parameters(),lr = lr,weight_decay = opt.weight_decay)
    #previous_loss = loss_meter.avg
############################################################################




