#dataset_folder是序列号
#label_folder是真正的文件夹名称
#选择要训练&测试的Label序号
dataset_folder = '10'
if dataset_folder == '1':
    label_folder = 'L01_WM'
elif dataset_folder == '2':
    label_folder = 'L02_GM'
elif dataset_folder == '3':
    label_folder = 'L03_CSF'
elif dataset_folder == '5':
    label_folder = 'L05_Scalp'
elif dataset_folder == '6':
    label_folder = 'L06_EyeBalls'
elif dataset_folder == '7':
    label_folder = 'L07_CompactBone'
elif dataset_folder == '8':
    label_folder = 'L08_SpongyBone'
elif dataset_folder == '9':
    label_folder = 'L09_Blood'
elif dataset_folder == '10':
    label_folder = 'L10_Muscle'
else:
    label_folder = None