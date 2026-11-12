nDataSet = 1
data_path_s = 'F:\liuwj\datasets\Open_set_domain_adaptation\datasets\Houston_7gt\Houston13.mat'
label_path_s = 'F:\liuwj\datasets\Open_set_domain_adaptation\datasets\Houston_7gt\Houston13_7gt.mat'

data_path_t = 'F:\liuwj\datasets\Open_set_domain_adaptation\datasets\Houston18\Houston18_OS.mat'
label_path_t = 'F:\liuwj\datasets\Open_set_domain_adaptation\datasets\Houston18\Houston18_OS_gt_merged.mat'
label_num = 500

N_DIMENSION = 48
BATCH_SIZE = 256

unknow_epochs = 100
pre_epochs = 200
epochs = 20  
K = 0.2  

neighbor_eps = 0.9 
num_samples = 3000

UNK_RATIO_THRESH = 0.4  
K_CLUSTERS = 2  

feat_dim = 64
LEARNING_RATE = 0.0001
CLASS_NUM = 7
SHARED_CLASS_NUM = 7
unknown_class = CLASS_NUM
HalfWidth = 4

patch_size = 2 * HalfWidth + 1

l2_decay = 5e-4

seeds = [1326, 1327, 1325, 1537, 1538, 1328, 1529, 1314, 1337, 1311]

train_end = 0
test_end = 0

source_loss_weights = {
        'source_cset_loss': 1,
        'cc_loss': 0.5,   
        'source_oset_known_loss': 1,
        'source_oset_unknown_loss': 1
    }

target_loss_weights = {
        'target_cluster_loss': 0.5,
        'target_cset_loss': 0.1,
        'target_oset_unknown_loss': 0.1,
        'cc_loss': 1,
        'neighbor_loss': 0.1
    }