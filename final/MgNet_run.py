import os
import shlex
import subprocess
import threading
import signal
from itertools import product

f = open("train_data", 'w')
f.close()

def run_command(env):
    print('=> run commands on GPU:{}', env['CUDA_VISIBLE_DEVICES'])

    while True:
        try:
            comm = command_list.pop(0)
        except IndexError:
            break

        proc = subprocess.Popen(comm, env=env)
        proc.wait()


test_param_groups = {}
test_param_groups["optim_group"] = ['ssm','sasa+','sgd']          
test_param_groups["dataset_group"] = ['cifar10']     
test_param_groups["channel_group"] = [256] 
test_param_groups["minibatchsize_group"] = [128]
test_param_groups["epochs_group"] = [150]
test_param_groups["lr_group"] = [0.1]
test_param_groups["momentum_group"] = [0.9]
test_param_groups["dampening_group"] = [0.9]
test_param_groups["trun_group"] = [0.02, 0.03]
test_param_groups["sig_group"] = [0.05]
test_param_groups["lk_group"] = [8,10,12]
test_param_groups["minstat_group"] = [75, 85, 100]
test_param_groups["sf_group"] = [10, 15]
test_param_groups["vm_group"] = ['bm'] 
test_param_groups["km_group"] = ['loss_plus_smooth', 'loss'] 
test_param_groups["drop_group"] = [10] 

# flexible

standard_command = 'python MgNet_train.py --cuda --optim={} --dataset={} --channel={} --minibatch-size={} --epochs={} --lr={} --momentum={} --dampening={} '

#SSM test command list
ssm_ordered_param_group = ['dataset_group', 'channel_group', 'minibatchsize_group', 'epochs_group', 'lr_group', 'momentum_group', 'dampening_group', 
                           'trun_group', 'sig_group', 'lk_group', 'minstat_group', 'sf_group', 'vm_group', 'km_group', 'drop_group']
ssm_value_list = [['ssm']] + [test_param_groups[group] for group in ssm_ordered_param_group]
ssm_test_list = list(product(*ssm_value_list))
ssm_command = standard_command + '--trun={} --sig={} --lk={} --minstat={} --sf={} --vm={} --km={} --drop={}'

    
#SASA test command list
sasaplus_ordered_param_group = ['dataset_group', 'channel_group', 'minibatchsize_group', 'epochs_group', 'lr_group', 'momentum_group', 'dampening_group', 
                                'sig_group', 'lk_group', 'minstat_group', 'vm_group', 'km_group', 'drop_group']
sasaplus_value_list = [['sasa+']] + [test_param_groups[group] for group in sasaplus_ordered_param_group]
sasaplus_test_list = list(product(*sasaplus_value_list))
sasaplus_command = standard_command + '--sig={} --lk={} --minstat={} --vm={} --km={} --drop={}'

#SGD test command list
sgd_ordered_param_group = ['dataset_group', 'channel_group', 'minibatchsize_group', 'epochs_group', 'lr_group', 'momentum_group', 'dampening_group']
sgd_value_list = [['sgd']] + [test_param_groups[group] for group in sgd_ordered_param_group]
sgd_test_list = list(product(*sgd_value_list))
sgd_command = standard_command



test_list = ssm_test_list + sasaplus_test_list + sgd_test_list
# flexible
command_list = []
for file in test_list:
    if file[0] == 'ssm':
        command_list += [ssm_command.format(*file)]
    elif file[0] == 'sasa+':
        command_list += [sasaplus_command.format(*file)]
    elif file[0] == 'sgd': 
        command_list += [sgd_command.format(*file)] 
print(command_list)
command_list = [shlex.split(comm) for comm in command_list]
print("the total combinations of hyperparamater is", len(command_list))

# List all the GPUs you have
ids_cuda = [3,4,5]
for c in ids_cuda:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(c)
    thread = threading.Thread(target = run_command, args = (env, ))
    thread.start()
