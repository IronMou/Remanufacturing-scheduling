import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#from FJSP_Env import FJSP
from validation import validate
from Params import configs
from models.mlp import *
from PPOwithValue import PPO, Memory
from PPOwithValue import *
from mb_agg import *
from torch.utils.data import DataLoader
from uniform_instance import uni_instance_gen, FJSPDataset
from FJSP_Env import FJSP, DFJSP_GANTT_CHART
from copy import deepcopy
from agent_utils import eval_actions
from agent_utils import select_action, select_action2
from models.PPO_Actor1 import *
import os
device = torch.device(configs.device)
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR



# 记录参数历史、makespan 
parameter_history = []
makespan_history = []
'''
def record_parameters(policy_job, policy_mch):
    """
    记录策略网络最后一层的参数
    """
    last_layer_1 = policy_job.linears[-1]
    weights_1 = last_layer_1.weight.detach().cpu().numpy().flatten()
    bias_1 = last_layer_1.bias.detach().cpu().numpy().flatten()
    last_layer_2 = policy_mch.linears[-1]
    weights_2 = last_layer_2.weight.detach().cpu().numpy().flatten()
    bias_2 = last_layer_2.bias.detach().cpu().numpy().flatten()
    parameters = np.concatenate([weights_1, bias_1],[weights_2, bias_2])
    parameter_history.append(parameters)

'''

# 假设这里已经有 Job_Actor 和 Mch_Actor 类的定义 


def record_parameters(Job_Actor,Mch_Actor):
    parameters = []
    last_layer_job = Job_Actor.linears[-1]
    weights_job = last_layer_job.weight.detach().cpu().numpy().flatten()
    bias_job = last_layer_job.bias.detach().cpu().numpy().flatten()

    last_layer_mch = Mch_Actor.linears[-1]
    weights_mch = last_layer_mch.weight.detach().cpu().numpy().flatten()
    bias_mch = last_layer_mch.bias.detach().cpu().numpy().flatten()

    # 合并参数 
    parameters = np.concatenate([weights_job, bias_job, weights_mch, bias_mch]) 
    parameter_history.append(parameters) 
    return parameters


def visualize_training_process(parameter_matrix, makespan_history):
    """
    使用t-SNE对参数进行降维可视化，并结合makespan分析
    """
    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    reduced_parameters = tsne.fit_transform(parameter_matrix)
    
    # 可视化
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_parameters[:, 0], reduced_parameters[:, 1],
                          c=makespan_history, cmap='viridis', s=50, edgecolor='k')
    plt.title('t-SNE visualization of Reinforcement Learning Parameters')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(scatter, label='Makespan Value')
    plt.show()

def training_loop(epochs=100):
    """
    训练循环，包含参数记录和可视化
    """
    from uniform_instance import FJSPDataset
    #from FJSP_Env1 import FJSP

    filepath = "saved_network"
    log = []

    filepath = "saved_network"
    log = []

    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size(
            [configs.batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]
        ),
        n_nodes=configs.n_j * configs.n_m,
        device=configs.device,
    )

    ppo = PPO(
        configs.lr,
        configs.gamma,
        configs.k_epochs,
        configs.eps_clip,
        n_j=configs.n_j,
        n_m=configs.n_m,
        num_layers=configs.num_layers,
        neighbor_pooling_type=configs.neighbor_pooling_type,
        input_dim=configs.input_dim,
        hidden_dim=configs.hidden_dim,
        num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
        num_mlp_layers_actor=configs.num_mlp_layers_actor,
        hidden_dim_actor=configs.hidden_dim_actor,
        num_mlp_layers_critic=configs.num_mlp_layers_critic,
        hidden_dim_critic=configs.hidden_dim_critic,
    )
    
    # 初始化训练集和验证集
    from uniform_instance import FJSPDataset
    train_dataset = FJSPDataset(
        configs.n_j, configs.n_m, configs.low, configs.high, configs.num_ins, 200
    )
    validat_dataset = FJSPDataset(
        configs.n_j, configs.n_m, configs.low, configs.high, 1280, 200
    )

    data_loader = DataLoader(train_dataset, batch_size=configs.batch_size)
    valid_loader = DataLoader(validat_dataset, batch_size=configs.batch_size)
    
    for epoch in range(epochs):
        memory = Memory()
        ppo.policy_old_job.train()
        ppo.policy_old_mch.train()

        times, losses, rewards2, critic_rewards = [], [], [], []
        start = time.time()

        costs = []
        losses, rewards, critic_loss = [], [], []
        for batch_idx, batch in enumerate(data_loader):
            env = FJSP(configs.n_j, configs.n_m)
            data = batch.numpy()

            adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(
                data
            )

            job_log_prob = []
            mch_log_prob = []
            r_mb = []
            done_mb = []
            first_task = []
            pretask = []
            j = 0
            pool = None
            mch_a = None
            last_hh = None
            ep_rewards = -env.initQuality
            env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(configs.device)
            env_dur = torch.from_numpy(np.copy(dur)).float().to(configs.device)
            while True:

                env_adj = aggr_obs(
                    deepcopy(adj).to(configs.device).to_sparse(), configs.n_j * configs.n_m
                )
                env_fea = torch.from_numpy(np.copy(fea)).float().to(configs.device)
                env_fea = deepcopy(env_fea).reshape(-1, env_fea.size(-1))
                env_candidate = torch.from_numpy(np.copy(candidate)).long().to(configs.device)

                env_mask = torch.from_numpy(np.copy(mask)).to(configs.device)
                env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(configs.device)
                # env_job_time = torch.from_numpy(np.copy(job_time)).float().to(configs.device)

                

                action, a_idx, log_a, action_node, _, mask_mch_action, hx = ppo.policy_old_job(x=env_fea,
                                                                                               graph_pool=g_pool_step,
                                                                                               padded_nei=None,
                                                                                               adj=env_adj,
                                                                                               candidate=env_candidate
                                                                                               , mask=env_mask

                                                                                               , mask_mch=env_mask_mch
                                                                                               , dur=env_dur
                                                                                               , a_index=0
                                                                                               , old_action=0
                                                                                               , mch_pool=pool
                                                                                               )






                pi_mch, pool = ppo.policy_old_mch(action_node, hx, mask_mch_action, env_mch_time,mch_a,last_hh)
                

                #pi_mch, _, last_hh = ppo.policy_old_mch(
                    #action_node, hx, mask_mch_action, env_mch_time, mch_a, last_hh
                #)
                # print(action[0].item(),mch_a[0].item())
                mch_a, log_mch = select_action2(pi_mch)
                job_log_prob.append(log_a)

                # print(action[0].item(),mch_a[0].item())
                mch_log_prob.append(log_mch)

                if j == 0:
                    first_task = action.type(torch.long).to(configs.device)

                pretask = action.type(torch.long).to(configs.device)

                memory.mch.append(mch_a)
                memory.pre_task.append(pretask)
                memory.adj_mb.append(env_adj)
                memory.fea_mb.append(env_fea)
                memory.candidate_mb.append(env_candidate)
                memory.action.append(deepcopy(action))
                memory.mask_mb.append(env_mask)
                memory.mch_time.append(env_mch_time)
                memory.a_mb.append(a_idx)

                adj, fea, reward, done, candidate, mask, job, _, mch_time, job_time = (
                    env.step(action.cpu().numpy(), mch_a)
                )
                ep_rewards += reward

                r_mb.append(deepcopy(reward))
                done_mb.append(deepcopy(done))

                j += 1
                if env.done():
                    break
            memory.dur.append(env_dur)
            memory.mask_mch.append(env_mask_mch)
            memory.first_task.append(first_task)
            memory.job_logprobs.append(job_log_prob)
            memory.mch_logprobs.append(mch_log_prob)
            memory.r_mb.append(torch.tensor(r_mb).float().permute(1, 0))
            memory.done_mb.append(torch.tensor(done_mb).float().permute(1, 0))
            # -------------------------------------------------------------------------------------
            ep_rewards -= env.posRewards
            # -------------------------------------------------------------------------------------
            loss, v_loss = ppo.update(memory, batch_idx)
            memory.clear_memory()
            mean_reward = np.mean(ep_rewards)
            log.append([batch_idx, mean_reward])
            if batch_idx % 100 == 0:
                file_writing_obj = open(
                    "./"
                    + "log_"
                    + str(configs.n_j)
                    + "_"
                    + str(configs.n_m)
                    + "_"
                    + str(configs.low)
                    + "_"
                    + str(configs.high)
                    + ".txt",
                    "w",
                )
                file_writing_obj.write(str(log))

            rewards.append(np.mean(ep_rewards).item())
            losses.append(loss)
            critic_loss.append(v_loss)

            cost = env.mchsEndTimes.max(-1).max(-1)
            costs.append(cost.mean())
            step = 10
            if (batch_idx + 1) % step == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-step:])
                mean_reward = np.mean(costs[-step:])
                critic_losss = np.mean(critic_loss[-step:])

                print(
                    "  Batch %d/%d, reward: %2.3f, loss: %2.4f,critic_loss:%2.4f,took: %2.4fs"
                    % (
                        batch_idx,
                        len(data_loader),
                        mean_reward,
                        mean_loss,
                        critic_losss,
                        times[-1],
                    )
                )
                record = 1000000

                t4 = time.time()

                validation_log = validate(
                    valid_loader,
                    configs.batch_size,
                    ppo.policy_old_job,
                    ppo.policy_old_mch,
                ).mean()

                epoch_dir = os.path.join(filepath, "%s" % batch_idx)
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)
                job_savePath = os.path.join(
                    epoch_dir,
                    "{}.pth".format(
                        "policy_job"
                        + str(configs.n_j)
                        + "_"
                        + str(configs.n_m)
                        + "_"
                        + str(configs.low)
                        + "_"
                        + str(configs.high)
                    ),
                )
                machine_savePate = os.path.join(
                    epoch_dir,
                    "{}.pth".format(
                        "policy_mch"
                        + str(configs.n_j)
                        + "_"
                        + str(configs.n_m)
                        + "_"
                        + str(configs.low)
                        + "_"
                        + str(configs.high)
                    ),
                )
                "if np.array(validation_log).mean() < record:"
                torch.save(ppo.policy_job.state_dict(), job_savePath)
                torch.save(ppo.policy_mch.state_dict(), machine_savePate)

                record = validation_log
                print("The validation quality is:", validation_log)
                file_writing_obj1 = open(
                    "./"
                    + "vali_"
                    + str(configs.n_j)
                    + "_"
                    + str(configs.n_m)
                    + "_"
                    + str(configs.low)
                    + "_"
                    + str(configs.high)
                    + ".txt",
                    "w",
                )
                file_writing_obj1.write(str(validation_log))
                t5 = time.time()

                print("Training:", t4 - t3)
                print("Validation:", t5 - t4)

            # 记录参数和makespan
            record_parameters(ppo.policy_job,ppo.policy_mch)
            makespan_history.append(validation_log)

    # 将参数历史记录转换为矩阵形式 (T, D)
    parameter_matrix = np.array(parameter_history)
    
    return parameter_matrix, makespan_history

if __name__ == "__main__":
    total1 = time.time()
    parameter_matrix, makespan_history = training_loop(epochs=100)
    total2 = time.time()
    print(total2 - total1)
    
    # 可视化训练过程
    visualize_training_process(parameter_matrix, makespan_history)