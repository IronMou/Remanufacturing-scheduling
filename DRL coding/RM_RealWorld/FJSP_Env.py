import os
import io
import random
import numpy as np
import gym
from gym.utils import EzPickle
from uniform_instance import override
from updateEndTimeLB import calEndTimeLB, calEndTimeLBm
from Params import configs
from permissibleLS import permissibleLeftShift
from updateAdjMat import getActionNbghs
from copy import deepcopy
from min_job_machine_time import min_job_mch
from dispatichRule import DRs
import torch
import time
from PIL import Image
import matplotlib.pyplot as plt

import logging

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__file__)


class FJSP(gym.Env, EzPickle):
    def __init__(self, n_j, n_m, EachJob_num_operation, product_num_jobs=None):
        EzPickle.__init__(self)

        self.step_count = 0
        self.computation_time = 0

        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.product_num_jobs = product_num_jobs
        self.num_operation = EachJob_num_operation
        self.number_of_tasks = EachJob_num_operation.sum(axis=1)[0]
        # self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        # the task id for first column
        self.max_operation = EachJob_num_operation.max()

        self.last_col = np.cumsum(EachJob_num_operation, -1) - 1
        self.first_col = np.cumsum(EachJob_num_operation, -1) - EachJob_num_operation
        self.product_arr, self.product_job_indices = self.set_product_arr()

        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs

    def done(self):
        if np.all(self.partial_sol_sequeence[0] >= 0):
            return True
        return False

    def set_product_arr(self):
        indices = 0
        product_job_indices = [[], []]
        product_arr = np.array([], dtype=np.int32)
        for pid, num in enumerate(self.product_num_jobs):
            # record product start and stop: 0 and 1 array
            product_job_indices[0].append(indices)
            indices += int(num)
            product_job_indices[1].append(indices - 1)
            # setup product_arr for indexing product id
            tmp = np.array([int(pid)] * int(num), dtype=np.int32)
            product_arr = np.concatenate(
                (product_arr, tmp),
                axis=0,
                dtype=np.int32,
            )

        return product_arr, product_job_indices

    def set_adj(self):
        assert len(self.product_num_jobs) > 0, "missing jobs number of product"

        self.adj = []
        # initialize adj matrix
        for i in range(self.batch_sie):
            self_as_nei = np.eye(self.number_of_tasks, dtype=np.single) + np.eye(
                self.number_of_tasks, k=-1, dtype=np.single
            )
            job_idx = np.int32(0)
            # N个product
            for n in self.product_num_jobs:
                product_start = self.first_col[i][job_idx]
                job_idx += np.int32(n)
                product_end = self.first_col[i][job_idx - 1]
                for j in range(product_start, product_end + 1):
                    if j in self.first_col[i][1:]:
                        # if j > 0:
                        self_as_nei[j][j - 1] = 0
                        self_as_nei[j][product_start] = 1 if j != product_end else 0
                        self_as_nei[product_end][j - 1] = 1 if j != product_start else 0
            self.adj.append(self_as_nei)
            np.savetxt(
                f"data/adj{i}.csv", np.array(self_as_nei, dtype=np.int8), fmt="%d"
            )
        self.adj = torch.tensor(np.array(self.adj))

    def get_priority_from_adj(self):
        # adj 表示邻接矩阵
        # 计算每个任务的入度: 每列的和表示入度
        return self.adj.sum(axis=-1)

    def update_mask(self):
        """
        - 根据当前邻接矩阵adj, 获取priority
        - 根据priority更新mask
        """
        priority = self.get_priority_from_adj()
        mask = np.full(
            shape=(self.batch_sie, self.number_of_jobs), fill_value=1, dtype=bool
        )
        batch_size = mask.shape[0]
        for bat in range(batch_size):
            # 找到所有优先级最大的任务索引
            # 根据入度生成优先级，入度越小优先级越高
            max_priority_jobs = np.where(priority[bat] == max(priority.min(), 1))[0]
            mask_job_indices = np.where(
                np.isin(self.first_col[bat], max_priority_jobs)
            )[0]
            # update mask
            mask[bat, mask_job_indices] = False

        return mask | self.mask

    def update_latest_action_and_time(self, action, startTime_a, i, row, col):
        self.temp1[i, row, col] = startTime_a + self.dur_a  # 完工时间
        # update the latest action time and action
        if (
            self.temp1[i, row, col]
            > self.product_latest_action_time[i, self.product_arr[row]]
        ):
            self.product_latest_action_time[i, self.product_arr[row]] = self.temp1[
                i, row, col
            ]
            self.product_latest_action[i, self.product_arr[row]] = action[i]

    def get_product_id(self, bat, row):
        if row in self.product_job_indices[0]:
            # product start pid depend on nothing
            pid = None
        elif row in self.product_job_indices[1]:
            # product stop pid(action) depend on all product jobs above
            pid = self.product_latest_action[bat, self.product_arr[row]]
        else:
            # product jobs pid(action) depend on product start
            pid = self.first_col[
                bat, self.product_job_indices[0][self.product_arr[row]]
            ]
        return pid

    @override
    def step(self, action, mch_a, gantt_plt=None):
        # action is a int 0 - 224 for 15x15 for example
        feas, rewards, dones, masks, mch_masks = [], [], [], [], []
        mch_spaces, mchForJobSpaces = [], []
        for i in range(self.batch_sie):
            # redundant action makes no effect 多余的动作无效
            if action[i] not in self.partial_sol_sequeence[i]:
                # UPDATE BASIC INFO:
                # row is for job indices
                # col is for operation indices
                # pid is for product action indices
                row = np.where(action[i] <= self.last_col[i])[0][0]
                col = action[i] - self.first_col[i][row]
                pid = self.get_product_id(i, row)

                self.dispatched_num_opera[i][row] += 1
                if i == 0:
                    self.step_count += 1
                self.finished_mark[i, action[i]] = 1
                self.dur_a = self.dur[i, row, col, mch_a[i]]

                # action time
                self.partial_sol_sequeence[i][
                    np.where(self.partial_sol_sequeence[i] < 0)[0][0]
                ] = action[i]
                self.mchMat[i][row][col] = mch_a[i]

                # UPDATE STATE:
                # permissible left shift 允许向左移动
                startTime_a, flag = permissibleLeftShift(
                    a=action[i],
                    mch_a=mch_a[i],
                    durMat=self.dur_cp[i],
                    mchMat=self.mchMat[i],
                    mchsStartTimes=self.mchsStartTimes[i],
                    opIDsOnMchs=self.opIDsOnMchs[i],
                    mchEndTime=self.mchsEndTimes[i],
                    pid=pid,
                    row=row,
                    col=col,
                    first_col=self.first_col[i],
                    last_col=self.last_col[i],
                )

                self.flags.append(flag)
                if gantt_plt != None:
                    gantt_plt.gantt_plt(
                        row,
                        col,
                        mch_a.cpu().numpy(),
                        startTime_a,
                        self.dur_a,
                        self.product_job_indices,
                    )

                # update omega or mask
                if action[i] not in self.last_col[i]:
                    self.omega[i, row] += 1
                    self.job_col[i, row] += 1
                else:
                    self.mask[i, row] = 1

                # update job finished time
                self.update_latest_action_and_time(action, startTime_a, i, row, col)

                # self.LBs[i] = calEndTimeLB(self.temp1[i], self.input_min[i],self.input_mean[i])
                self.LB[i] = calEndTimeLBm(self.temp1[i], self.input_min[i])
                LBm = []
                for y in range(self.batch_sie):
                    LB = []
                    for j in range(self.number_of_jobs):
                        for k in range(self.num_operation[y][j]):
                            LB.append(self.LB[y, j, k])
                    LBm.append(LB)
                self.LBm = np.array(LBm)

                # self.LBs为所有task最快的完工时间
                # adj matrix
                precd, succd = self.getNghbs(action[i], self.opIDsOnMchs[i])

                self.adj[i, :, action[i]] = 0
                self.adj[i, action[i], action[i]] = 1
                if action[i] not in self.first_col[i]:
                    self.adj[i, action[i], action[i] - 1] = 1
                # self.adj[i, action[i], precd] = 1
                # self.adj[i, succd, action[i]] = 1
                """np.savetxt(
                    f"data/step_adj{action[i]}.csv",
                    np.array(self.adj[i], dtype=np.int8),
                    fmt="%d",
                )"""

                """if action[i] not in self.first_col[i]:
                    self.adj[i,action[i]-1, action[i]] = 0
                self.adj[i, precd, action[i]] = 0
                self.adj[i, action[i], succd] = 0"""

                done = self.done()

                # min_job_mch(mch_time, mchsEndTimes, number_of_machines, dur, temp, first_col)
                if self.rule != None:
                    mask1, mch_mask = DRs(
                        self.mch_time[i],
                        self.job_time[i],
                        self.mchsEndTimes[i],
                        self.number_of_machines,
                        self.dur_cp[i],
                        self.temp1[i],
                        self.omega[i],
                        self.mask[i],
                        done,
                        self.mask_mch[i],
                        self.num_operation[i],
                        self.dispatched_num_opera[i],
                        self.input_min[i],
                        self.job_col[i],
                        self.input_max[i],
                        self.rule,
                        self.last_col[i],
                        self.first_col[i],
                    )
                else:
                    mch_space, mchForJobSpace, mask1, mch_mask = min_job_mch(
                        self.mch_time[i],
                        self.job_time[i],
                        self.mchsEndTimes[i],
                        self.number_of_machines,
                        self.dur_cp[i],
                        self.temp1[i],
                        self.omega[i],
                        self.mask[i],
                        done,
                        self.mask_mch[i],
                        self.first_col[i],
                        self.last_col[i],
                    )

                masks.append(mask1)
                mch_masks.append(mch_mask)
                # print('action_space',mchForJobSpaces,'mchspace',mch_space)
            # prepare for return
            # -------------------------------------------------------------------------------------
            """fea = np.concatenate((self.LBs[i].reshape(-1, 2)/configs.et_normalize_coef,
                                  self.finished_mark[i].reshape(-1, 1)), axis=-1)"""
            # ----------------------------------------------------------------------------------------
            """fea = np.concatenate((self.dur[i].reshape( -1, self.number_of_machines)/configs.et_normalize_coef,
                                  self.finished_mark[i].reshape( -1, 1)), axis=-1)"""
            # --------------------------------------------------------------------------------------------------------------------
            """fea = self.LBm[i].reshape(-1, 1) / configs.et_normalize_coef"""
            fea = np.concatenate(
                (
                    self.LBm[i].reshape(-1, 1) / configs.et_normalize_coef,
                    # np.expand_dims(self.job_time[i], 1).repeat(self.number_of_machines, axis=1).reshape(
                    # self.number_of_tasks, 1)/configs.et_normalize_coef,
                    self.finished_mark[i].reshape(self.number_of_tasks, 1),
                ),
                axis=-1,
            )
            feas.append(fea)
            """reward = self.mchsEndTimes[i][mch_a[i]].max()-self.up_mchendtime[i][mch_a[i]].max()-self.dur_a


            if reward < 0.00001:
                reward = 0
            self.up_mchendtime = np.copy(self.mchsEndTimes)
            for b,c in zip(self.up_mchendtime[i],range(self.number_of_machines)):
                self.up_mchendtime[i][c] = [0 if i < 0 else i for i in b]
            rewards.append(reward)"""
            reward = -(self.LBm[i].max() - self.max_endTime[i])
            if reward == 0:
                reward = configs.rewardscale
                self.posRewards[i] += reward
            rewards.append(reward)
            self.max_endTime[i] = self.LBm[i].max()

            dones.append(done)

        mch_masks = np.array(mch_masks)
        if self.rule != None:
            self.mask_mch = mch_masks

        # logger.info(f"job{action} time: {t2 - t1}")

        return (
            self.adj,
            np.array(feas),
            rewards,
            dones,
            self.omega,
            masks,
            mchForJobSpaces,
            self.mask_mch,
            self.mch_time,
            self.job_time,
        )

    @override
    def reset(self, data, rule=None):
        # data (batch_size,n_job,n_mch,n_mch)
        self.rule = rule
        self.batch_sie = data.shape[0]
        self.oper_max = data.shape[2]

        """for i in range(self.batch_sie):

            first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
            self.first_col.append(first_col)
        # the task id for last column
            last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
            self.last_col.append(last_col)
        self.first_col = np.array(self.first_col)"""

        self.product_latest_action = np.zeros(
            shape=(self.batch_sie, len(self.product_num_jobs)), dtype=np.int32
        )
        self.product_latest_action_time = np.zeros(
            shape=(self.batch_sie, len(self.product_num_jobs)), dtype=np.int32
        )
        self.job_col = np.zeros(
            shape=(self.batch_sie, self.number_of_jobs), dtype=np.int32
        )

        self.last_col = np.array(self.last_col)
        self.step_count = 0

        # self.num_operation = np.full(shape=(self.number_of_jobs), fill_value=self.number_of_machines)
        self.dispatched_num_opera = np.zeros(
            shape=(self.batch_sie, self.number_of_jobs)
        ).astype(int)

        self.mchMat = -1 * np.ones(
            (self.batch_sie, self.number_of_jobs, self.max_operation), dtype=np.int64
        )

        # single单精度浮点数
        self.dur = data.astype(np.single)
        self.dur_cp = deepcopy(self.dur)

        # record action history
        self.partial_sol_sequeence = -1 * np.ones(
            (self.batch_sie, self.number_of_tasks), dtype=np.int64
        )

        self.flags = []
        self.posRewards = np.zeros(self.batch_sie)
        self.adj = []

        # initialize adj matrix
        self.set_adj()
        # initialize product array
        # self.set_product_arr()

        """
        for i in range(self.batch_sie):
            conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
            conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
            # first column does not have upper stream conj_nei
            conj_nei_up_stream[self.first_col] = 0
            # last column does not have lower stream conj_nei
            conj_nei_low_stream[self.last_col] = 0
            self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
            adj = self_as_nei + conj_nei_up_stream
            self.adj.append(adj)
            np.savetxt(f"data/adj{i}.csv", np.array(adj))
        self.adj = torch.tensor(np.array(self.adj))
        """

        """for i in range(self.batch_sie):
            dat = torch.from_numpy(data[i].reshape(-1, self.number_of_machines)).permute(1, 0)
            adj = np.eye(self.number_of_tasks)
            conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
            conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
            # first column does not have upper stream conj_nei
            conj_nei_up_stream[self.first_col] = 0
            # last column does not have lower stream conj_nei
            conj_nei_low_stream[self.last_col] = 0
            one = np.where(dat > 0)
            for i in range(self.number_of_machines):
                index = np.where(one[0] == i)[0]
                for j in one[1][index]:
                    for z in one[1][index]:
                        adj[j][z] = 1
            for i in range(1,self.number_of_tasks):
                adj[i][i - 1] = conj_nei_up_stream[i][i - 1]
                adj[i - 1][i] = 0
            self.adj.append(adj)
        self.adj = torch.tensor(self.adj).type(torch.float)"""

        # initialize features
        self.mask_mch = np.full(
            shape=(
                self.batch_sie,
                self.number_of_jobs,
                self.max_operation,
                self.number_of_machines,
            ),
            fill_value=0,
            dtype=bool,
        )
        input_min = []
        input_mean = []
        input_max = []
        start = time.time()
        for t in range(self.batch_sie):
            min = []
            mean = []
            max = []
            for i in range(self.number_of_jobs):
                dur_min = []
                dur_mean = []
                dur_max = []
                for j in range(self.max_operation):
                    durmch = self.dur[t][i][j][np.where(self.dur[t][i][j] > 0)]
                    mean_value = durmch.mean() if durmch.size > 0 else 0
                    self.mask_mch[t][i][j] = [
                        1 if i <= 0 else 0 for i in self.dur_cp[t][i][j]
                    ]
                    self.dur[t][i][j] = [
                        mean_value if i <= 0 else i for i in self.dur[t][i][j]
                    ]
                    if len(durmch) == 0:
                        dur_min.append(1)
                        dur_mean.append(1)
                        dur_max.append(1)
                    else:
                        dur_min.append(durmch.min().tolist())
                        dur_mean.append(durmch.mean().tolist())
                        dur_max.append(durmch.max().tolist())
                min.append(dur_min)
                mean.append(dur_mean)
                max.append(dur_max)
            input_min.append(min)
            input_mean.append(mean)
            input_max.append(max)

        end = time.time() - start

        self.input_min = np.array(input_min)
        self.input_max = np.array(input_max)

        self.input_mean = np.array(input_mean)

        self.input_2d = np.concatenate(
            [
                self.input_min.reshape(
                    (self.batch_sie, self.number_of_jobs, self.max_operation, 1)
                ),
                self.input_mean.reshape(
                    (self.batch_sie, self.number_of_jobs, self.max_operation, 1)
                ),
            ],
            -1,
        )

        self.LBs = np.cumsum(self.input_2d, -2)
        self.LB = np.cumsum(self.input_min, -1)

        if rule != None:
            LBm = []
            for i in range(self.batch_sie):
                LB = []
                for j in range(self.number_of_jobs):
                    for k in range(self.num_operation[i][j]):
                        LB.append(self.LB[i, j, k])
                LBm.append(LB)
            self.LBm = np.array(LBm)
        else:
            LBm = []
            mch_mask = []
            durs = []
            for y in range(self.batch_sie):
                LB = []
                m = []
                d = []
                for j in range(self.number_of_jobs):
                    for k in range(self.num_operation[y][j]):
                        LB.append(self.LB[y, j, k])
                        m.append(self.mask_mch[y, j, k])
                        d.append(self.dur[y, j, k])
                LBm.append(LB)
                mch_mask.append(m)
                durs.append(d)
            self.LBm = np.array(LBm)

            self.mask_mch = np.array(mch_mask)
            self.op_dur = np.array(durs)

        self.initQuality = np.ones(self.batch_sie)
        for i in range(self.batch_sie):
            self.initQuality[i] = (
                self.LBm[i].max() if not configs.init_quality_flag else 0
            )

        self.max_endTime = self.initQuality

        self.job_time = np.zeros((self.batch_sie, self.number_of_jobs))
        self.finished_mark = np.zeros(shape=(self.batch_sie, self.number_of_tasks))
        # --------------------------------------------------------------------------------------------------------------------------
        """fea = self.LBm.reshape(self.batch_sie,-1, 1) / configs.et_normalize_coef"""
        fea = np.concatenate(
            (
                self.LBm.reshape(self.batch_sie, -1, 1) / configs.et_normalize_coef
                # ,np.expand_dims(self.job_time,2).repeat(self.number_of_machines,axis=2).reshape(self.batch_sie,self.number_of_tasks,1)/ configs.et_normalize_coef
                ,
                self.finished_mark.reshape(self.batch_sie, self.number_of_tasks, 1),
            ),
            axis=-1,
        )

        # --------------------------------------------------------------------------------------------------------------------------
        """fea = self.dur.reshape(self.batch_sie, -1, self.number_of_machines)/configs.et_normalize_coef"""

        """fea = np.concatenate((self.LBs.reshape(self.batch_sie,-1, 2)/configs.et_normalize_coef,
                                #self.dur.reshape(self.batch_sie,-1,self.number_of_machines)/configs.high,
                              # self.dur.reshape(-1, 1)/configs.high,
                              # wkr.reshape(-1, 1)/configs.wkr_normalize_coef,
                              self.finished_mark.reshape(self.batch_sie,-1, 1)), axis=-1)"""
        # initialize feasible omega
        self.omega = self.first_col.astype(np.int64)
        # initialize mask
        self.mask = np.full(
            shape=(self.batch_sie, self.number_of_jobs), fill_value=0, dtype=bool
        )
        # self.mask = self.update_mask(self.mask, self.adj)
        self.mch_time = np.zeros((self.batch_sie, self.number_of_machines))
        # start time of operations on machines
        self.mchsStartTimes = -configs.high * np.ones(
            (self.batch_sie, self.number_of_machines, self.number_of_tasks)
        )
        self.mchsEndTimes = -configs.high * np.ones(
            (self.batch_sie, self.number_of_machines, self.number_of_tasks)
        )
        # Ops ID on machines
        self.opIDsOnMchs = -self.number_of_jobs * np.ones(
            (self.batch_sie, self.number_of_machines, self.number_of_tasks),
            dtype=np.int32,
        )
        self.up_mchendtime = np.zeros_like(self.mchsEndTimes)
        # 用number_of_jobs填充数组的形状
        self.temp1 = np.zeros((self.batch_sie, self.number_of_jobs, self.max_operation))
        if rule != None:
            dur = self.dur_cp.reshape(self.batch_sie, -1, self.max_operation)
        else:
            dur = self.op_dur

        # self.mask_mch = self.mask_mch.reshape(self.batch_sie,-1,self.mask_mch.shape[-1])
        return (
            self.adj,
            fea,
            self.omega,
            self.mask,
            self.mask_mch,
            dur,
            self.mch_time,
            self.job_time,
        )


global_tiff_count = 0


class DFJSP_GANTT_CHART:
    def __init__(self, total_n_job, number_of_machines, product_arr=None):
        super(DFJSP_GANTT_CHART, self).__init__()
        self.frames = []
        self.total_n_job = total_n_job
        self.number_of_machines = number_of_machines
        self.product_arr = product_arr
        self.initialize_plt()
        self.init_tiff()

    def colour_gen(self, n):
        """
        为工件生成随机颜色
        :param n: 工件数
        :return: 颜色列表
        """
        color_bits = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
        ]
        self.colours = []
        random.seed(234)
        for i in range(n):
            colour_bits = ["#"]
            colour_bits.extend(random.sample(color_bits, 6))
            self.colours.append("".join(colour_bits))

    def initialize_plt(self):
        self.colour_gen(self.product_arr[-1] + 1)
        plt.figure(figsize=((self.total_n_job * 1.5, self.number_of_machines)))
        y_value = list(range(1, 21))

        plt.xlabel("Makespan", size=20, fontdict={"family": "SimSun"})
        plt.ylabel("Machine", size=20, fontdict={"family": "SimSun"})
        plt.yticks(y_value, fontproperties="Times New Roman", size=20)
        plt.xticks(fontproperties="Times New Roman", size=20)

    def init_tiff(self):
        # 创建 pic 目录（如果不存在）
        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.pic_dir = os.path.join(current_directory, "FJSP_FIGURE", "pic")
        if not os.path.exists(self.pic_dir):
            os.makedirs(self.pic_dir, exist_ok=True)

    def gantt_plt(self, job, operation, mach_a, start_time, dur_a, product_job_indices):
        """
        绘制甘特图
        :param job: 工件号
        :param operation: 工序号
        :param mach_a: 机器号
        :param start_time: 开始时间
        :param dur_a: 加工时间
        :param colors: 颜色列表
        """
        # colors = self.colour_gen(number_of_jobs)
        plt.barh(
            mach_a + 1,
            dur_a,
            0.5,
            left=start_time,
            color=self.colours[self.product_arr[job]],
        )
        if job in np.array(product_job_indices).flatten():
            text = f"P{int(self.product_arr[job] + 1)}"
        else:
            text = f"P{int(self.product_arr[job] + 1)}\nJ{job - np.where(self.product_arr == self.product_arr[job])[0][0]}\nO{operation + 1}"
        plt.text(
            start_time + dur_a / 30,
            mach_a + 0.8,
            text,
            size=6,
        )
        if os.environ.get("FJSP_VERBOSE", "0") == "1":
            plt.pause(0.001)
            # save to tif
            # 创建一个内存缓冲区
            with io.BytesIO() as buf:
                plt.savefig(buf, format="png")
                buf.seek(0)
                img = Image.open(buf)
                self.frames.append(img.copy())

    def save_to_tiff(self):
        global global_tiff_count
        global_tiff_count += 1
        self.filename = f"{self.pic_dir}/J{self.total_n_job}_ma{self.number_of_machines}_{global_tiff_count}.tif"
        self.filename1 = f"{self.pic_dir}/J{self.total_n_job}_ma{self.number_of_machines}_{global_tiff_count}.svg"

        if os.path.exists(self.filename):
            os.remove(self.filename)

        if len(self.frames) > 1:
            self.frames[0].save(
                self.filename,
                save_all=True,
                append_images=self.frames[1:],
                compression="tiff_lzw",
            )
        plt.savefig(self.filename1, format="svg", dpi=300, bbox_inches="tight")
