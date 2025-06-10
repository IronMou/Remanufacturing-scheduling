import time
import numpy as np
from torch.utils.data import DataLoader
from DataRead import get_imlist, getdata
from FJSP_Env import FJSP, logger
from Params import configs
from uniform_instance import uni_instance_gen, FJSPDataset


def runDRs(rule, file):
    batch_size = 1
    n_j = 6
    n_m = 6
    num_operation = np.array([[6, 5, 6, 6, 6, 6]]).astype(int)
    """num_opt = []
    for i in range(batch_size):
        n = np.full(shape=(n_m),fill_value=n_m)
        num_opt.append(n)
    num_operation = np.array(num_opt)"""

    low = -99
    high = 99
    SEED = 200
    # np.random.seed(SEED)
    t3 = time.time()
    # ------------------------------------------------------------------
    """train_dataset = FJSPDataset(n_j, n_m, low, high,1,seed=SEED)
    np.random.seed(200)"""
    # ------------------------------------------------------------------
    Data = getdata(file)
    n_j = Data["n"]
    n_m = Data["m"]
    num_operation = []
    num_operations = []
    for i in Data["J"]:
        num_operation.append(Data["OJ"][i][-1])
    num_operation_max = np.array(num_operation).max()

    time_window = np.zeros(shape=(Data["n"], num_operation_max, Data["m"]))

    data_set = []
    for i in range(Data["n"]):

        for j in Data["OJ"][i + 1]:
            mchForJob = Data["operations_machines"][(i + 1, j)]
            for k in mchForJob:
                time_window[i][j - 1][k - 1] = Data["operations_times"][(i + 1, j, k)]

    for i in range(batch_size):
        num_operations.append(num_operation)
        data_set.append(time_window)
    num_operation = np.array(num_operations)

    number_of_tasks = num_operation.sum(axis=1)[0]
    number_of_tasks = int(number_of_tasks)
    train_dataset = np.array(data_set)
    # ------------------------------------------------------------------
    data_loader = DataLoader(train_dataset, batch_size=batch_size)
    result = []
    for batch_idx, data_set in enumerate(data_loader):
        data_set = data_set.numpy()
        batch_size = data_set.shape[0]

        env = FJSP(
            n_j=n_j,
            n_m=n_m,
            EachJob_num_operation=num_operation,
            product_num_jobs=Data["product_num_jobs"],
            need_gantt=True,
        )
        # rollout env random action
        t1 = time.time()
        # data = uni_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high,seed=SEED)

        # start time of operations on machines
        mchsStartTimes = -configs.high * np.ones((n_m, n_m * n_j), dtype=np.int32)
        mchsEndtTimes = -configs.high * np.ones((n_m, n_m * n_j), dtype=np.int32)
        # Ops ID on machines
        opIDsOnMchs = -n_j * np.ones([n_m, n_m * n_j], dtype=np.int32)

        # random rollout to test
        # count = 0
        adj, _, omega, mask, mch_mask, _, mch_time, _ = env.reset(data_set, rule)

        # print(env.adj)
        # mch_mask = mch_mask.reshape(batch_size, -1,n_m)
        job = omega
        rewards = []
        flags = []
        # ts = []
        # print(env.mask_mch[0])

        d = 0
        t1 = time.time()
        while True:
            action = []
            mch_a = []
            # logger.info(f"mask before: {mask}")
            mask = env.update_mask()
            # logger.info(f"mask after: {mask}")
            for i in range(batch_size):
                # choose job
                a = np.random.choice(omega[i][np.where(mask[i] == 0)])
                # index = np.where(job[i] == a)[0].item()
                row = np.where(a <= env.last_col[i])[0][0]
                col = a - env.first_col[i][row]
                # choose machine
                m = np.random.choice(np.where(mch_mask[i][row][col] == 0)[0])

                action.append(a)
                mch_a.append(m)
            # logger.info(f"job action: {action}  \tmachine action: {mch_a}")
            d += 1

            """mch_a = np.random.choice()
            mch_a = PredictMch(env,action,1)"""

            """row = action // n_j  # 取整除
            col = action % n_m  # 取余数
            job_time=data_set[0][row][col]

            mch_a=np.random.choice(np.where(job_time>0)[0])"""

            """dur_a=data[row][col][mch_a]
            print(mch_a)
            print('action:', action)
            t3 = time.time()
            print('env_opIDOnMchs\n', env.opIDsOnMchs)
            print('11',env.mchsEndTimes[0])"""

            adj, _, reward, done, omega, mask, job, mch_mask, mch_time, _ = env.step(
                action, mch_a
            )

            """print('33',env.mchsEndTimes[0])
            print('reward',reward[0],env.dur_a)
            t4 = time.time()
            ts.append(t4 - t3)
            jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(a=action,mch_a=mch_a, mchMat=m, durMat=data, mchsStartTimes=mchsStartTimes, opIDsOnMchs=opIDsOnMchs)
            print('mchRdyTime_a:', mchRdyTime_a,"\n",'jobrdytime',jobRdyTime_a)

            startTime_a, flag = permissibleLeftShift(a=action, mch_a=mch_a,durMat=data.astype(np.single), mchMat=m, mchsStartTimes=mchsStartTimes, opIDsOnMchs=opIDsOnMchs,mchEndTime=mchsEndtTimes,dur_a=dur_a)
            flags.append(flag)

            print('startTime_a:', startTime_a)
            print('mchsStartTimes\n', mchsStartTimes)
            print('NOOOOOOOOOOOOO' if not np.array_equal(env.mchsStartTimes, mchsStartTimes) else '\n')
            print('opIDsOnMchs\n', opIDsOnMchs)"""

            # print('LBs\n', env.LBs)
            rewards.append(reward)
            # print('ET after action:\n', env.LBs)
            # print()
            if env.done():
                if env.need_gantt:
                    env.gantt.save_to_tiff()
                break
        t2 = time.time()
        logger.info(f"latency: {t2-t1:.3f} s")
        """print(sum(ts))
        print(np.sum(opIDsOnMchs // n_m, axis=1))
        print(np.where(mchsStartTimes == mchsStartTimes.max()))
        print(opIDsOnMchs[np.where(mchsStartTimes == mchsStartTimes.max())])
        print(mchsStartTimes.max() + np.take(data[0], opIDsOnMchs[np.where(mchsStartTimes == mchsStartTimes.max())]))
        np.save('sol', opIDsOnMchs // n_m)
        np.save('jobSequence', opIDsOnMchs)
        np.save('testData', data)
        print(mchsStartTimes)
        print(data)"""

        result.append(env.mchsEndTimes.max(-1).max(-1))

        # print(len(np.where(np.array(rewards) == 0)[0]))
        # print(rewards)
        """print()

        print(env.mchsStartTimes)
        print('reward---------------', env.mchsEndTimes, env.mchsEndTimes.max(-1).max(-1))
        print()
        print(env.opIDsOnMchs[0])
        print(env.adj[0])
        # print(sum(flags))
        # data = np.load('data.npy')
        t4 = time.time() - t3
        print(t4)"""

    logger.info(f"total machine time for {rule}: {np.array(result).mean()}")
    return np.array(result).mean()


if __name__ == "__main__":

    filename = ".//data//fjs"
    filename = get_imlist(filename)
    print(filename)
    DRs = [
        "FIFO_SPT",
        "FIFO_EET",
        "MOPNR_SPT",
        "MOPNR_EET",
        "LWKR_SPT",
        "LWKR_EET",
        "MWKR_SPT",
        "MWKR_EET",
    ]
    results = []
    for i in range(len(DRs)):
        result = []
        for file in filename:
            a = runDRs(
                DRs[i],
                file,
            )
            result.append(a)
        results.append(result)
    # logger.info(f"total machine time:\n{np.array(results)}")
