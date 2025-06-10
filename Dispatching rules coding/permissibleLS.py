from Params import configs
import numpy as np


def permissibleLeftShift(
    a,
    mch_a,
    durMat,
    mchMat,
    mchsStartTimes,
    opIDsOnMchs,
    mchEndTime,
    pid,
    row,
    col,
    first_col,
    last_col,
):  #
    # a=action, durMat=self.dur, mchMat=mchaine, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs

    jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(
        a,
        mch_a,
        mchMat,
        durMat,
        mchsStartTimes,
        opIDsOnMchs,
        pid,
        row,
        col,
        first_col,
        last_col,
    )

    dur_a = durMat[row][col][mch_a]

    startTimesForMchOfa = mchsStartTimes[mch_a]  # 机器mch_a的start数组
    endtineformch0fa = mchEndTime[mch_a]
    # print('starttimesformchofa',startTimesForMchOfa)
    opsIDsForMchOfa = opIDsOnMchs[mch_a]  # 机器mch_a处理task的数组
    flag = False
    possiblePos = np.where(jobRdyTime_a < startTimesForMchOfa)[0]

    # machine中以调度的task的开始时间大于job中action的上一个task的完工时间

    if len(possiblePos) == 0:

        startTime_a = putInTheEnd(
            a,
            jobRdyTime_a,
            mchRdyTime_a,
            startTimesForMchOfa,
            opsIDsForMchOfa,
            endtineformch0fa,
            dur_a,
        )

    else:
        idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(
            dur_a,
            mch_a,
            jobRdyTime_a,
            durMat,
            possiblePos,
            startTimesForMchOfa,
            opsIDsForMchOfa,
            first_col,
            last_col,
        )
        # print('legalPos:', legalPos)
        if len(legalPos) == 0:
            startTime_a = putInTheEnd(
                a,
                jobRdyTime_a,
                mchRdyTime_a,
                startTimesForMchOfa,
                opsIDsForMchOfa,
                endtineformch0fa,
                dur_a,
            )
        else:
            flag = True
            startTime_a = putInBetween(
                a,
                idxLegalPos,
                legalPos,
                endTimesForPossiblePos,
                startTimesForMchOfa,
                opsIDsForMchOfa,
                endtineformch0fa,
                dur_a,
            )
    return startTime_a, flag


def putInTheEnd(
    a,
    jobRdyTime_a,
    mchRdyTime_a,
    startTimesForMchOfa,
    opsIDsForMchOfa,
    endtineformch0fa,
    dur_a,
):
    # index = first position of -config.high in startTimesForMchOfa
    # print('Yes!OK!')

    index = np.where(startTimesForMchOfa == -configs.high)[0][0]
    startTime_a = max(jobRdyTime_a, mchRdyTime_a)

    startTimesForMchOfa[index] = startTime_a

    opsIDsForMchOfa[index] = a

    endtineformch0fa[index] = startTime_a + dur_a

    return startTime_a


def calLegalPos(
    dur_a,
    mch_a,
    jobRdyTime_a,
    durMat,
    possiblePos,
    startTimesForMchOfa,
    opsIDsForMchOfa,
    first_col,
    last_col,
):
    startTimesOfPossiblePos = startTimesForMchOfa[
        possiblePos
    ]  # possiblepos有可能是一个有可能是多个task，找到machine中tasks的starttimefomach
    durOfPossiblePos = []
    for possiblePo in possiblePos:
        row = np.where(opsIDsForMchOfa[possiblePo] <= last_col)[0][0]
        col = opsIDsForMchOfa[possiblePo] - first_col[row]
        durOfPossiblePos.append(durMat[row, col][mch_a])

    durOfPossiblePos = np.array(durOfPossiblePos)  # tasks的加工时间
    if possiblePos[0] != 0:
        row1 = np.where(opsIDsForMchOfa[possiblePos[0] - 1] <= last_col)[0][0]
        col1 = opsIDsForMchOfa[possiblePos[0] - 1] - first_col[row1]

        startTimeEarlst = max(
            jobRdyTime_a,
            startTimesForMchOfa[possiblePos[0] - 1] + durMat[row1, col1][mch_a],
        )
    else:
        startTimeEarlst = max(jobRdyTime_a, 0)

    endTimesForPossiblePos = np.append(
        startTimeEarlst, (startTimesOfPossiblePos + durOfPossiblePos)
    )[
        :-1
    ]  # end time for last ops don't care

    possibleGaps = startTimesOfPossiblePos - endTimesForPossiblePos

    idxLegalPos = np.where(dur_a <= possibleGaps)[0]

    legalPos = np.take(possiblePos, idxLegalPos)

    return idxLegalPos, legalPos, endTimesForPossiblePos


def putInBetween(
    a,
    idxLegalPos,
    legalPos,
    endTimesForPossiblePos,
    startTimesForMchOfa,
    opsIDsForMchOfa,
    endtineformch0fa,
    dur_a,
):
    earlstIdx = idxLegalPos[0]
    # print('idxLegalPos:', idxLegalPos)
    earlstPos = legalPos[0]
    startTime_a = endTimesForPossiblePos[earlstIdx]
    # print('endTimesForPossiblePos:', endTimesForPossiblePos)
    startTimesForMchOfa[:] = np.insert(startTimesForMchOfa, earlstPos, startTime_a)[:-1]
    endtineformch0fa[:] = np.insert(endtineformch0fa, earlstPos, startTime_a + dur_a)[
        :-1
    ]
    opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, a)[:-1]

    return startTime_a


def calJobAndMchRdyTimeOfa(
    a,
    mch_a,
    mchMat,
    durMat,
    mchsStartTimes,
    opIDsOnMchs,
    pid,
    row,
    col,
    first_col,
    last_col,
):
    # numpy.take（a，indices，axis = None，out = None，mode ='raise' ）取矩阵中所有元素的第a个元素
    # cal jobRdyTime_a
    if col != 0:
        jobPredecessor = a - 1
    else:
        jobPredecessor = pid

    # jobPredecessor = a - 1 if col != 0 else None#if a % mchMat.shape[1] = 0即该job调度完成或为第一个调度的task
    # job中action前一个task
    if jobPredecessor is not None:
        if col == 0:
            row = np.where(pid <= last_col)[0][0]
            col = pid - first_col[row] + 1

        mchJobPredecessor = mchMat[row][col - 1]  # 处理该task的机器
        durJobPredecessor = durMat[row, col - 1, mchJobPredecessor]  # 加工时间
        jobRdyTime_a = (
            mchsStartTimes[mchJobPredecessor][
                np.where(opIDsOnMchs[mchJobPredecessor] == jobPredecessor)
            ]
            + durJobPredecessor
        ).item()  # opIDsOnMchs->对应mchJobPredecessor----shape（machine,n_job）
        # 找到数组opIDsOnMchs[mchJobPredecessor]中等于jobPredecessor的索引值####opIDsOnMchs->shape(machine,job)
    else:
        jobRdyTime_a = 0
    # cal mchRdyTime_a

    mchPredecessor = (
        opIDsOnMchs[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1]
        if len(np.where(opIDsOnMchs[mch_a] >= 0)[0]) != 0
        else None
    )

    # machine中action前一个task

    if mchPredecessor is not None:

        row_1 = np.where(mchPredecessor <= last_col)[0][0]
        col_1 = mchPredecessor - first_col[row_1]
        durMchPredecessor = durMat[row_1, col_1, mch_a]

        mchRdyTime_a = (
            mchsStartTimes[mch_a][np.where(mchsStartTimes[mch_a] >= 0)][-1]
            + durMchPredecessor
        ).item()

        # np.where()返回一个索引数组，这里返回在该machine中以调度task的索引。最后返回machine中action上一个task的结束时间
    else:

        mchRdyTime_a = 0

    return jobRdyTime_a, mchRdyTime_a


"""if __name__ == "__main__":
    from FJSP_Env import FJSP
    from uniform_instance import uni_instance_gen, FJSPDataset
    import time
    from torch.utils.data import DataLoader
    import os
    from DataRead import getdata

    filename = "./data"
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
    for i in range(1, 2):
        result = []
        for file in filename:
            a = runDRs(DRs[i], file)
            result.append(a)
        results.append(result)
    print(results)
"""
