import numpy as np
import pandas as pd

def transformData(filename='./data/S03.xlsx'):
    # Convert the updated table to FJS format considering '-' represents infeasible machine operations
    df = pd.read_excel(filename, sheet_name='RW-Instance1')
    df = df.replace('-', np.nan)

    product_job_to_id = {}
    current_id = 0

    # 遍历DataFrame
    for index, row in df.iterrows():
        # 获取当前行的product和job
        product = row['Product']
        job = row['Job']
        #action=row['Operation']

        # 检查(product, job)是否已在字典中
        if (product, job) not in product_job_to_id:
            # 如果不在，添加到字典中并将current_id作为它的值
            product_job_to_id[(product, job)] = current_id
            # ID递增
            current_id += 1
    id_to_product_job = {v: k for k, v in product_job_to_id.items()}

    # Get the unique combinations of products and jobs as unique job identifiers
    product_job_combinations = df.groupby(['Product', 'Job']).size().reset_index()[['Product', 'Job']]

    # Initialize the fjs data structure with the number of unique job combinations and the number of machines
    machine_columns = [col for col in df.columns if 'W' in col]
    fjs_data = [[len(product_job_combinations), len(machine_columns)]]

    # Now we build the data for each unique combination of product and job
    for _, row in product_job_combinations.iterrows():
        product, job = row['Product'], row['Job']

        # Get all operations for the current product and job combination
        operations = df[(df['Product'] == product) & (df['Job'] == job)]

        # The first number in the job's data is the number of operations for that job
        job_data = [operations['Operation'].nunique()]

        # Add the job data to the fjs dataset
        fjs_data.append(job_data)

        # Handle special rows for each product
        special_rows = {}
        for prod in df['Product'].unique():
            first_row = df[(df['Product'] == prod) & (df['Job'] == df[df['Product'] == prod]['Job'].min())]
            last_row = df[(df['Product'] == prod) & (df['Job'] == df[df['Product'] == prod]['Job'].max())]

            if not first_row.empty:
                special_rows[('first', prod)] = first_row.iloc[0].tolist()
            if not last_row.empty:
                special_rows[('last', prod)] = last_row.iloc[0].tolist()

            # Now we add the operations data
            for _, operation_row in operations.iterrows():
                # Filter out the infeasible machine operations
                feasible_operations = operation_row[machine_columns].dropna()

                # The first number for each operation is the count of feasible machines
                job_data.append(len(feasible_operations))

                # Add each feasible machine and its processing time for the operation
                for idx, time in feasible_operations.items():
                    machine_index = machine_columns.index(idx) + 1  # Convert machine name to index
                    job_data.extend([int(machine_index), int(time)])

                    # Add special rows for the current product
                for key, value in special_rows.items():
                    if key[0] == 'first':
                        fjs_data.insert(0, [1, len(machine_columns)])  # Add the first special row
                    elif key[0] == 'last':
                        fjs_data.append([1, len(machine_columns)])  # Add the last special row

    return fjs_data, id_to_product_job


def getdata(filename='./FJSSPinstances/0_BehnkeGeiger/Behnke1.fjs'):

    # filename='./FJSSPinstances/1_Brandimarte/BrandimarteMk1.fjs'
    # filename='./FJSSPinstances/0_BehnkeGeiger/Behnke1.fjs'
    # filename='./FJSSPinstances/2a_Hurink_sdata/HurinkSdata1.fjs'
    f=open(filename,'r')
    line=f.readline()

    line_data=line.split()

    numbers_float=list(map(float,line_data))

    n=int(numbers_float[0])
    m=int(numbers_float[1])
    average_num_machine=numbers_float[2]
    #print(n)
    #print(m)
    operations_machines={}#the available machines for each operation of jobs
    operations_times={}#the processing time for each operation of jobs


    # jobs=[[]for i in range(n)]
    numonJobs=[]
    # print(jobs)
    for i in range(n):
        line=f.readline()
        line_data=line.split()
        numbers_float = list(map(int, line_data))
        operation_num=int(numbers_float[0])
        numonJobs.append(operation_num)
        # operations=[[] for j in range(operation_num)]
        jj=1
        j=0
        while jj<len(numbers_float):
            o_num=int(numbers_float[jj])
            job_op=[]
            job_machines=[]
            job_processingtime=[]
            for kk in range(0,o_num*2,2):
                # job_op.append(numbers_float[jj+kk+1])
                job_machines.append(numbers_float[jj+kk+1])
                job_processingtime.append(numbers_float[jj+kk+1+1])
            # operations[j]=job_op
            operations_machines[(i+1,j+1)]=job_machines
            for l in range(len(job_machines)):
                operations_times[(i + 1, j + 1,job_machines[l])] = job_processingtime[l]

            j+=1
            jj+=o_num*2+1
    f.close()  # close the file
        # jobs[i]=operations

# print(operations_machines)
    # print(operations_times)
    # print(numonJobs)
    J=list(range(1,n+1)) #define the index of jobs
    M=list(range(1,m+1)) #define the index of machines
    OJ={}
    for j in range(n):
        OJ[(J[j])]=list(range(1,numonJobs[j]+1))

    #define large_M
    largeM=0
    for job in J:
        for op in OJ[(job)]:
            protimemax=0
            for l in operations_machines[(job,op)]:
                if protimemax<operations_times[(job,op,l)]:
                    protimemax=operations_times[(job,op,l)]
            largeM+=protimemax


    Data={
        'n':n,
        'm':m,
        'J':J,
        'M':M,
        'OJ':OJ,
        'operations_machines':operations_machines,
        'operations_times':operations_times,
        'largeM':largeM,
    }
    return Data


if __name__ == '__main__':
    data=getdata('./FJSSPinstances/1_Brandimarte/BrandimarteMk6.fjs')
    print(data)
    print(largeM)
    print(operations_times)

    data=[]
    for line in open(filename,"r"):
        line_data=line.split()
        numbers_float=map(float,line_data)
        print(line_data)
        print(type(line_data))
        data.append(line)
    print(data)
    with open(filename,'r') as f:
        my_data=f.readline()
        line_data=my_data.split()
        numbers_float = list(map(float, line_data))
        print(line_data)
        print(numbers_float)
        for line in my_data:
            line_data=line.split()
            numbers_float=map(float,line_data)
        print(numbers_float)
