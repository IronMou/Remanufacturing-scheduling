This is the coding of our paper "CombinedConstraint Rule withDeep Reinforcement Learning Method to Optimize Multi-stageRemanufacturing Scheduling Problem".

File 'DRL coding'
This coding presents an implementation of the combined constraint rule with deep reinforcement learning method applied to the multi-stage remanufacturing scheduling problem. Remanufacturing scheduling contains integrated flexible-job-shop-based disassembly, reprocessing, and reassembly shops. Proximal policy optimization algorithm is applied to optimize scheduling decisions. Combined with graph neural network to model this multi-stage scheduling problem, using adjacency matrices to capture the multi-stage constraints inherent in remanufacturing scheduling. The in-degree and constraint values of operation nodes are generated into constraint rules, enabling a more precise characterization of inter-node relationships. For testing the instance, you can run Remanufacturing-scheduling/DRL coding/RM_RealWorld/validation_realWorld.py.

File 'Dispatching rules coding'
For operation sequencing rule, for classic rules are selected: First in First Out (FIFO), Most Operation Number Remaining (MOPNR), LeastWork Remaining (LWKR), and Most Work Remaining (MWKR).
For workstation allocation rule is to select the compatible workstation for the scheduled operation, two classics are selected: Shortest Processing Time (SPT) and Earliest End Time(EET).
Thus, 8 compound rules are set for coding to do the experiments: FIFO_SPT, FIFO_EET, MOPNR_SPT, MOPNR_EET, LWKR_SPT, LWKR_EET, MWKR_SPT, MWKR_EET.

File 'Experiment results'
You can find the experiment results (including makespan, computation time and gantt charts) of all methods (GA, GAVNS, 8 compound dispatching rules and our method) shown in the paper.

File 'dataset_fjs'
It contains 18 instances which can be divided into three categories with different problem scales: (a) small-size instances (S01 to S06);(b)medium-sizeinstances(M01 to M06);(c)large-size instances (L01 to L06). We extend the classic format (fjs) of flexible job-shop datasets, adding the product and component elements to better satisfy the special and unique relationship of end-of-life products. The characteristic of classic flexible job-shop problem contains the job-operation relationship. Whileas the remanufacturing scheduling problem contains the product-component-operation relationship. Thus, the RM_fjs dataset is improved, the difference between our RM_fjs and calssic fjs datatset is the first line. The fourth number of the first line represents the quantity of products; Following this, each number represents the quantity of components.
