# GPU-DDVFS: Data Driven GPU Dynamic Voltage Scaling

## Abstract
Modern computing paradigms, such as cloud computing, are increasingly adopting GPUs to boost their computing capabilities primarily 
due to the heterogeneous nature of AI/ML/deep learning workloads. However, the energy consumption of GPUs is a critical problem. 
Dynamic Voltage Frequency Scaling (DVFS) is a widely used technique to reduce the dynamic power of GPUs. 
Yet, configuring the optimal clock frequency for essential performance requirements is a non-trivial task due to the complex nonlinear
relationship between the application's runtime performance characteristics, energy, and execution time.
It becomes more challenging when different applications behave distinctively with similar clock settings.
Simple analytical solutions and standard GPU frequency scaling heuristics fail to capture these intricacies and scale 
the frequencies appropriately. In this regard, this work propose a data-driven frequency scaling technique by predicting the power and
execution time of a given application over different clock settings. We collect the data from application profiling and train the
models to predict the outcome accurately. The proposed solution is generic and can be easily extended to different kinds of workloads and GPU architectures. Furthermore, using this frequency scaling by prediction models, we present a deadline-aware application scheduling algorithm to reduce energy consumption while simultaneously meeting their deadlines. We conduct real extensive experiments on  NVIDIA GPUs using several benchmark applications. The experiment results have shown that our prediction models have high accuracy with the average RMSE values of 0.38 and 0.05 for energy and time prediction, respectively. 
Also, the scheduling algorithm consumes $15.07\%$ less energy as compared to the baseline policies.

## Main Features of the Work:
 1. A data-driven prediction model to accurately predict the energy and execution time of applications to assist the efficient frequency scaling
 configuration by observing key architectural, power, and performance counters and metrics.
 2. Design and present a deadline-aware energy-efficient application scheduling algorithm using the prediction models. 
 3. A prototype system and evaluate the proposed solution on a real platform using standard benchmarking applications
 4. Comparison of results with existing state-of-the-art solutions. 



## References 
Shashikant Ilager, Rajeev Muralidhar, Kotagiri Rammohanrao, and Rajkumar Buyya, [A Data-Driven Frequency Scaling Approach for Deadline-aware Energy Efficient Scheduling on Graphics Processing Units (GPUs)](http://www.buyya.com/papers/DDFreqScaleGPU.pdf), Proceedings of the 20th IEEE/ACM International Symposium on Cluster, Cloud, and Internet Computing (CCGrid 2020, IEEE CS Press, USA), 
Melbourne, Australia, May 11-14, 2020.

