Speaker: Xiang Luo

Title: Integer linear programming and its use in polyhedral-based deep learning compiler

Abstract:

For compute-intensive applications which are particularly common in scientific application and deep learning, most running time has been spent in nested loops. Ways to tile these loops into small blocks to fully take advantage of parallelism and locality to boost the performance have been widely investigated for decades. The polyhedral model provides a powerful abstraction to reason about these affine transformations in the form of linear programming. Moreover, recent works such as AKG and Tensor Comprehensions have demonstrated the polyhedral model a promising direction in deep learning compilers. By formulating the scheduling problem during compilation into an integer linear programming (ILP) problem, we can avoid dealing with intricate hardware-related issues and focus on a clean optimization problem instead .

In this talk, I will first give a brief introduction about the deep learning compiler, and then discuss how the polyhedral model, as a pass of dl compilers, formulates the scheduling problem into an ILP problem. Finally, I will present the simplex-like method from PIP to solve the ILP problem.

---

Reference: 

[1]【深度学习编译器前沿综述】Mingzhen Li, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hailong Yang, Zhongzhi Luan, Lin Gan, Guangwen Yang, and Depei Qian. The deep learning compiler: A comprehensive survey. IEEE Transactions on Parallel and Distributed Systems, 32(3):708–727, 2020. https://arxiv.org/pdf/2002.03794.pdf

[2] U. Bondhugula, M. Baskaran, S. Krishnamoorthy, J. Ramanujam, A. Rountev, and P. Sadayappan. Affine transformations for communication minimal parallelization and locality optimization of arbitrarily-nested loop sequences. Technical Report OSU-CISRC5/07-TR43, The Ohio State University, May 2007. https://www.ece.lsu.edu/jxr/Publications-pdf/tr43-07.pdf

[3]【单纯形算法】E. K. P. Chong and S. H. Zak, An Introduction to Optimization. New York: Wiley, 2001.

[4] P. Feautrier. Parametric integer programming. Operationnelle/Operations Research, 22(3):243–268, 1988. http://www.numdam.org/item/RO_1988__22_3_243_0.pdf

[5] 赵捷 Polyhedral 编译调度算法：

Pluto算法 https://zhuanlan.zhihu.com/p/199683290

Feautrier算法 https://zhuanlan.zhihu.com/p/232070003

isl中的调度算法 https://zhuanlan.zhihu.com/p/259311866

[6] Jie Zhao, Bojie Li, Wang Nie, Zhen Geng, Renwei Zhang, Xiong Gao, Bin Cheng, Chen Wu, Yun Cheng, Zheng Li, Peng Di, Kun Zhang, and Xuefeng Jin. 2021. AKG: automatic kernel generation for neural processing units using polyhedral transformations. In Proceedings of the 42nd ACM SIGPLAN International Conference on Programming Language Design and Implementation (PLDI 2021). Association for Computing Machinery, New York, NY, USA, 1233–1248. DOI:https://doi.org/10.1145/3453483.3454106

[7] 陈天奇 深度学习编译技术的现状和未来 https://zhuanlan.zhihu.com/p/65452090

[8] S. Chetlur, C. Woolley, P. Vandermersch, J. Cohen, J. Tran, B. Catanzaro, and E. Shelhamer. cuDNN: Efficient primitives for deep learning. arXiv preprint, 1410.0759, 2014. arxiv.org/abs/1410.0759.

[9] 【TVM 编译优化 Pytorch】

使用TVM编译优化Pytorch模型 https://bbs.huaweicloud.com/blogs/224847

TVM-Pytorch模型编译体验+性能测试 https://blog.csdn.net/jiangpeng59/article/details/105516970

Optimizing PyTorch models for fast CPU inference using Apache TVM https://spell.ml/blog/optimizing-pytorch-models-using-tvm-YI7pvREAACMAwYYz

