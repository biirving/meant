# MEANT
This repository contains the code from the following paper:

MEANT:Multimodal encoding with antecedent information. 

## IMPORTANT
For the data, please contact the author: irving.b@northeastern.edu
A lot of the scripts won't run, because they require paths to data that you don't have. Please contact the authors for information about running tests,
and to get data.


## Other peoples code
We use the TimeSformer implementation from Phil Wang https://github.com/lucidrains/TimeSformer-pytorch, along with his rotary embedding implementation https://github.com/lucidrains/rotary-embedding-torch. 

We also use FlashAttention https://github.com/Dao-AILab/flash-attention. 

```
@misc{bertasius2021spacetime,
    title   = {Is Space-Time Attention All You Need for Video Understanding?}, 
    author  = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    year    = {2021},
    eprint  = {2102.05095},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
@misc{su2021roformer,
    title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
    author  = {Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
    year    = {2021},
    eprint  = {2104.09864},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
@inproceedings{dao2022flashattention,
  title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
@article{tokshift2021,
    title   = {Token Shift Transformer for Video Classification},
    author  = {Hao Zhang, Yanbin Hao, Chong-Wah Ngo},
    journal = {ACM Multimedia 2021},
}
@inproceedings{dao2023flashattention2,
  title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
