
## 安装教程

- 安装 Anaconda
- 激活环境
- 打开 <https://pytorch.org/get-started/locally/>，安装 pytorch
- 执行 "conda install pandas jupyter jupyterlab scikit-learn matplotlib"
- "jupyter lab"

https://betterdatascience.com/pytorch-install/#pytorch-version-check


有些操作尚未使用 MPS 实现，我们可能需要设置一些环境变量来使用 CPU 回退，参考以下链接

conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

https://stackoverflow.com/questions/72416726/how-to-move-pytorch-model-to-gpu-on-apple-m1-chips
