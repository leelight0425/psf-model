# **A Physics-Informed Blur Learning Framework for Imaging Systems**
<p align="center" style="font-size:18px;">
  <a href="https://arxiv.org/abs/2502.11382"><b>📜 Paper</b></a> &nbsp;  
  <a href="https://github.com/OpenImagingLab/PSF-Estimation"><b>💻 Code</b></a> &nbsp;  
  <a href="https://openimaginglab.github.io/PSF-Estimation/"><b>🌐 Project Page</b></a>
</p>



[//]: # (Liqun Chen, Yuyao Hu, Jiewen Nie, Tianfan Xue and Jinwei Gu)

## Environment requirements
The codes was tested on Windows 10, with Python and PyTorch. Required packages:
- numpy  
- tqdm
- python
- matplotlib
- torch
- torchvision
- pandas
- opencv-python
- pyyaml

## File structure
This repository contains codes for OAE(optical aberration estimation).
```
OAE
|   README.md
|   main.py
|   generate_edges_rgb.py
|   generate_fov_weight.py
|
|---configs
|   |   lensname.yaml
|
|---sfrmat5_dist
|   |---sfrmat5
|       |   user_sfrmat5_rgb.m
|
|---dataset 
|   |---lensname
|       |   npy
| 
|---input 
|   |   lensname.xlsx
| 
|---model 
|   |   optics_rgb.py
|   |   PSF_mlp.py
| 
|---results 
|
|---utils 
|   |   tools.py
|   |   train.py
```
`/model` contains the optical aberration model.

`/dataset` includes datasets used for training the optical aberration model.

`/sfrmat5_dist` contains the SFR calculation algorithm, which was downloaded from [ISO 12233](https://www.imaging.org/site/IST/Standards/Digital_Camera_Resolution_Tools/IST/Standards/Digital_Camera_Resolution_Tools.aspx#msw.).

`/results` stores the results, including the PSF map and PSF comparisons.

## SFR Data Preparation

Follow these steps to generate the necessary SFR (Spatial Frequency Response) data for training.

1.  **Generate Blurry Edges:**
    Run the Python script `generate_edges_rgb.py` to process your images and obtain the initial blurry edge data.
    ```bash
    python generate_edges_rgb.py
    ```

2.  **Calculate SFR Curves:**
    Next, open MATLAB and execute the `user_sfrmat5_rgb.m` script. This will analyze the edges from the previous step to compute and save the SFR curves.

3.  **Generate Training Data:**
    Finally, run the `generate_fov_weight.py` script. This will process the SFR curves and generate the final `.npz` files that will be used as input for the training process.
    ```bash
    python generate_fov_weight.py
    ```

## Training
To train an aberration learning model from scratch, run `main.py`. The results will be saved in /results/lensname.

## Q & A
**1. Why is F=5 used as the scaling factor (see scale in optics_rgb.py)?​​**  
<small>When extracting a 2D MTF slice from the 3D MTF data (as implemented in slice within tool.py), we intentionally scale the coordinate system by a factor of 5.  
This expansion:  
    - Reduces quantization errors during interpolation  
    - Improves numerical precision in the slicing operation  
The scaling factor (F=5) is subsequently applied to compensate for this coordinate expansion, ensuring the final results maintain their proper scale and accuracy.</small>

For any questions, please contact: meview.global@gmail.com  
PSF-Estimation 代码使用指南
📜 论文：A Physics-Informed Blur Learning Framework for Imaging Systems (CVPR 2025)
一、项目结构
文本

编辑



PSF-Estimation/
│   README.md
│   main.py                    ← 训练入口
│   generate_edges_rgb.py      ← 步骤1：提取模糊边缘
│   generate_fov_weight.py     ← 步骤3：生成训练数据
│
├── configs/
│   └── lensname.yaml          ← 镜头配置文件
│
├── sfrmat5_dist/
│   └── sfrmat5/
│       └── user_sfrmat5_rgb.m ← 步骤2：MATLAB 计算 SFR 曲线
│
├── dataset/
│   └── lensname/
│       └── *.npy              ← 训练数据（SFR 数据）
│
├── input/
│   └── lensname.xlsx          ← 镜头参数输入
│
├── model/
│   ├── optics_rgb.py          ← 光学像差物理模型
│   └── PSF_mlp.py             ← PSF 估计 MLP 网络
│
├── results/                   ← 输出结果（PSF map + 对比图）
│
└── utils/
    ├── tools.py               ← 工具函数（MTF 切片等）
    └── train.py               ← 训练逻辑



项目概述：PSF-Estimation（光学像差估计 / OAE）
这是一个 CVPR 2025 论文 "A Physics-Informed Blur Learning Framework for Imaging Systems" 的官方代码实现。核心目标是：从镜头拍摄的图像中，自动估计该镜头在全视场范围内的光学像差（PSF，即点扩散函数）。

核心思路
任何相机镜头都不是完美的——不同视场位置、不同波长（RGB）的光线经过镜头后会产生不同程度的模糊（像差），这在数学上由 Zernike 多项式系数 或 Seidel 像差系数 描述。这个项目做的是：

📷 拍摄测试图 → 提取 MTF/SFR 数据 → 用物理模型 + MLP 神经网络反推出 Zernike/Seidel 系数 → 重建全视场的 PSF 分布图

项目文件结构
文件	角色
main.py	训练入口：加载配置 → 初始化光学系统(IS) → 创建 PSF_mlp 网络 → 训练
model/optics_rgb.py	光学物理模型核心：从 Excel 读取镜头设计参数(Zernike系数)，计算 PSF。包含 Zernike→波前→PSF 的完整物理推导
model/PSF_mlp.py	MLP 网络：输入归一化视场高度 H，输出 Seidel 像差系数 → 波前 → PSF。包含 seidel2wavefront、wavefront2psf、shift_net（色差校正）
utils/train.py	训练循环：逐 FOV、逐颜色通道优化 PSF，使其 MTF 逼近真实 SFR 数据。分两阶段：先优化单色 PSF，再优化横向色差（R/B 通道相对 G 的偏移）
utils/tools.py	工具函数：PSF→MTF 转换、MTF 2D 切片、下采样、PSF map 拼接、图像退化模拟等
generate_edges_rgb.py	数据准备步骤1：从棋盘格图像中提取模糊边缘 patch
generate_fov_weight.py	数据准备步骤3：将 MATLAB 输出的 SFR 曲线转换为训练用的 .npz 文件
checkerboard_rgb.py	棋盘格生成与处理：生成 checkerboard 图案、Harris 角点检测、按 FOV 裁剪边缘区域、模拟模糊
filters.py	各种图像滤波器：卷积、双边滤波、域变换滤波、高斯滤波、梯度计算
edgetaper.py	边缘渐变（edge taper）处理，用于减少 FFT 卷积的边界伪影
utils_fft.py	FFT 复数运算工具（复数乘法、除法、共轭、上/下采样等）
configs/63762BB.yaml	镜头配置文件（镜头名称、数据集路径、学习率、epoch 数等）
工作流程

原始图像（棋盘格）
    │
    ▼ ① generate_edges_rgb.py
模糊边缘 patch
    │
    ▼ ② MATLAB: user_sfrmat5_rgb.m（ISO 12233 标准 SFR 算法）
SFR/MTF 曲线 (.mat)
    │
    ▼ ③ generate_fov_weight.py
训练数据 (.npz，含 SFR、权重、FOV、旋转角)
    │
    ▼ ④ main.py (训练)
    │  PSF_mlp 网络：H → Seidel系数 → 波前 → PSF → MTF
    │  损失函数：预测MTF 与 真实SFR 的 L1 误差
    │
    ▼
结果：全视场 PSF map + GT vs 预测对比图
关键技术细节
物理先验：网络输出的是 Seidel 像差系数（9个），通过物理公式转为波前 → FFT → PSF，而非直接预测 PSF 像素，这保证了物理合理性
三通道独立处理：RGB 三个波长分别建模，最后通过 shift_net 校正横向色差（R/B 相对于 G 的空间偏移）
坐标缩放因子 F=5：在 MTF 2D 切片时放大坐标 5 倍以减少插值量化误差
输入数据：镜头设计 Zernike 系数存在 input/*.xlsx，SFR 训练数据在 dataset/，数据集中的 .tif 文件是按不同 FOV 和角度裁剪的模糊边缘 patch