简体中文 | [English](README_en.md)

## 使用说明

仅训练时必须使用半监督检测的配置文件去训练，评估、预测、部署也可以按基础检测器的配置文件去执行。
本代码为测试版本，最终版本请见
https://github.com/PaddlePaddle/PaddleDetection/tree/develop

### 训练

```bash
# 单卡训练 (不推荐，需按线性比例相应地调整学习率)
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/semi_det/ppyoloe/ssod_ppyoloe_plus_crn_s_coco_semi010_load.yml --eval

# 多卡训练
python -m paddle.distributed.launch --log_dir=denseteacher_fcos_semi010/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/semi_det/ppyoloe/ssod_ppyoloe_plus_crn_s_coco_semi010_load.yml --eval
```

### 评估

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c -c configs/semi_det/ppyoloe/ssod_ppyoloe_plus_crn_s_coco_semi010_load.yml -o weights=output/ssod_ppyoloe_plus_crn_s_coco_semi010_loadmodel_final.pdparams
```

### 预测

```bash
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/semi_det/ppyoloe/ssod_ppyoloe_plus_crn_s_coco_semi010_load.yml-o weights=output/ssod_ppyoloe_plus_crn_s_coco_semi010_load/model_final.pdparams --infer_img=demo/000000014439.jpg
```
## 半监督数据集准备

半监督目标检测**同时需要有标注数据和无标注数据**，且无标注数据量一般**远多于有标注数据量**。
对于COCO数据集一般有两种常规设置：

（1）抽取部分比例的原始训练集`train2017`作为标注数据和无标注数据；

从`train2017`中按固定百分比（1%、2%、5%、10%等）抽取，由于抽取方法会对半监督训练的结果影响较大，所以采用五折交叉验证来评估。运行数据集划分制作的脚本如下：
```bash
python tools/gen_semi_coco.py
```
会按照 1%、2%、5%、10% 的监督数据比例来划分`train2017`全集，为了交叉验证每一种划分会随机重复5次，生成的半监督标注文件如下：
- 标注数据集标注：`instances_train2017.{fold}@{percent}.json`
- 无标注数据集标注：`instances_train2017.{fold}@{percent}-unlabeled.json`
其中，`fold` 表示交叉验证，`percent` 表示有标注数据的百分比。

注意如果根据`txt_file`生成，需要下载`COCO_supervision.txt`:
```shell
wget https://bj.bcebos.com/v1/paddledet/data/coco/COCO_supervision.txt
```

（2）使用全量原始训练集`train2017`作为有标注数据 和 全量原始无标签图片集`unlabeled2017`作为无标注数据；


### 下载链接

PaddleDetection团队提供了COCO数据集全部的标注文件，请下载并解压存放至对应目录:

```shell
# 下载COCO全量数据集图片和标注
# 包括 train2017, val2017, annotations
wget https://bj.bcebos.com/v1/paddledet/data/coco.tar

# 下载PaddleDetection团队整理的COCO部分比例数据的标注文件
wget https://bj.bcebos.com/v1/paddledet/data/coco/semi_annotations.zip

# unlabeled2017是可选，如果不需要训‘full’则无需下载
# 下载COCO全量 unlabeled 无标注数据集
wget https://bj.bcebos.com/v1/paddledet/data/coco/unlabeled2017.zip
wget https://bj.bcebos.com/v1/paddledet/data/coco/image_info_unlabeled2017.zip
# 下载转换完的 unlabeled2017 无标注json文件
wget https://bj.bcebos.com/v1/paddledet/data/coco/instances_unlabeled2017.zip
```