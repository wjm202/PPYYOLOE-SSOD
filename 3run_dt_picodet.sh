export FLAGS_allocator_strategy=auto_growth
model_type=semi_det/denseteacher
job_name=denseteacher_picodet_s_416_coco_lcnet_semi010_load
config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
weights=output/denseteacher_picodet_s_416_coco_lcnet_semi010_load/.pdparams
#weights=~/.cache/paddle/weights/picodet_s_416_coco_lcnet_10.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/train.py -c ${config} # -r ${weights}
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} #--eval

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/eval.py -c ${config} -o weights=${weights}

# 3. tools infer
#CUDA_VISIBLE_DEVICES=7 python3.7 tools/infer_mot.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.3
#CUDA_VISIBLE_DEVICES=4 python3.7 tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.3

# 4.导出模型
#CUDA_VISIBLE_DEVICES=1 python3.7 tools/export_model.py -c ${config} -o weights=${weights} #exclude_nms=True trt=True

# 5.部署预测
#CUDA_VISIBLE_DEVICES=1 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU
