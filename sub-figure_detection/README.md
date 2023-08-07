## Sub-figure detection

We use the YoloV7-Tiny model to perform the detection of the desired images and store the output of the model for later image interception processing. As training data, we used 1658 images annotated by Labelme and selected the lightweight Tiny model for training. 

### Inference：

`detect.py` runs inference on various data sources, and saves the results to the `runs/detect` directory.

```bash
$ python detect.py  --weights image_dec.pt #模型
                    --source  images/ # 图像路径
                    --device 0
                    --save-txt True #保存检测结果
```

### Predict result preview🚀
<div align="center">
<img  src=./picture/23451160_pone.0057128.g006.jpg width=90% />
</div>