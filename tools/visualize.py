from mmdet3d.apis import inference_detector, init_model
import numpy as np
import string, random
import os

os.environ["MODEL_CONFIG"] = (
    "~/Autocomm/mmdetection3d/configs/SOS_Lab/config.py"
)
os.environ["MODEL_CHECKPOINT"] = (
    "~/Autocomm/mmdetection3d/tools/model_weights.pth")

if __name__ == "__main__":
    config_file = os.environ["MODEL_CONFIG"]
    checkpoint_file = os.environ["MODEL_CHECKPOINT"]
    model = init_model(config_file, checkpoint_file, device="cuda:0")
    print(model)
