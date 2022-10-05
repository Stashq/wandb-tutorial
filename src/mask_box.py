import matplotlib.pyplot as plt
import numpy as np

import wandb

wandb.init(project="wandb tutorial", name="image with mask")

img = plt.imread("media/Fallen_Angel_(Alexandre_Cabanel).jpg")
pred_mask = np.random.choice([1, 2, 3], img.shape[:2])
true_mask = np.random.choice([1, 2, 3], img.shape[:2])

class_labels = {1: "tree", 2: "car", 3: "road"}


mask_img = wandb.Image(
    img,
    masks={
        "predictions": {"mask_data": pred_mask, "class_labels": class_labels},
        "ground_truth": {"mask_data": true_mask, "class_labels": class_labels},
    },
)

wandb.log({"masked image": mask_img})

table = wandb.Table(columns=["ID", "Image"])
boxes = [
    {
        "minX": 500,
        "maxX": 900,
        "minY": 300,
        "maxY": 600,
        "class_id": 1,
        "caption": "Meow",
    },
    {
        "minX": 300,
        "maxX": 450,
        "minY": 200,
        "maxY": 900,
        "class_id": 2,
        "caption": "Woof!",
    },
]

class_labels2 = {1: "cat", 2: "dog"}

box_img = wandb.Image(
    img,
    boxes={
        "prediction": {
            "box_data": [
                {
                    "position": {
                        "minX": box["minX"],
                        "minY": box["minY"],
                        "maxX": box["maxX"],
                        "maxY": box["maxY"],
                    },
                    "class_id": box["class_id"],
                    "box_caption": box["caption"],
                    "domain": "pixel",
                }
                for box in boxes
            ],
            "class_labels": class_labels2,
        }
    },
)

wandb.log({"boxes": box_img})

wandb.finish()
