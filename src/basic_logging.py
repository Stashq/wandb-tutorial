import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import wandb

wandb.init(project="wandb tutorial", config={"lr": 0.01})


thresh = 0.9

wandb.log({"accuracy": np.random.randn(), "loss": np.random.randn()})
wandb.alert(
    title="Low accuracy",
    text=f"Accuracy {0.5} is below the acceptable threshold {thresh}",
)

# art = wandb.Artifact("my-object-detector", type="model")
# art.add_file("saved_model_weights.pt")
# wandb.log_artifact(art)

img = plt.imread("media/Fallen_Angel_(Alexandre_Cabanel).jpg")
table = wandb.Table(columns=["Text", "Predicted Label", "True Label"])
table.add_data("I love my phone", "1", "1")
table.add_data("My phone sucks", "0", "-1")
data = [["I love my phone", "1", "1"], ["My phone sucks", "0", "-1"]]
points = np.random.uniform(size=(250, 3))

df = pd.DataFrame(
    [["cat", 5.3], ["cat", 3.4], ["dog", 12.9]], columns=["animal", "weight"]
)

wandb.log(
    {
        "img": [wandb.Image(img, caption="Alexandre Cabanel")],
        "video": wandb.Video("media/kikuchiyo.gif", fps=32, format="gif"),
        "audio": wandb.Audio("media/joji.mp3", sample_rate=32, caption="Joji"),
        "table": table,
        "table2": wandb.Table(
            data=data, columns=["Text", "Predicted Label", "True Label"]
        ),
        "pandas_table": df,
        "point_scene": wandb.Object3D(
            {
                "type": "lidar/beta",
                "points": points,
                "boxes": np.array(
                    [
                        {
                            "corners": [
                                [0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0],
                                [1, 1, 0],
                                [0, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1],
                            ],
                            "label": "Box",
                            "color": [123, 321, 111],
                        },
                        {
                            "corners": [
                                [0, 0, 0],
                                [0, 2, 0],
                                [0, 0, 2],
                                [2, 0, 0],
                                [2, 2, 0],
                                [0, 2, 2],
                                [2, 0, 2],
                                [2, 2, 2],
                            ],
                            "label": "Box-2",
                            "color": [111, 321, 0],
                        },
                    ]
                ),
                "vectors": np.array([]),
            }
        ),
    }
)

wandb.finish()
