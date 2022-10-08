# Weights and Biases

## Init

First you have to login using terminal:

```console
wandb login
```

Insert in your code:

```python
# At the beggining
wandb.init(project='tutorial')
...
# At the ending
wandb.finish()
```

## Hyperparameters saving

You can save hyperparameters in init or in config:

```python
wandb.config.batch_size = 32
wandb.init(config={"epochs": 4})
# later, treating config as dict
wandb.config.update({"lr": 0.1, "channels": 16})
```

If you create a file called config-defaults.yaml, and it will automatically be loaded into wandb.config.

## Logging

You can log multiple metrics at same time:

```python
wandb.log({"loss": 0.314, "epoch": 5,
           "inputs": wandb.Image(inputs),
           "logits": wandb.Histogram(ouputs),
           "captions": wandb.Html(captions)})
```

In order to synchronize metrics you can provide step:

```python
wandb.log({'loss': 0.2}, step=step)
```

You can log media using wandb objects like: *Image, Video, Audio, Table, Object3d, ect.*.  
Images can be register as *Artifacts* and reused in other run (about *Artifacts* later):

```python
# First run
wandb.init()
art = wandb.Artifact("my_images", "dataset")
for path in IMAGE_PATHS:
    art.add(wandb.Image(path), path)
wandb.log_artifact(art)

# Second run
wandb.init()
art = wandb.use_artifact("my_images:latest")
img_1 = art.get(PATH)
wandb.log({"image": img_1})
```

Tables can be logged using wandb.*Table* or just pandas *DataFrame*.

### Metric Summarization

By default W&B takes last value of metric as its summarization. You can set other summarization like "min", "max" and few more:

```python
wandb.define_metric("loss", summary="min")
```

## Vizualizations

In UI you can create panels using collected data, images, comparing models performance with *parallel coordinates plot* and many more.  

*Weaves* are queries defining what data you want collect and how to present it:

```python
runs.summary["accuracy"]
```

*Embedding Projector* allows you to plot 2d embedding using PCA, t-SNE, UMAP only using UI.  

[NLP attention](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM) visualized. [Other useful examples](https://docs.wandb.ai/guides/data-vis/tables) like image classification, NER, ect.  

W&B charts are created with [Vega](https://vega.github.io/vega/). You can customize them in your code but most important you are able to create these in UI adding *custom chart*. On the right will be described query structure which using you change the plot.  

Tables can be transformed in UI (filtering, grouping and many more) using Weave expressions. You can group records by values from certain column clicking on it or filter those examples where predicted and true label were different:

```python
row["pred"] != row["true"]
```

After clicking on *Columns* (bottom of Table on right side) you can select which columns to show.

## API

You can go back to previous runs and take artifacts from them:

```python
api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

Taking many runs at once:

```python
runs = api.runs(entity + "/" + project)

# Filtering with MongoDB Query
runs = api.runs('<entity>/<project>', {
    "$and": [{
    'created_at': {
        "$lt": 'YYYY-MM-DDT##',
        "$gt": 'YYYY-MM-DDT##'
        }
    }]
})
```

Updating past runs, in this example renaming metric:

```python
run = api.run("<entity>/<project>/<run_id>")
run.summary['new_name'] = run.summary['old_name']
del run.summary['old_name']
run.summary.update()
```

Downloading file:

```python
run.file("model-best.h5").download()
```

## Sweeps

Sweeps to automate hyperparameter search and explore the space of possible models (similar to optuna). To run experiments with sweep:

1. define experiment function,
2. set hyperparameters config,
3. init sweep with them,
4. use agent to run sweep experiment.

In sweep setup you define strategy:

- grid - check every combination,
- random - choose parameters randomly,
- bayes - create a probabilistic model of a metric score as a function of the hyperparameters, and choose parameters with high probability of improving the metric.

It is possible to pause, resume, stop, and cancel sweep run via console.
Sweeps can be run in parallel with many agents (also on different machines).

## Reports

With wandb reports you can create amazing documents containing latext formula, experiments artifacts and many more. You can define sets of experiments and switch between them in report. It will adjust plots to these sets.  

## Versioning

WandB Artifacts allows you to store and version data and models. Example of logging dataset:

```python
artifact = wandb.Artifact('mnist', type='dataset')
artifact.add_dir('mnist/')
wandb.log_artifact(artifact)
```

## DL logging

Logging models and metrics:

```python
wandb_logger = WandbLogger(log_model="all")
trainer = Trainer(
    logger=wandb_logger, callbacks=[checkpoint_callback])
```

Logging onnx model representation:

```python
dummy_input = torch.zeros(input_shape, device=self.device)
file_name = f"path/to/model.onnx"
torch.onnx.export(self, dummy_input, file_name)
wandb.save(file_name)
```

To log gradient you have to use:

```python
wandb.watch(
    model,
    log="gradients",
    log_freq=1000
)
```

## Additional notes

To deal with `ModuleNotFoundError: No module named 'src'` add to `pyproject.toml`:

```toml
[tool.poetry]
...
packages = [
    { include = "src" },
]
```
