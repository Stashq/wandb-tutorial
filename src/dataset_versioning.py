import wandb
from sklearn.datasets import load_iris


iris_ds = load_iris(as_frame=True)
df = iris_ds['data']
df['target'] = iris_ds['target']

n_rec = df.shape[0]
splits = {
    'train': (0, int(n_rec*0.6)),
    'val': (int(n_rec*0.6), int(n_rec*0.8)),
    'test': (int(n_rec*0.8), n_rec)
}

for name, (start, end) in splits.items():
    df.iloc[start:end].to_csv(f"data/{name}/iris_{name}.cvs")

run = wandb.init(job_type="dataset-creation", project="wandb tutorial")
my_data = wandb.Artifact("new_dataset", type="raw_data")

for dir_ in ["train", "val", "test"]:
    my_data.add_dir('data/' + dir_)

run.log_artifact(my_data)

wandb.finish()
