import wandb
from sklearn.datasets import load_digits

wandb.init(job_type='collect_data', project="wandb tutorial")

# Load the dataset
ds = load_digits(as_frame=True)
df = ds.data

# Create a "target" column
df["target"] = ds.target.astype(str)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

# Create an "image" column
df["image"] = df.apply(
    lambda row: wandb.Image(row[1:].values.reshape(8, 8) / 16.0), axis=1)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

wandb.log({"digits": df})
wandb.finish()
