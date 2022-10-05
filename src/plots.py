import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

import wandb

wandb.init(project="wandb tutorial")

# Wandb scatter
class_x_scores = np.linspace(0.1, 1.5, 30)
class_y_scores = 0.5 + np.random.randn(30)
data = [[x, y] for (x, y) in zip(class_x_scores, class_y_scores)]
table = wandb.Table(data=data, columns=["class_x", "class_y"])
wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})

# Matplotlib
plt.plot([1, 2, 3, 4])
plt.ylabel("some interesting numbers")
wandb.log({"chart": plt})

# Plotly
# Initialize a new run
run = wandb.init(project="log-plotly-fig-tables", name="plotly_html")

# Create a table
table = wandb.Table(columns=["plotly_figure"])

# Create path for Plotly figure
path_to_plotly_html = "plots/plotly_figure.html"

# Example Plotly figure
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# Write Plotly figure to HTML
fig.write_html(
    path_to_plotly_html, auto_play=False
)  # Setting auto_play to False prevents animated Plotly
# charts from playing in the table automatically

# Add Plotly figure as HTML file into Table
table.add_data(wandb.Html(path_to_plotly_html))

# Log Table
run.log({"test_table": table})
wandb.finish()
