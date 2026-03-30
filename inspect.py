import json

with open('task_graph_0.json', 'r') as f:
    task_graph = json.load(f)

print(task_graph['all_tasks'][19])