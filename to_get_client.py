from mlflow.tracking import MlflowClient

import mlflow

client = MlflowClient()

experiments = client.list_experiments() # returns a list of mlflow.entities.Experiment
run = client.create_run(experiments[1].experiment_id) 
"""print(run.info)
print('\n\n')
print(run.data)
client.log_param(run.info.run_id, "hello", "world")
print('\n\n')
print('\n\n')
print(run.info)
print('\n\n')
print(run.data)
client.set_terminated(run.info.run_id)"""
print(mlflow.projects.run())