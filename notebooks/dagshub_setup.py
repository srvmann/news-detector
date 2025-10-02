import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/srvmann/news-detector.mlflow")
dagshub.init(repo_owner='srvmann', repo_name='news-detector', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)


