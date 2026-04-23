import mlflow


def log_params_recursive(d):
    for k, v in d.items():
        if isinstance(v, dict):
            log_params_recursive(v)
        elif isinstance(v, (int, float, str, bool)):
            mlflow.log_param(k, v)
