import logging
from dataclasses import dataclass, field
from typing import List, Dict

import mlflow

from src.custom_logs.create_logger import get_logger


@dataclass
class ExperimentLog:
    config: Dict = field(init=True)
    time_mins: float = field(init=False)
    running_epochs: int = field(init=False)
    dataset: str

    train_loss: List[float] = field(init=False)
    val_loss: List[float] = field(init=False)
    val_acc: List[float] = field(init=False)
    val_f1: List[float] = field(init=False)

    final_test_acc: float = field(init=False)
    final_test_f1: float = field(init=False)


    def __post_init__(self):
        self.logger, self.logger_name, self.logger_path = get_logger(self.config, self.dataset)

    def info(self, *args, **kwargs):
        return self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        return self.logger.warning(*args, **kwargs)

    def log_all(self):
        mlflow.end_run()
        project_name = "fake_news"
        mlflow.set_experiment(experiment_name=project_name)

        with mlflow.start_run():
            # mlflow.set_tag("version", model_version)

            for k, v in self.config.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("time_mins", self.time_mins)
            mlflow.log_param("running_epochs", self.running_epochs)

            for i, loss in enumerate(self.train_loss, 1):
                mlflow.log_metric("train_loss", loss, step=i)
            for i, loss in enumerate(self.val_loss, 1):
                mlflow.log_metric("val_loss", loss, step=i)
            for i, acc in enumerate(self.val_acc, 1):
                mlflow.log_metric("val_acc", acc, step=i)
            for i, f1 in enumerate(self.val_f1, 1):
                mlflow.log_metric("val_f1", f1, step=i)

            mlflow.log_metric("final_test_acc", self.final_test_acc)
            mlflow.log_metric("final_test_f1", self.final_test_f1)

            # full_path = os.path.join(os.getcwd(), self.logger_path)
            # try:
            #     mlflow.log_artifact(full_path)
            #     os.remove(self.logger_path)
            # except PermissionError:
            #     mlflow.log_param("logger_path", self.logger_path)
            #     print(f"can not write at {full_path}")

        self.logger.info("logged to MLFlow!")