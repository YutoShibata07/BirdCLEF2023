from logging import getLogger

import pandas as pd

logger = getLogger(__name__)


class TrainLogger(object):
    def __init__(self, log_path: str, resume: bool) -> None:
        self.log_path = log_path
        self.columns = [
            "epoch",
            "lr",
            "train_time[sec]",
            "train_loss",
            "train_score",
            "val_time[sec]",
            "val_loss",
            "val_score",
        ]

        if resume:
            self.df = self._load_log()
        else:
            self.df = pd.DataFrame(columns=self.columns)

    def _load_log(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.log_path)
            logger.info("successfully loaded log csv file.")
            return df
        except FileNotFoundError as err:
            logger.exception(f"{err}")
            raise err

    def _save_log(self) -> None:
        self.df.to_csv(self.log_path, index=False)
        logger.debug("training logs are saved.")

    def update(
        self,
        epoch: int,
        lr: float,
        train_time: int,
        train_loss: float,
        train_score: float,
        val_time: int,
        val_loss: float,
        val_score: float,
    ) -> None:
        tmp = pd.Series(
            [
                epoch,
                lr,
                train_time,
                train_loss,
                train_score,
                val_time,
                val_loss,
                val_score,
            ],
            index=self.columns,
        )

        self.df = self.df.append(tmp, ignore_index=True)
        self._save_log()
        logger.info(
            f"epoch: {epoch}\tepoch time[sec]: {train_time + val_time}\tlr: {lr}\t"
            f"train loss: {train_loss:.4f}\ttrain_score:{train_score:.5f}\tval loss: {val_loss:.4f}\t"
            f"\tval_score:{val_score:.5f}"
        )


class ValidLogger(object):
    def __init__(self, log_path: str, resume: bool) -> None:
        self.log_path = log_path
        self.columns = [
            "train_acc",
            "train_auc",
            "val_time[sec]",
            "val_loss",
            "val_acc",
            "val_auc",
        ]

        if resume:
            self.df = self._load_log()
        else:
            self.df = pd.DataFrame(columns=self.columns)

    def _load_log(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.log_path)
            logger.info("successfully loaded log csv file.")
            return df
        except FileNotFoundError as err:
            logger.exception(f"{err}")
            raise err

    def _save_log(self) -> None:
        self.df.to_csv(self.log_path, index=False)
        logger.debug("training logs are saved.")

    def update(
        self,
        epoch: int,
        lr: float,
        train_time: int,
        train_loss: float,
        train_acc: float,
        train_auc: float,
        val_time: int,
        val_loss: float,
        val_acc: float,
        val_auc: float,
    ) -> None:
        tmp = pd.Series(
            [
                epoch,
                lr,
                train_time,
                train_loss,
                train_acc,
                train_auc,
                val_time,
                val_loss,
                val_acc,
                val_auc,
            ],
            index=self.columns,
        )

        self.df = self.df.append(tmp, ignore_index=True)
        self._save_log()
        logger.info(
            f"epoch: {epoch}\tepoch time[sec]: {train_time + val_time}\tlr: {lr}\t"
            f"train loss: {train_loss:.4f}\tval loss: {val_loss:.4f}\t"
            f"val_acc: {val_acc:.5f}\tval_auc: {val_auc:.5f}"
        )
