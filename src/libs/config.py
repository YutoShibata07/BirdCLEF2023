import dataclasses
from logging import getLogger
from pprint import pformat
from typing import Any, Dict, Tuple
import yaml

__all__ = ["get_config"]

logger = getLogger(__name__)


@dataclasses.dataclass(frozen=False)
class Config:
    model: str = "effi_b2"

    pretrained: bool = True

    dataset_name: str = "train_images_512"

    use_class_weight: bool = False

    batch_size: int = 32

    num_workers: int = 2

    early_stop_epoch: int = 10

    max_epoch: int = 30

    input_feature: str = "logmel"

    embed_dim: int = 128

    scheduler: str = "CosineAnnealingLR"

    lr: float = 1e-5

    lr_max: float = 4e-4

    lr_min: float = 1e-6

    do_mixup: bool = False

    augmentations: Tuple[str, ...] = ""

    training_year: str = "2023"

    loss_fn: str = "ce"

    aug_ver: int = 1

    duration: int = 5

    model_path: str = ""

    make_train_oof: bool = False

    cleaning_path: str = ""  # ''の場合はresult_dirのoofが使用される。 'False'の場合はcleaningなし。　別のresult_dirのoofを使いたい時はそのdir_nameを入力

    n_split: int = 5
    
    is_hard:bool = False

    def __post_init__(self) -> None:
        self._type_check()
        self._value_check()

        logger.info(
            "Experiment Configuration\n" + pformat(dataclasses.asdict(self), width=1)
        )

    def _value_check(self) -> None:
        if self.max_epoch <= 0:
            message = "max_epoch must be positive."
            logger.error(message)
            raise ValueError(message)

    def _type_check(self) -> None:
        """Reference:
        https://qiita.com/obithree/items/1c2b43ca94e4fbc3aa8d
        """

        _dict = dataclasses.asdict(self)

        for field, field_type in self.__annotations__.items():
            # if you use type annotation class provided by `typing`,
            # you should convert it to the type class used in python.
            # e.g.) Tuple[int] -> tuple
            # https://stackoverflow.com/questions/51171908/extracting-data-from-typing-types

            # check the instance is Tuple or not.
            # https://github.com/zalando/connexion/issues/739
            if hasattr(field_type, "__origin__"):
                # e.g.) Tuple[int].__args__[0] -> `int`
                element_type = field_type.__args__[0]

                # e.g.) Tuple[int].__origin__ -> `tuple`
                field_type = field_type.__origin__

                self._type_check_element(field, _dict[field], element_type)

            # bool is the subclass of int,
            # so need to use `type() is` instead of `isinstance`
            if type(_dict[field]) is not field_type:
                message = f"The type of '{field}' field is supposed to be {field_type}."
                logger.error(message)
                raise TypeError(message)

    def _type_check_element(
        self, field: str, vals: Tuple[Any], element_type: type
    ) -> None:
        for val in vals:
            if type(val) is not element_type:
                message = (
                    f"The element of '{field}' field is supposed to be {element_type}."
                )
                logger.error(message)
                raise TypeError(message)


def convert_list2tuble(_dict: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in _dict.items():
        if isinstance(val, list):
            _dict[key] = tuple(val)

    logger.debug("converted list to tuble in dictionary")
    return _dict


def get_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dict = convert_list2tuble(config_dict)
    config = Config(**config_dict)

    logger.info("successfully loaded configuration.")
    return config
