import abc
from typing import Mapping, Tuple, Dict, Sequence

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.base import merge_dicts

from transformers_from_scratch.core.modelling.model import Model
from transformers_from_scratch.core.modelling.structures import (
    BackboneInput,
    ModelOutput
)
from transformers_from_scratch.core.utilities.arguments import ArgparserExtender


class PLModule(LightningModule, ArgparserExtender):
    def __init__(self, model: Model):
        super().__init__()

        self.model = model

    def forward(self, backbone_inp: BackboneInput) -> ModelOutput:
        output = self.model(backbone_inp)
        return output

    def training_step(
            self,
            backbone_inp: BackboneInput,
            batch_idx: int
    ) -> Dict:
        loss, log = self._step(backbone_inp=backbone_inp)
        return {'loss': loss, 'log': log}

    def validation_step(
            self,
            backbone_inp: BackboneInput,
            batch_idx: int
    ) -> Dict:
        loss, log = self._step(backbone_inp=backbone_inp)
        return {'val_loss': loss, 'log': log}

    def validation_epoch_end(self, val_step_results: Sequence):
        validation_epoch_result = merge_dicts(
            dicts=val_step_results,
            default_func=lambda x: torch.stack(x).mean().item()
        )

        return validation_epoch_result

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(optimizer=optimizer)

        if scheduler is None:
            return optimizer
        else:
            return [optimizer], [scheduler]

    def _step(
            self,
            backbone_inp: BackboneInput
    ) -> Tuple[torch.Tensor, Mapping]:
        output = self.forward(backbone_inp=backbone_inp)
        log = self._get_step_log(model_output=output)
        return output.loss, log

    @abc.abstractmethod
    def _get_optimizer(self):
        pass

    @abc.abstractmethod
    def _get_lr_scheduler(self, optimizer):
        pass

    @abc.abstractmethod
    def _get_step_log(self, model_output: ModelOutput) -> Dict:
        pass

    @abc.abstractmethod
    def get_description(self) -> Dict:
        pass
