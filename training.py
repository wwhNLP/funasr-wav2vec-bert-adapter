"""FunASR training boundary for summed fairseq2 SSL objectives."""
import logging
import torch
from funasr.train_utils.trainer_ds import Trainer
from funasr.train_utils.device_funcs import to_device


class PretrainingTrainer(Trainer):
    """Normalize accumulated gradients by masked targets, not microbatch count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_deepspeed or self.use_fsdp:
            raise ValueError("Target-normalized pretraining currently supports native PyTorch/DDP; DeepSpeed/FSDP need a sharded gradient normalization adapter.")
        if self.save_checkpoint_interval % self.accum_grad:
            raise ValueError("save_checkpoint_interval must be divisible by accum_grad.")
        self._target_count = None

    def train_epoch(self, *args, **kwargs):
        loader = kwargs.get("dataloader_train")
        if loader is not None and len(loader) % self.accum_grad:
            raise ValueError("The remaining epoch batch count must be divisible by accum_grad.")
        if kwargs.get("start_step", 0) % self.accum_grad:
            raise ValueError("Resume from an optimizer-step boundary (start_step divisible by accum_grad).")
        return super().train_epoch(*args, **kwargs)

    def forward_step(self, model, batch, loss_dict=None):
        super().forward_step(model, batch, loss_dict)
        count = loss_dict["stats"]["num_targets"]
        loss_dict["loss_sum"] = loss_dict["loss"]
        loss_dict["num_targets"] = count
        loss_dict["loss"] = loss_dict["loss"] / count
        loss_dict["weight"] = count
        loss_dict["stats"] = {key: value if key == "num_targets" else value / count
                              for key, value in loss_dict["stats"].items()}

    def backward_step(self, model, scaler, loss_dict=None):
        count = loss_dict["num_targets"].detach()
        self._target_count = count.clone() if self._target_count is None else self._target_count + count
        loss = loss_dict["loss_sum"]
        (scaler.scale(loss) if scaler is not None else loss).backward()

    def update_step(self, model, optim, scheduler, scaler, loss_dict=None):
        if (loss_dict["batch_idx"] + 1) % self.accum_grad:
            return
        count = self._target_count
        self._target_count = None
        if self.use_ddp:
            torch.distributed.all_reduce(count)
        # DDP already averages gradients across ranks; undo that factor.
        factor = (self.world_size if self.use_ddp else 1) / count
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)
        if scaler is not None:
            scaler.unscale_(optim)
        if self.grad_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip, self.grad_clip_type)
            if not torch.isfinite(norm):
                logging.warning("Non-finite gradient norm; skipping optimizer update.")
                if scaler is not None:
                    scaler.update()
                optim.zero_grad(set_to_none=True)
                return
        if scaler is not None:
            previous_scale = scaler.get_scale()
            scaler.step(optim)
            scaler.update()
            updated = scaler.get_scale() >= previous_scale
        else:
            optim.step()
            updated = True
        if updated:
            scheduler.step()
        optim.zero_grad(set_to_none=True)

    @torch.no_grad()
    def validate_epoch(self, model=None, dataloader_val=None, epoch=None, writer=None, **kwargs):
        training = model.training
        model.eval()
        totals = torch.zeros(2, device=self.device, dtype=torch.float64)
        dataloader_val.batch_sampler.set_epoch(epoch)
        try:
            for index, batch in enumerate(dataloader_val):
                values = dict(epoch=epoch, batch_idx=index, batch_total=self.batch_total,
                              batch_num_epoch=len(dataloader_val), lr=0.0)
                batch = to_device(batch, self.device, non_blocking=True)
                self.forward_step(model, batch, values)
                totals[0] += values["loss_sum"].detach().double()
                totals[1] += values["num_targets"].double()
                self.log(values, tag="val")
            if self.use_ddp:
                torch.distributed.all_reduce(totals)
            if totals[1] <= 0 or not torch.isfinite(totals).all():
                raise RuntimeError("Validation has no finite masked-target loss.")
            self.val_loss_avg = (totals[0] / totals[1]).item()
            self.val_acc_avg = float("nan")
            suffix = f".{kwargs['step_in_epoch']}" if kwargs.get("step_in_epoch") is not None else ""
            name = f"model.pt.ep{epoch}{suffix}"
            self.val_loss_step_or_epoch[name] = self.val_loss_avg
            self.val_acc_step_or_epoch.pop(name, None)
        finally:
            model.train(training)

    def log(self, loss_dict=None, tag="train", **kwargs):
        if (loss_dict["batch_idx"] + 1) % self.log_interval:
            return
        loss = loss_dict["loss"].detach().item()
        logging.info("%s rank=%s epoch=%s step=%s/%s loss_per_target=%.6f lr=%.3e",
                     tag, self.rank, loss_dict["epoch"], loss_dict["batch_idx"] + 1,
                     loss_dict["batch_num_epoch"], loss, loss_dict["lr"])
        # An SSL objective including feature penalties is not a language-model perplexity.
        if self.writer is not None:
            step = loss_dict["batch_total"]
            self.writer.add_scalar(f"rank{self.rank}_loss/{tag}", loss, step)
            self.writer.add_scalar(f"rank{self.rank}_lr/{tag}", loss_dict["lr"], step)
            for key, value in loss_dict["stats"].items():
                self.writer.add_scalar(f"stats_rank{self.rank}_{key}/{tag}", value.item(), step)
