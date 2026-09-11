"""Learning-rate schedules registered with FunASR's training entry point."""
from torch.optim.lr_scheduler import LambdaLR
from funasr.schedulers import scheduler_classes


class WarmupPolynomialDecayLR(LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, power=1.0, end_lr=0.0, last_epoch=-1):
        if not 0 <= warmup_steps < total_steps or power <= 0:
            raise ValueError("Require 0 <= warmup_steps < total_steps and positive power.")
        factors = []
        for group in optimizer.param_groups:
            base_lr = group.get("initial_lr", group["lr"])
            if not 0 <= end_lr <= base_lr or base_lr <= 0:
                raise ValueError("Require 0 <= end_lr <= initial learning rate.")
            end_factor = end_lr / base_lr

            def factor(step, end_factor=end_factor):
                if step < warmup_steps:
                    return (step + 1) / warmup_steps
                progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
                return end_factor + (1 - end_factor) * (1 - progress) ** power

            factors.append(factor)
        super().__init__(optimizer, factors, last_epoch=last_epoch)


scheduler_classes["WarmupPolynomialDecayLR"] = WarmupPolynomialDecayLR
