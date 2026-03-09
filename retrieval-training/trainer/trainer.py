import os
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from typing import Optional



class Trainer:
    def __init__(self, model, optimizer, loss_fn, scheduler, save_dir, model_name, logger):
        self.accelerator = Accelerator()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.model_name = model_name
        self.logger = logger
        os.makedirs(self.save_dir, exist_ok=True)

    def fit(self, train_loader, epochs: int, val_loader: Optional[DataLoader] = None):
        self.model, self.optimizer, self.scheduler, train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, train_loader
        )
        if val_loader is not None:
            val_loader = self.accelerator.prepare(val_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["val_loss"]
            val_alpha = val_metrics["val_alpha"]
            
            if self.accelerator.is_main_process:
                self.logger.info(f"Initial evaluation: "
                            f"val_loss={val_loss:.4f} val_alpha={val_alpha:.2f}")

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            steps = 0
            last_100_losses = []  # Track last 100 losses
            last_100_alphas = []  # Track last 100 alphas

            if self.accelerator.is_main_process:
                pbar = tqdm(
                    total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{epochs}",
                    dynamic_ncols=True,
                    ncols=0,
                    position=0,
                    leave=True,
                )

            for batch in train_loader:
                sim, alpha = self.model(batch)
                loss = self.loss_fn(sim, batch["desirability_label"], batch["relevance_label"])

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                loss = loss.detach()
                running_loss += loss
                steps += 1
                
                # Track last 100 losses
                last_100_losses.append(loss.item())
                last_100_alphas.append(alpha.detach().cpu().mean().item())
                if len(last_100_losses) > 100:
                    last_100_losses.pop(0)
                    last_100_alphas.pop(0)

                if self.accelerator.is_main_process:
                    lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}", "alpha": f"{alpha.mean().item():.2f}"})
                    pbar.update(1)
                    
                    if steps == 1 or steps % 100 == 0:
                        avg_100_loss = sum(last_100_losses) / len(last_100_losses)
                        avg_100_alpha = sum(last_100_alphas) / len(last_100_alphas)
                        self.logger.info(f"[Epoch {epoch+1}, Step {steps}] avg_loss_last_100_steps={avg_100_loss:.4f} alpha={avg_100_alpha:.2f}")

            avg_loss = self.accelerator.reduce(running_loss / steps, reduction="mean")
            self.logger.info(f"[Epoch {epoch}] avg_train_loss={avg_loss.item():.4f}")
            if self.accelerator.is_main_process:
                pbar.close()
                unwrapped = self.accelerator.unwrap_model(self.model)
                save_path = os.path.join(self.save_dir, f"{self.model_name}_{epoch+1}.pt")
                torch.save(unwrapped.state_dict(), save_path)
                self.logger.info(f"Saved model checkpoint to {save_path}")

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics["val_loss"]
                val_alpha = val_metrics["val_alpha"]
                
                if self.accelerator.is_main_process:
                    self.logger.info(f"[Epoch {epoch+1}] train_loss={avg_loss.item():.4f} | "
                                     f"val_loss={val_loss:.4f} val_alpha={val_alpha:.2f}")

    def evaluate(self, eval_loader):
        self.logger.info("Starting evaluation...")
        self.model.eval()
        running_loss = 0.0
        running_alpha = 0.0
        steps = 0
        
        if self.accelerator.is_main_process:
            pbar = tqdm(
                total=len(eval_loader),
                desc="Evaluating",
                dynamic_ncols=True,
                ncols=0,
                position=0,
                leave=True,
            )
        
        with torch.no_grad():
            for batch in eval_loader:
                sim, alpha = self.model(batch)
                
                loss = self.loss_fn(sim, batch["desirability_label"], batch["relevance_label"])
                
                running_loss += loss.detach()
                running_alpha += alpha.detach().cpu().mean().item()
                steps += 1
                
                if self.accelerator.is_main_process:
                    pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
                    pbar.update(1)
        
        if self.accelerator.is_main_process:
            pbar.close()
        
        avg_loss = self.accelerator.reduce(running_loss / steps, reduction="mean")
        avg_alpha = running_alpha / steps
        
        metrics = {
            "val_loss": avg_loss.item(),
            "val_alpha": avg_alpha
        }
        
        return metrics