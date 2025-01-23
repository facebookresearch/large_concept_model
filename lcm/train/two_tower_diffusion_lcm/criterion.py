# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn.functional as F
from fairseq2.logging import get_log_writer
from fairseq2.nn.padding import pad_seqs
from torch import Tensor

from lcm.datasets.batch import EmbeddingsBatch, LCMInput, LCMStyle
from lcm.models.two_tower_diffusion_lcm.builder import TwoTowerDiffusionLCModel
from lcm.train.criterion import CriterionsFactory
from lcm.train.lcm.criterion import (
    LCMCriterion,
    LCMCriterionConfig,
    compute_standard_mse,
)
from lcm.train.metrics import LossTerm, format_as_float, register_metric_formatter
from lcm.train.step_sampler import StepsSampler, StepsSamplerConfig

logger = get_log_writer(__name__)


@dataclass
class TowerDiffusionLCMCriterionConfig(LCMCriterionConfig):
    cf_guidance_probability: float = 0.0
    """Probability to use classifier-free guidance by dropping conditioning.
       Note that this requires the model to be set with
       `trained_with_cf_guidance = True`!
    """
    step_sampling: StepsSamplerConfig = field(
        default_factory=lambda: StepsSamplerConfig()
    )

    log_losses_per_timestep_bucket: bool = False


@CriterionsFactory.register("two_tower_diffusion_next_sent")
class TwoTowerDiffusionCriterion(LCMCriterion):
    """Computes the LCM training objective for next-sentence prediction with diffusion"""

    config: TowerDiffusionLCMCriterionConfig
    model: TwoTowerDiffusionLCModel

    def __init__(
        self,
        config: TowerDiffusionLCMCriterionConfig,
        model: TwoTowerDiffusionLCModel,
        style: LCMStyle = LCMStyle.UNSUPERVISED,
    ):
        super().__init__(config, model, style)
        assert hasattr(self.base_model, "noise_scheduler"), (
            "Expecting the diffusion model to have a `noise_scheduler`"
        )
        self.noise_scheduler = self.base_model.noise_scheduler

        self.prediction_type = self.noise_scheduler.prediction_type

        self.trained_with_cf_guidance = self.base_model.config.trained_with_cf_guidance

        self.cf_guidance_probability = config.cf_guidance_probability

        assert (
            bool(self.cf_guidance_probability > 0) == self.trained_with_cf_guidance
        ), (
            "Expecting the config's cf_guidance_probabilitya to align with the model's `trained_with_cf_guidance` ",
            f"Found cf_guidance_probability={config.cf_guidance_probability} and "
            f"trained_with_cf_guidance={self.trained_with_cf_guidance}",
        )

        assert self.normalize_in_criterion, (
            "We only support `normalize_in_criterion = True` in the diffusion criterions"
        )

        self.summands.append("unnormalized_reconstruction_loss")

        if self.config.log_losses_per_timestep_bucket:
            # customize if needed
            self.step_bucketing_boundaries = torch.linspace(
                0, self.noise_scheduler.num_diffusion_train_steps, 11
            )
            self.step_bucketing_labels: List[str] = []
            for e in range(len(self.step_bucketing_boundaries) - 1):
                bucket_left = self.step_bucketing_boundaries[e]
                bucket_right = self.step_bucketing_boundaries[e + 1]
                self.step_bucketing_labels.append(
                    f"reconstruction_loss_t{bucket_left:.0f}-{bucket_right:.0f}"
                )

            self.summands.extend(self.step_bucketing_labels)
            for label in self.step_bucketing_labels:
                register_metric_formatter(
                    label, label, 1000, format_as_float, overwrite=True
                )

        # Step sampler + loss weighter
        self.step_sampler = StepsSampler(
            config.step_sampling,
            noise_scheduler=self.noise_scheduler,
        )

    def prepare_input_and_mask(
        self,
        batch: LCMInput,
    ) -> Tuple[EmbeddingsBatch, EmbeddingsBatch, torch.Tensor]:
        """
        A method for preparing model inputs and mask for a batch.
        It will be typically reused by the `__call__`
        implementations of the subclasses.
        Returns:
            - input_batch: context
            - target_batch: denoiser input
            - target_mask  mask of positions to compute the loss over

        """
        # Prepare the input as in MSE LCM: each sequence is (src, tgt)
        input_embeddings = batch.prepare_input(style=self.style)

        # Normalize the embeddings
        if self.normalize_in_criterion:
            input_embeddings = input_embeddings.normalize_seqs(self.sonar_normalizer)

        target_mask = torch.ones(
            size=input_embeddings.seqs.shape[:-1],
            dtype=torch.bool,
            device=input_embeddings.seqs.device,
        )

        # Factor in padded positions:
        if input_embeddings.padding_mask is not None:
            target_mask &= input_embeddings.padding_mask.materialize()

        return input_embeddings, input_embeddings.clone(), target_mask

    def sample_noisy_input_and_targets(self, input_batch, target_mask):
        """
        (1)
        Prepares the noised inputs (latents) by sampling diffusion timesteps and calling
        on the model's noise_scheduler to add noise accordingly
        (2) Given the scheduler prediction type, prepares the target that the model will be
        trained to predict.

        :param input_bach: EmbeddingsBatch of the ground truth embeddings with seqs in (B, T, C)
        :param target_mask: Bool tensor in (B, T) where `True` signals that the
                            model will be asked to predict the position
        """
        input_seqs, padding_mask = input_batch.seqs, input_batch.padding_mask

        timesteps = self.step_sampler.sample(
            size=input_seqs[..., 0].size(), device=input_seqs.device
        )

        # Sample noise
        noise_seqs = torch.randn_like(input_seqs)

        # Define target in (B*T, C)
        sonar_dim = input_seqs.size(-1)
        if self.prediction_type == "sample":
            """Predict the clean ground truth embeddings. Default mode"""
            target = input_seqs.view(-1, sonar_dim)

        elif self.prediction_type == "epsilon":
            """Predict the added noise"""
            target = noise_seqs.view(-1, sonar_dim)

        elif self.prediction_type == "v_prediction":
            """Predict an interpolation of the ground truth clean
            embeddings and the added noise.
            As introduced in https://arxiv.org/pdf/2305.08891
            """
            target = self.noise_scheduler.get_velocity(
                input_seqs.view(-1, sonar_dim),
                noise_seqs.view(-1, sonar_dim),
                timesteps.view(-1),
            ).clone()
        else:
            raise ValueError(
                "Prediction type should be either: sample, epsilon, v_prediction"
            )

        # Add noise
        # Reshape inputs and noise into in (B*T , C) -> add noise -> reshape back as (B, T, C)
        noisy_input_seqs = self.noise_scheduler.add_noise(
            input_seqs.view(-1, sonar_dim),
            noise_seqs.view(-1, sonar_dim),
            timesteps.view(-1),
        ).view(input_seqs.size())

        # Create sequence batch with diffusion timesteps
        noisy_input_batch = EmbeddingsBatch(
            noisy_input_seqs,
            padding_mask,
            diffusion_timesteps=timesteps,
        )
        return noisy_input_batch, target, target_mask

    def compute_loss(
        self, flattened_predictions, flattened_target
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters:
            flattened_predictions (Tensor): The predictions in (N, C)
            flattened_target (Tensor): The targets in (N, C)

        Returns:
            reconstruction_loss (Tensor): The Reconstruction loss we want to optimize (RMSE, SmoothL1, Huber etc.).
            plain_reconstruction_loss (Tensor): plain RMSE loss.
            unnormalized_reconstruction_loss (Tensor): plain RMSE loss between unnormalized features.
        """
        reconstruction_loss, plain_reconstruction_loss = compute_standard_mse(
            flattened_predictions,
            flattened_target,
        )

        unnormalized_reconstruction_loss, _ = compute_standard_mse(
            flattened_predictions,
            flattened_target,
            normalizer=self.sonar_normalizer,
        )
        # For backward compatibility with ongoing runs, take the sqrt
        if self.config.compute_rmse:
            epsilon = 1e-5
            reconstruction_loss = torch.sqrt(reconstruction_loss + epsilon)
            plain_reconstruction_loss = torch.sqrt(plain_reconstruction_loss + epsilon)
            unnormalized_reconstruction_loss = torch.sqrt(
                unnormalized_reconstruction_loss + epsilon
            )

        return (
            reconstruction_loss,
            plain_reconstruction_loss,
            unnormalized_reconstruction_loss,
        )

    @torch.no_grad()
    def _log_losses_per_step(self, batch_steps, reconstruction_loss):
        # Aggregate loss terms based on their bucket of diffusion steps for tracking
        summands = {}
        if self.config.log_losses_per_timestep_bucket:
            # Reconstruction_loss in BT,
            # batch_steps in BT,
            bucket_index = torch.bucketize(
                batch_steps, self.step_bucketing_boundaries.to(batch_steps.device)
            )
            onehot = F.one_hot(
                bucket_index,
                num_classes=self.step_bucketing_boundaries.numel(),
            )
            loss_per_step = torch.matmul(onehot.t().float(), reconstruction_loss)
            count_steps = onehot.sum(dim=0) + 1e-6
            if self.reduction == "mean":
                loss_per_step /= count_steps

            for e, label in enumerate(self.step_bucketing_labels):
                summands[label] = (
                    loss_per_step[e].item(),
                    count_steps[e].long().item(),
                )

        return summands

    def __call__(self, batch: LCMInput) -> LossTerm:
        """
        Input batch is LCMInput with:
            source: List[Tensor]
            target: Union[None, List[Tensor]]
        """

        # Prepare the clean inputs and target mask:
        input_batch, target_batch, target_mask = self.prepare_input_and_mask(batch)

        noisy_target_batch, target, target_mask = self.sample_noisy_input_and_targets(
            target_batch, target_mask
        )
        # Encode the context and diffuse:
        output_batch = self.model(
            input_batch,
            noisy_target_batch,
            cf_guidance_prob=self.cf_guidance_probability,
        )

        # Shape B, T, C
        output_seqs = output_batch.seqs

        sonar_dim = output_seqs.size(-1)

        # only measure distance over `target_mask = True` positions
        target_mask = target_mask.reshape(-1)

        # The target is basically the doubled ground truth sequence before noising
        # (with some modification to adjust for the denoiser's prediction type)

        # contextualized latents (noised inputs preceding the target) e_1, e_2, ...
        flattened_predictions = output_seqs.view(-1, sonar_dim)[target_mask]

        # x1, x2, ..., xT
        # Target is already in B*T, C
        flattened_target = target[target_mask]

        # Cast features to float32 before computing the loss:
        (
            reconstruction_loss,
            mse_loss,
            unnormalized_reconstruction_loss,
        ) = self.compute_loss(flattened_predictions.float(), flattened_target.float())

        num_target_elements = target_mask.sum()

        batch_steps = noisy_target_batch.diffusion_timesteps.view(-1)[target_mask]

        summands = self._log_losses_per_step(batch_steps, reconstruction_loss)

        # Get loss scales per timestep (gamma)
        gammas = self.step_sampler.get_loss_scales(batch_steps)
        # Weight the loss terms
        if gammas is not None:
            reconstruction_loss = torch.mul(reconstruction_loss, gammas)

        if self.reduction == "sum" or num_target_elements == 0:
            reduced_reconstruction_loss = reconstruction_loss.sum()
            mse_loss = mse_loss.sum()
            unnormalized_reconstruction_loss = unnormalized_reconstruction_loss.sum()

        elif self.reduction == "mean":
            reduced_reconstruction_loss = reconstruction_loss.mean()
            mse_loss = mse_loss.mean()
            unnormalized_reconstruction_loss = unnormalized_reconstruction_loss.mean()

        final_loss = reduced_reconstruction_loss

        # Loss summands for records
        summands.update(
            {
                "mse_loss": (mse_loss.item(), -1),
                "reconstruction_loss": (reduced_reconstruction_loss.item(), -1),
                "unnormalized_reconstruction_loss": (
                    unnormalized_reconstruction_loss.item(),
                    -1,
                ),
            }
        )

        return LossTerm(
            value=final_loss,
            batch_size=output_seqs.size(0),
            num_target_elements=num_target_elements.item(),
            summands=summands,
        )


@CriterionsFactory.register("two_tower_diffusion_next_sent_finetuning")
class DiffusionNextSentFinetuningCriterion(TwoTowerDiffusionCriterion):
    def __init__(
        self,
        config: TowerDiffusionLCMCriterionConfig,
        model: TwoTowerDiffusionLCModel,
    ):
        super().__init__(config, model, LCMStyle.SUPERVISED)

    def prepare_input_and_mask(
        self,
        batch: LCMInput,
    ) -> Tuple[EmbeddingsBatch, EmbeddingsBatch, torch.Tensor]:
        """
        A method for preparing model inputs and mask for a batch.
        It will be typically reused by the `__call__`
        implementations of the subclasses.

        Returns:
            - input_batch: context
            - target_batch: denoiser input
            - target_mask  mask of positions to compute the loss over
        """

        # Prepare the input as in MSE LCM
        input_embeddings = batch.prepare_input(style=self.style)

        assert input_embeddings.source_lengths is not None, (
            "Missing source lengths needed for the two-tower supervised fintuning"
        )

        target_embeddings = EmbeddingsBatch(*pad_seqs(batch.target))  # type: ignore

        # Normalize the embeddings
        if self.normalize_in_criterion:
            input_embeddings = input_embeddings.normalize_seqs(self.sonar_normalizer)
            target_embeddings = target_embeddings.normalize_seqs(self.sonar_normalizer)

        target_mask = torch.ones(
            size=target_embeddings.shape[:-1],
            dtype=torch.bool,
            device=input_embeddings.seqs.device,
        )

        # Factor in padded positions:
        if target_embeddings.padding_mask is not None:
            target_mask &= target_embeddings.padding_mask.materialize()

        return input_embeddings, target_embeddings, target_mask
