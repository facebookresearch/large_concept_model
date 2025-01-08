# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import gc
import logging
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import pyarrow.compute as pc
import torch
from fairseq2.data.data_pipeline import DataPipeline, read_sequence
from fairseq2.data.text import TextTokenizer
from fairseq2.gang import FakeGang, Gang
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import pad_seqs
from fairseq2.typing import DataType
from fairseq2.utils.state import Stateful
from sonar.models.sonar_text import load_sonar_tokenizer

from lcm.datasets.base import DataLoader
from lcm.datasets.batch import LCMInput
from lcm.datasets.configs import (
    ColumnsNames,
    DataLoadingConfig,
    ParquetDatasetConfig,
    ParquetDatasetLimitOptions,
    SonarDecoderConfig,
)
from lcm.datasets.utils import move_eos_to_the_end
from lcm.utils.common import set_mkl_num_threads

logger = logging.getLogger(__name__)


def truncate_sequence(tokens: torch.Tensor, max_len: int = 512) -> torch.Tensor:
    if len(tokens) > max_len:
        return tokens[:max_len]
    return tokens


class LCMDataLoader(DataLoader[LCMInput, ParquetDatasetConfig], Stateful):
    def __init__(
        self,
        data_config: DataLoadingConfig,
        datasets: Sequence[ParquetDatasetConfig],
        dtype: DataType = torch.float16,
        use_decoder_backprop: bool = False,
        max_subword_length: int = 64,
        gang: Gang = None,
        sonar_decoder_config: Optional[SonarDecoderConfig] = None,
    ) -> None:
        gang = gang or FakeGang()

        super().__init__(
            data_config=data_config,
            datasets=datasets,
            dtype=dtype,
            gang=gang,
        )
        set_mkl_num_threads()

        self.use_decoder_backprop = use_decoder_backprop
        self.sonar_tokenizer: Optional[TextTokenizer] = None
        self.max_subword_length = max_subword_length
        if sonar_decoder_config is not None:
            self.setup_sonar_decoder_tokenizer(config=sonar_decoder_config)
        self._dummy_example: Optional[LCMInput] = None

    def setup_sonar_decoder_tokenizer(
        self,
        config: SonarDecoderConfig,
    ):
        if self.use_decoder_backprop:
            # The tokenizer
            self.tokenizer = load_sonar_tokenizer(config.tokenizer, progress=False)
            # Target text encoder
            self.sonar_tokenizer = self.tokenizer.create_encoder(
                task="translation",
                lang=config.lang,
                mode="target",
                device=self.gang.device,
            )
        else:
            self.sonar_tokenizer = None

    def _prepare_subword_tokens(
        self, batch: Dict[str, Any]
    ) -> Tuple[Optional[SequenceBatch], Optional[SequenceBatch]]:
        """
        Given a batch of paragraphs/documents,
        prepare a batch of sentences (flattened) tokenized at the subword-level
        to feed to the SONAR decoder (a standard token-level decoder)

        Args:
            batch: attributes of a batch from the dataset.
                    A batch is M documents/paragraphs each spanning
                    a variable number of sentences {N_1, ..., N_M}.

            E.g., {'text_sentences': [[sent^1_1, ...sent^1_{N_1}],
                                        ...[sent^M_1, ... sent^M_{N_M}],
                  'text_sentences_sonar_emb': [X^1 in (N_1, D), ... X^M in (N_M, D)]}
                  where D is the sonar embedding dimension.
        Returns:
            Toeknized sentences (subword-level) in (\sum_i=1^M N_i, max_len)
            where max_len is min(self.max_subword_length, max length of the sentences in the batch)

        """

        if not self.use_decoder_backprop:
            return None, None

        # flatten the sentences from different documents/paragraphs
        flattened_source_text = (
            pc.list_flatten(batch[ColumnsNames.source_text_column.value])
            .to_pandas()
            .values
        )

        pipeline: DataPipeline = (
            read_sequence(flattened_source_text)
            .map(
                [
                    self.sonar_tokenizer,  # type: ignore
                    partial(truncate_sequence, max_len=self.max_subword_length),
                ],
                num_parallel_calls=int(max(8 * self.data_config.num_parallel_calls, 1)),
            )
            .and_return(max_num_warnings=4)
        )

        tokens_seqs, tokens_padding_mask = pad_seqs(list(pipeline))  # type: ignore
        prefix_batch = SequenceBatch(tokens_seqs, tokens_padding_mask)
        # TODO: instead of moving the EOS around, make the tokenizer append at the tokenization.
        target_batch = move_eos_to_the_end(
            prefix_batch,
            eos_token_id=self.tokenizer.vocab_info.eos_idx,
            pad_token_id=self.tokenizer.vocab_info.pad_idx,
        )

        return prefix_batch, target_batch

    def _tokenize_batch(self, batch: Dict[str, Any]) -> LCMInput:
        """
        Given a batch of documents,
        prepare a batch of input features for the LCM
        This step is to simply fetch the right column for source/target & source text
        and convert torch NestedTensors to list of tensors

        Args:
            batch: attributes of a batch from the dataset.
                    A batch is M documents each spanning
                    a variable number of sentences {N_1, ..., N_M}.

            E.g., {'text_sentences': [[sent^1_1, ...sent^1_{N_1}],
                                        ...[sent^M_1, ... sent^M_{N_M}],
                  'text_sentences_sonar_emb': [X^1 in (N_1, D), ... X^M in (N_M, D)]}
                  where D is the sonar embedding dimension.
        Returns:
            LCMInput(
            source: SONAR embeddings of the source text
                i.e [X^1 in (N_1, D), ... X^M in (N_M, D)]
            target: If supervised data:  SONAR embeddings of the source text
            tokens: Tokenized flattened sentences for the SONAR decoder (see `_prepare_subword_tokens`)
            )

        """

        # Prepare sentence-wise subword tokens if needed:
        tokens, target_tokens = self._prepare_subword_tokens(batch)

        # Load target embeddings if requested and to propagate all other embeddings

        possible_emb_columns = {
            "source": ColumnsNames.source_column,
            "target": ColumnsNames.target_column,
        }

        outputs = {
            "tokens": tokens,
            "target_tokens": target_tokens,
            "name": batch[ColumnsNames.dataset_name.value],
            "batch": batch,
        }
        # Convert each embedding to torch Tensor
        for key, col in possible_emb_columns.items():
            col_name = col.value
            if col_name in batch:
                dtype = self.dtype if "_length" not in key else torch.int64
                # We check if `x` is a pyarrow.Scalar; if so, convert it to a Python object.
                embs = []
                for x in batch[col_name]:
                    if isinstance(x, torch.Tensor):
                        # Already a tensor, just ensure correct device/dtype
                        t = x.to(device=self.gang.device, dtype=dtype)
                    elif isinstance(x, pa.Scalar):
                        # Convert PyArrow scalar to Python float/int
                        t = torch.tensor(x.as_py(), device=self.gang.device, dtype=dtype)
                    else:
                        # Convert list / numpy array / etc. to tensor
                        t = torch.tensor(x, device=self.gang.device, dtype=dtype)
                    # If 1D but should be 2D, e.g. (D) -> (1, D), handle here if needed
                    if t.dim() == 1 and "_length" not in key:
                        t = t.unsqueeze(0)
                    embs.append(t)
            else:
                embs = None
            outputs[key] = embs
        assert (
            outputs["source"] is not None
        ), "LCMDataLoader requires `source` sequences to be present in batches"
        return LCMInput(**outputs)

    def iterate_batches(self) -> Iterator[LCMInput]:
        yield from map(self._tokenize_batch, self.pipeline)

    def iterate_dummy_batches(self) -> Iterator[LCMInput]:
        """
        it's needed to simulate the data that follows the strucutre of self.pipeline (by always returning the same element).
        It can be used only for fast forward pass (to avoid uneven sharding multi-gpus training).
        """
        if self._dummy_example is None:
            # patching the params to get less data with less cost
            limited_datasets = deepcopy(self.datasets)
            for ds_conf in limited_datasets:
                assert isinstance(ds_conf, ParquetDatasetConfig)
                ds_conf.limit = ParquetDatasetLimitOptions(nb_fragments=1)

            # Copy the true data config and reduce the batch size.
            # When wrapping data, we want to also wrap the dummy batches
            # to not exceed model max_length
            dummy_dataloading_config = deepcopy(self.data_config)
            dummy_dataloading_config.batch_size = 1

            self._dummy_example = self._tokenize_batch(
                next(
                    iter(
                        self.builder_func(
                            limited_datasets, dummy_dataloading_config, 0, 1
                        )
                    )
                )
            )
        gc.collect()

        while True:
            yield self._dummy_example

    def state_dict(self) -> Dict[str, Any]:
        logger.info("Getting the data pipeline state ...")
        state = self.pipeline.state_dict(strict=False)
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict is not None:
            assert self.pipeline is not None
            if self.data_config.ignore_checkpointed_pipeline:
                logger.warning("Ignoring existing dataloader state")
            else:
                try:
                    self.pipeline.load_state_dict(state_dict)
                    logger.info(f"Reloaded datapipeline state: {str(state_dict)[:400]}")
                except ValueError:
                    logger.warning(
                        f"Failed to load dataloader state: {str(state_dict)[:400]}"
                    )
        else:
            # retro-compatibility
            logger.warning(f"Attempt to restore a dataloader {self} with empty state")
