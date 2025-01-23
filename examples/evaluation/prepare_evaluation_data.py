# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#
#
# This module illustrates how to prepare a public dataset into the format
# ready for the evaluation of both LCMs and LLMs.
#
# The example datassets are available in HuggingFace, although it should
# be straightforward to apply to other datasets as well.
#

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import datasets
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from scipy.signal import find_peaks
from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements
from stopes.modules.partitioned_data_mapper import stopes_data_mapper
from stopes.modules.preprocess.sonar_text_embedding import (
    LangColumnConfig,
    SonarTextBatchEmbedder,
    SonarTextEmbedderConfig,
)
from stopes.utils.sharding.abstract_shards import (
    BatchFormat,
    batch_to_pandas,
    batch_to_table,
)
from stopes.utils.sharding.json_shards import JSONShardConfig
from stopes.utils.sharding.parquet_shards import ParquetOutputConfig
from wtpsplit import SaT, indices_to_sentences

from lcm.datasets.sentence_splitting import remove_emojis

logger = logging.getLogger(__name__)

SPLITS = ["test", "validation"]

INPUT_KEY = "prompt"
OUTPUT_KEY = "answer"


@dataclass
class SonarColumnRenameAndEmbedConfig(SonarTextEmbedderConfig):
    input_column: str = INPUT_KEY
    output_column: Optional[str] = OUTPUT_KEY


class InstSonarEmbedder(SonarTextBatchEmbedder):
    config: SonarColumnRenameAndEmbedConfig

    def __init__(self, config: SonarColumnRenameAndEmbedConfig) -> None:
        super().__init__(config)
        self.sat_splitter = SaT("sat-3l")
        self.sat_splitter.to("cuda")

    def split_one_single_column(
        self,
        col: pd.Series,
    ) -> pd.Series:
        texts = col.to_list()
        try:
            texts = list(map(remove_emojis, texts))
        except TypeError:
            logger.error(
                f"Could not remove emojis from text={texts} type={type(texts)}"
            )
            raise TypeError("cannot remove emojis!")

        outputs = self.sat_splitter.split(
            texts,
            threshold=0.6,
            batch_size=256,
            outer_batch_size=1024,
        )
        sentences = []
        for row in outputs:
            sentences.append([s.strip() for s in row if s.strip()])
        return pd.Series(sentences)

    @torch.inference_mode()
    def resplit_sentence(self, text, max_length_char: int = 256):
        if len(text) < max_length_char:
            return text

        probs = self.sat_splitter.predict_proba(text)
        seq_len = len(probs)
        nb_split = seq_len // max_length_char + 1
        sentence_threshold = np.quantile(probs, 1 - (2.5 * nb_split / seq_len))
        peaks, _ = find_peaks(probs, height=sentence_threshold, distance=32)
        sentences = indices_to_sentences(
            text,
            peaks,
            strip_whitespace=False,
        )
        return sentences

    def resplit_long_sentences(
        self, col: pd.Series, max_length_char: int = 256
    ) -> pd.Series:
        max_sentence_lenght_in_doc = col.apply(lambda row: max([len(x) for x in row]))
        transformed_col = []
        for max_len, doc in zip(max_sentence_lenght_in_doc, col):
            if max_len <= max_length_char:
                transformed_col.append(doc)
            else:  # resplit:
                _doc = []
                for sent in doc:
                    if len(sent) > max_length_char:
                        resplits = self.resplit_sentence(
                            sent, max_length_char=max_length_char
                        )
                        _doc.extend(resplits)
                    else:
                        _doc.append(sent)
                transformed_col.append(_doc)

        return pd.Series(transformed_col)

    def __call__(self, batch: pa.Table) -> pa.Table:
        batch = batch_to_pandas(batch)

        batch[f"{INPUT_KEY}_sentences"] = self.split_one_single_column(
            batch[self.config.input_column]
        )
        batch[f"{INPUT_KEY}_sentences"] = self.resplit_long_sentences(
            batch[f"{INPUT_KEY}_sentences"],
            max_length_char=256,
        )

        if self.config.output_column is not None:
            batch[f"{OUTPUT_KEY}_sentences"] = self.split_one_single_column(
                batch[self.config.output_column]
            )
            # Avoid too much resplitting on the target side
            batch[f"{OUTPUT_KEY}_sentences"] = self.resplit_long_sentences(
                batch[f"{OUTPUT_KEY}_sentences"], max_length_char=256
            )

        return super().__call__(batch_to_table(batch))


def prepare_data(
    dataset_name: str,
    output_dir: str,
    source_text_column: str = INPUT_KEY,
    target_text_column: Optional[str] = OUTPUT_KEY,
    version: Optional[str] = None,
    prompt_prefix: Optional[str] = None,
    prompt_suffix: Optional[str] = None,
):
    """Download HuggingFace datasets and parse them into JSON format"""
    ds = datasets.load_dataset(dataset_name, version)

    prompt_prefix = prompt_prefix or ""
    prompt_suffix = prompt_suffix or ""

    # If there is no prompt added to the dataset, we keep the original column names
    if not prompt_prefix and not prompt_suffix:
        source = source_text_column
        target = target_text_column
    else:
        source = INPUT_KEY
        target = OUTPUT_KEY

    for split in SPLITS:
        with open(
            Path(output_dir) / f"{dataset_name}/{split}.jsonl", "w", encoding="utf-8"
        ) as o:
            for item in ds[split]:
                prompt = prompt_prefix + item[source_text_column] + prompt_suffix
                output_item = {
                    source: prompt,
                    "split": split,
                    "category": f"{dataset_name}",
                }
                if target_text_column is not None:
                    output_item[target] = item[target_text_column]
                o.write(json.dumps(output_item) + "\n")


async def embed(
    input_path: str,
    output_dir: str,
    source_text_column: str = INPUT_KEY,
    target_text_column: Optional[str] = OUTPUT_KEY,
    lang: str = "eng_Latn",
    mode: Literal["local", "slurm"] = "local",
    shards: int = 1,
    log_dir: Optional[str] = None,
):
    inst_sonar_config = SonarColumnRenameAndEmbedConfig(
        column_config=[
            LangColumnConfig(f"{OUTPUT_KEY}_sentences", lang_value=lang),
            LangColumnConfig(f"{INPUT_KEY}_sentences", lang_value=lang),
        ],
        input_column=source_text_column,
        output_column=target_text_column,
        batch_size=32,
        device="cuda",
    )

    input_config = JSONShardConfig(
        Path(input_path),
        batch_size=10,  # iterating by small number of documents
        batch_format=BatchFormat.ARROW,
        num_shards=shards,
    )

    output_config = ParquetOutputConfig(output_dir)

    if log_dir is None:
        log_dir = Path.home()

    wrapped_cls = stopes_data_mapper(
        Requirements(mem_gb=60, gpus_per_node=1, cpus_per_task=4)
    )(InstSonarEmbedder)

    inst_stopes_module = wrapped_cls(input_config, output_config, inst_sonar_config)

    launcher = Launcher(
        cache=None,
        config_dump_dir=Path(log_dir) / "conf",
        log_folder=Path(log_dir) / "logs",
        cluster=mode,
        update_parameters={"partition": "learn"},
    )
    _ = await launcher.schedule(inst_stopes_module)


if __name__ == "__main__":
    from fire import Fire

    Fire()
