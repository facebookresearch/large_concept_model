# Evaluation

After you have trained an LCM, the checkpoint will be saved in a folder under the name `model.pt`, together with the model card under the name `model_card.yaml`. We also provide the library to evaluate the LCM and LLM. Using this library brings many benefits: You can reproduce the experiments done in the paper, you can inspect the results in an unified way, and you can also scale up the experiments for very large datasets in SLURM cluster. This document shows how to evaluate the model for different downstream tasks using the LCM eval library.

## Step 1: Prepare the data

Since an LCM expects input data in sentence level, we need to preprocess the evaluation datasets accordingly. This includes parsing the raw content and
splitting  texts into sentences, then embedding them into vectors using a Sonar encoder.

The example below shows how we prepare the data for CNN Dailymail. We load the dataset from Huggingface using [`datasets` API](https://huggingface.co/docs/datasets/en/index). The sentence splitting is done using [wtpsplit](https://github.com/segment-any-text/wtpsplit). Make sure to specify `--extra data` in installing the project to include these libraries.

All processing logic is implemented in the file `prepare_evaluation_data.py`, as described below.

### Step 1.1: Process the split:
Next, we download and parse the content (source text and summaries), saving different splits into JSON format

```shell
uv run --extra data prepare_evaluation_data.py prepare_data \
    --dataset_name=cnn_dailymail \
    --output_dir=jsonl_dataset \
    --source_text_column=article \
    --target_text_column=highlights \
    --version=3.0.0 \
    --prompt_prefix="Summarize the following news to a concise list of highlights.\n[Text Start]:\n"
    --prompt_suffix="\n[Text End]"
```

Explain: In the above script, `cnn_dailymail` and `3.0.0` is the name and configuration of the dataset as available in HuggingFace `datasets`, `article` and `highlights` are source and summary columns. The `prompt_prefix` and `prompt_suffix` are optional arguments, if specified they will be prepended and appended to each source text to form the complete prompt. These arguments are useful if you want to embed the prompts into the dataset, and let them process all at once together with the text. Alternatively, we can specify them at later phase, when we evaluate the model (in which case the model will process the prompts on the fly)

> **_NOTE:_**  When `prompt_prefix` or `prompt_suffix` are specified, the dataset schema will change, i.e. the columns are renamed to "prompt" for input and "answer" for output. This is to indicate that we are handling the "processed" dataset and not the original one.

The output will be stored in different files `[split].jsonl` under the directory `output_dir`. 


### Step 1.2: Sentence splitting and embedding:

To perform sentence splitting and sonar embedding for each split, run the following command:

```shell
uv run --extra data prepare_evaluation_data.py embed \
    --input_path=jsonl_dataset/cnn_dailymail/test.jsonl \
    --source_text_column=prompt \
    --target_text_column=answer \
    --output_dir=parquet_dataset/cnn_dailymail \
    --lang=eng_Latn \
    --mode=local \
    --log_dir=/tmp/logs/embed_cnndm
```

Depending on your machine, this might take some time. Alternatively, you can try to run in your SLURM cluster with the argmnent `--mode=slurm --shards=NO_OF_PARALLEL_JOBS`. This requires changing your SLURM config accordingly. We use [submitit](https://github.com/facebookincubator/submitit) to configure the job launcher. Here is the relevant excerpt in the script:

```python
launcher = Launcher(
      cache=None,
      config_dump_dir=Path(log_dir) / "conf",
      log_folder=Path(log_dir) / "logs",
      cluster=mode,
      update_parameters={"partition": "your_slurm_partition"},
  )
  _ = await launcher.schedule(inst_stopes_module)
```



## Step 2: Choose the predictor for evaluation

To run the evaluation, we first need to map the model to a `Predictor`, which is an object that streamlines a number of steps: Loading the models, reading the prompts, performing the inference, decoding the outputs according to a given user setting, and finally formatting the text into the user-friendly format. Currently, the list of supported model families and their predictors is below. All predictors are found in "lcm/evaluation/predictors" and are registered in lcm.evaluation.predictors`_PREDICTOR_CONFIG_MAP`


| Predictor               | Model family           | Model identifier                                           |
|-------------------------|------------------------|------------------------------------------------------------|
| huggingface             | AutoModel transformers | `model_name`, `revision`, `model_class`, `tokenizer_class` |
| llama3                  | LlaMA 3.x              | `model_name`                                               |
| gemma                   | Gemma                  | `model_name`                                               |
| base_lcm                | Base LCM               | `model_card`                                               |
| two_tower_diffusion_lcm | Two-tower diffusion LCM| `model_card`                                               |


Next, we specify how the decoder generate texts with different generation options. 

For LLMs, the options are parameters found in [transformers.GenerationConfig](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig), and we port in the predictors the most popular ones: `repetition_penalty`, `encoder_repetition_penalty`, `encoder_no_repeat_ngram_size`, `no_repeat_ngram_size`.

For LCMs, the options are found in [LCMGeneratorOptions](https://github.com/facebookresearch/large_concept_model/blob/main/lcm/inference/lcm/generator.py#L31) (for Base LCM) or [DiffusionLCMGeneratorOptions](https://github.com/facebookresearch/large_concept_model/blob/main/lcm/inference/two_tower_diffusion_lcm/generator.py#L31) (for Two-tower diffusion LCM). These options only specify how to generate output embeddings using diffusion process. We also want to specify the sonar decoder options, which dictates how the embeddings are decoded into texts, using parameters in [SonarDecoderConfig](https://github.com/facebookresearch/large_concept_model/blob/main/lcm/datasets/configs.py#L69).


## Step 3: Choose a downstream task and run the evaluation

To run the downstream task, specify the task name and configuration, as well as parameters. We provide example tasks that were used in the paper:

### LLM evaluation tasks:

| Task name               | Task configuration               | Explanation                                                                                                 |
|-------------------------|----------------------------------|-------------------------------------------------------------------------------------------------------------|
| cnn_dailymail           | cnn_dailymail_{form}llm.{split}  | {form} can be empty for or "inverse_" for summary expansion, {split} can be "test", "validation" or "train" |
| xsum                    | xsum_{form}llm.{split}           | {form} can be empty for or "inverse_" for summary expansion, {split} can be "test", "validation" or "train" |
| xlsum_llm               | xlsum_llm.{lang}.{split}         | {lang} refers to one value in [language list](../../lcm/evaluation/tasks/xlsum.py), {split} can be "test", "validation" or "train" |

The evaluation library provides the handy CLI to evaluate using `lcm.evaluation` entry. Example command for evaluating the Meta Llama 3.1 8B instruction:

```shell
uv run torchrun --standalone --nnodes=1 --nproc-per-node=1 -m lcm.evaluation \
  --predictor llama3  \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --generator_batch_size 16 \
  --tasks cnn_dailymail_llm.test \
  --task_args '{"max_gen_len": 200}' \
  --dataset_dir jsonl_dataset/cnn_dailymail \
  --data_loading.batch_size 16 \
  --dataset.soure_text_column prompt \
  --dataset.source_target_column answer \
  --dump_dir output_results
```

In the example above, we load the model "meta-llama/Llama-3.1-8B-Instruct" as [specified](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) in HuggingFace, evaluate it on the CNN dailymail in which we process using the `prepare_evaluation_data.py` script as in Step 1.1, and store the results in the folder specified via `dump_dir`. The argument `dataset_dir` refers to the value of the argument `output_dir` in Step 1.1.

In some cases, the model requires authentication token to evaluate. You can obtain them in HuggingGface (see [User Access Tokens](https://huggingface.co/docs/hub/en/security-tokens)), then add the parameter `--use_auth_token [YOUR TOKEN]` to the CLI command.

In the above example, we need to provide the `source_text_column` and `source_target_column` parameters, because in Step 1, we inject the prompts direcly to the dataset and renamed the columns accordingly (to differentiate with "original" datasets).  You can also skip this part and customize the prompt for each for each evaluation run. To do this, instead of specifying the `prompt_prefix` and `prompt_suffix` when preparing the data (as shown in the example in Section 1.1), we specify `dataset.source_prefix_text` and `dataset.source_suffix_text` during the evaluation run:

```shell
uv run torchrun --standalone --nnodes=1 --nproc-per-node=1 -m lcm.evaluation \
  --predictor llama3  \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --generator_batch_size 16 \
  --tasks cnn_dailymail_llm.test \
  --task_args '{"max_gen_len": 200}' \
  --dataset_dir jsonl_dataset/cnn_dailymail \
  --data_loading.batch_size 16 \
  --dataset.source_prefix_text "Summarize the following news to a concise list of highlights.\n[Text Start]:\n" \
  --dataset.source_suffix_text "\n[Text End]" \
  --dump_dir output_results
```

> **_NOTE:_** the missing parameters `source_text_column` and `target_text_column` and the new parameters `source_prefix_text`, `target_prefix_text` are becase we do not modify the column schema. Therefore, the original text columns ("article", "highlights") are kept and not specified in the CLI.

It is also possible to provide the prompt from a YAML file. This is handy when you have to engineer the prompts carefully and have a very long detailed text. We provide one example prompt in the file [instruction.yaml](./instruction.yaml). The example command is:

```shell
uv run torchrun --standalone --nnodes=1 --nproc-per-node=1 -m lcm.evaluation \
  --predictor llama3  \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --generator_batch_size 16 \
  --tasks cnn_dailymail_llm.test \
  --task_args '{"max_gen_len": 200}' \
  --dataset_dir jsonl_dataset/cnn_dailymail \
  --data_loading.batch_size 16 \
  --prompt_file instruction.yaml \
  --dump_dir output_results
```

### LCM evaluation tasks:

In contrast to LLM, the LCMs expect dataset to be preprocessed in Parquet format, with inputs being (sonar-) sentence embeddings. To evaluate an LCM on a ddownstream task, point to the directory consisting of the parquet files, as specified in Step 1, and run (example for Two-tower diffusion LCM):


```shell
uv run torchrun --standalone --nnodes=1 --nproc-per-node=1 -m lcm.evaluation \
  --predictor two_tower_diffusion_lcm  \
  --model_card path/to/the/model_card.yaml \
  --generator_batch_size 16 \
  --tasks lcm_generation \
  --task_args '{"max_gen_len": 200}' \
  --dataset.parquet_path parquet_dataset/cnn_dailymail \
  --dataset.source_column prompt_sentences_sonar_emb \
  --dataset.source_text_column prompt_sentences \
  --dataset.target_column answer_sentences_sonar_emb \
  --dataset.target_text_column answer_sentences \
  --data_loading.batch_size 16 \
  --dump_dir output_results
```

Similar to LLM evaluation, it is possible to specify the prompt prefix and suffix ad-hoc. This text will be sentence-split and embedded using the standard Sonar encoder.


## Common CLI arguments
<a id="param_list"></a>

|  Argument | Description |
|----------|----------|
| `predictor` | The wrapper of the nodel to be evaluated. See Step 2 for more details
| `data_loading.max_samples`   | Evaluate on the maximum _k_ examples in the test data. Useful for debugging   |
| `data_loading.batch_size`   | Loading and evaluate data in batch. By default `batch_size=10`   |
| `dataset_dir` | The directory consists of different JSONL files processed in Step 1. Only used in LLM evaluation
| `dataset.parquet_path` | The parquet path  consists of different Parquet files files processed in Step 1. Only used in LCM evaluation
| `dataset.source_column` | The column in the data that refers to the input embedding. Not applicable when evaluating LLMs.
| `dataset.source_text_column` | The column in the data that refers to the input text.
| `dataset.target_column` | The column in the data that refers to the ground-truth embedding. Not applicable when evaluating LLMs.
| `dataset.target_text_column` | The column in the data that refers to the ground-truth text.
| `dataset.source_text_prefix` | The text that will prepended to each input text to make the prompt for the model.
| `dataset.source_text_suffix` | The text that will appended after each input text to make the prompt for the model.
| `task_args` | The JSON-formatted string that represents the task arguments. See [task param list](#task_param_list) below.
| `dump_dir` | The directory consisting output of the eval run. If successful, there should be a file `metrics.eval.jsonl` that consists of metric results, the directory `results` that capture the verbose command line used with the detailed output scores, and the directory `raw_results` that shows
the model output for each individual sample, together with the per-sample metric results.
| `task` | Task configuration. See Step 3 for examples.
| `task_args` | The JSON-formatted string that represents the task arguments. See [task param list](#task_param_list) below.
| `launcher` | Whether the CLI should be run locally, or in SLURM cluster. Accepted value is `local`, `submitit` (SLURM) or `standalone` (debug mode). 
| `job_args` | Parameters used when launching eval in SLURM. See [below](#slurm-eval) for more details.

*Table: List of common arguments in Evaluation CLI.*

_Note_: In above examples, free arguments such as `generator_batch_size`, `temperature`, etc. are generator options. They depend on specific predictor, as explained in Step 2. Giving a wrong option will trigger and error in the CLI. 

Outputs dumped in the directory specified by `dump_dir` will be structured as:
```
.
├── metadata.jsonl
├── metrics.eval.jsonl
├── raw_results
├── results
└── tb
```
where `metrics.eval.jsonl` contains corpus-level scores.

### Task arguments
<a id="task_param_list"></a>


In both LLM and LCM evaluation, we can configure how inputs and outputs are processed:

- `max_prompt_len`: The model context size, i.e. maximum number of tokens (in LLM) or sentences (in LCM) that the model can accept
- `max_gen_len`: The maximum number of tokens (in LLM) or sentences (in LCM) the model should generate. Note that some model generators have its own stopping criteria, so the actual generated text can be much lower than this value.
- `min_gen_len`: The minimum number of tokens (in LLM) or sentences (in LCM) the model should generate.
- `max_gen_len_ratio`: The maximum  number of tokens (in LLM) or sentences (in LCM) the model should generate _in comparison_ to the input length. For example, if the source document is 5K long and `max_gen_len_ratio=0.2`, we are asking the model to generate 1K-long output (Again, due to the model generators inner behaviour, the output can be much shorter)


## Evaluate big datasets
<a id="slurm-eval"></a>

The above command is sufficient for most cases where you load the model into one GPU and evaluate the whole dataset locally, i.e. the datasets and everyhing is loaded into the memory.
For bigger datasets, or for models which are not easily run in one GPU, or two slow to evaluate, we can submit the evaluation job to the SLURM cluster by choosing the `launcher=submitit`:

```shell

slurm_partition=YOUR_SLURM_PARTITION
shards=NUMBER_OF_SLURM_NODES
timeout_min=JOB_TIMEOUT_IN_MINUTES


uv run -m lcm.evaluation \
  --predictor two_tower_diffusion_lcm  \
  --model_card path/to/the/model_card.yaml \
  --generator_batch_size 16 \
  --tasks lcm_generation \
  --task_args '{"max_gen_len": 200}' \
  --dataset.parquet_path parquet_dataset/cnn_dailymail \
  --data_loading.batch_size 16 \
  --dump_dir output_results \
  --launcher submitit \
  --job_args '{"launcher.cache": "null", "launcher.partition": "'${slurm_partition}'", "launcher.qos": "'${qos}'", "nshards": '${shards}', "requirements": {"gpus_per_node": 1, "timeout_min": '${timeout_min}'}}' \
```

The parameters in `job_args` are submitit parameters. Please refer to https://github.com/facebookincubator/submitit for more comprehensive documentation and parameters list. 