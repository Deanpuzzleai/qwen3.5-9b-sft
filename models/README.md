# Models
For each checkpoint: Hugging Face URL, revision, base model revision, tokenizer and chat
template, quantization, run_id, and the full loading command.

## v1: LoRA SFT adapter

| Field | Value |
|---|---|
| Hugging Face | https://huggingface.co/deanoshuang/qwen3.5-9b-agent-sft-lora (public) |
| Revision | weights `9ce398b54fd1822067bedc58357d28866ea31f34`; model card `8f8cf4b071a6844c40f0629e533acdf2971e9765` (README only) |
| Checkpoint | final, step 350 of 350 (one epoch; fixed rule, no development-set selection) |
| run_id | `sft_main_20261012_154313` |
| Base model | `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, class `Qwen3_5ForConditionalGeneration` |
| Tokenizer and chat template | the base revision's, unchanged (copies in the repository: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`) |
| Quantization | none: bf16 LoRA r 32, alpha 64, 86,556,672 parameters; served merged in bf16 (`scripts/merge_adapter.sh merge`) |
| sha256 | `adapter_model.safetensors` 1ab8642f41e90c063cda58ec5ea1beec5a0e2ac8d8e660d8943f1c29a51840f2 (checked against the Hub's LFS hash); `adapter_config.json` 8cc815560fb4562f77c0c1e803a27d6a68996fbd73091c5dae90cb9fa5da57d4 |
| Model card | the Hugging Face repository's `README.md` (filled from the run's records by `scripts/fill_model_card.py`) |

Loading command:

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText

model = PeftModel.from_pretrained(
    AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a", dtype="bfloat16"),
    "deanoshuang/qwen3.5-9b-agent-sft-lora", revision="9ce398b54fd1822067bedc58357d28866ea31f34")
```

## v5: LoRA SFT adapter, end of epoch 1

| Field | Value |
|---|---|
| Hugging Face | https://huggingface.co/deanoshuang/qwen3.5-9b-agent-sft-efficient-lora (public) |
| Revision | weights `8a5304da419eb350bd62a9720468c5faa8bc139b`; model card `95f7a525b10a75765aaba919f359e98ab375c13c` (README only) |
| Checkpoint | end of epoch 1, step 239 of 478 (2-epoch run, cosine over both); picked by the rule fixed before any result (the end-of-epoch-1 checkpoint) |
| run_id | `sft_v5_20261013_072959`; trainer checkpoint-239 exported by verify-adapter --fix-adapter-config |
| Base model | `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, class `Qwen3_5ForConditionalGeneration` |
| Tokenizer and chat template | the base revision's, unchanged |
| Quantization | none: bf16 LoRA r 32, alpha 64, 86,556,672 parameters; served merged in bf16 (`/mnt/apex-shared/checkpoints/sft_v5_20261013_072959/epoch1_merged`) |
| sha256 | `adapter_model.safetensors` afe7a837ac653bb5dbd72566e590040765e5c739b56deb05c8b363235e55d4aa (checked against the Hub's LFS hash); `adapter_config.json` 34f3e2ed487358dae19afe37f4281233dced2751992eba5ad1e98284dd5b542a |
| Model card | the Hugging Face repository's `README.md` (filled from the run's records by `scripts/fill_model_card.py`) |

Loading command:

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText

model = PeftModel.from_pretrained(
    AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a", dtype="bfloat16"),
    "deanoshuang/qwen3.5-9b-agent-sft-efficient-lora", revision="8a5304da419eb350bd62a9720468c5faa8bc139b")
```

## v5 epoch 2: LoRA SFT adapter, end of epoch 2

| Field | Value |
|---|---|
| Hugging Face | https://huggingface.co/deanoshuang/qwen3.5-9b-agent-sft-v5ep2-lora (public) |
| Revision | weights `7d40e9cdf8f27d06faf45c6a41334fbc72a7b612`; model card `997dfe27213ecf4fe8fc20691fbbf5e2de0a052c` (README only) |
| Checkpoint | end of epoch 2, step 478 of 478 (the run's final checkpoint; the run's rule-based pick is the epoch-1 checkpoint above) |
| run_id | `sft_v5_20261013_072959`; adapter from `/mnt/apex-shared/checkpoints/sft_v5_20261013_072959/epoch2_adapter` |
| Training data and settings | v5 (`data/manifests/v5/DATA_CARD_v5.md`); LR 1e-4 cosine over 2 epochs, effective batch 16; LoRA r 32, alpha 64, dropout 0.05, 12 target modules |
| Base model | `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, class `Qwen3_5ForConditionalGeneration` |
| Tokenizer and chat template | the base revision's, unchanged (copies in the repository: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`) |
| Quantization | none: bf16 LoRA r 32, alpha 64, 86,556,672 parameters; served merged in bf16 (`/mnt/apex-shared/checkpoints/sft_v5_20261013_072959/epoch2_merged`) |
| sha256 | `adapter_model.safetensors` 17788493934de2398ffa3d20f0a6d7df57f5b22a29b1a9d9d2a7aaed4e3f8d03 (matches the ready marker and the Hub's LFS hash); `adapter_config.json` 34f3e2ed487358dae19afe37f4281233dced2751992eba5ad1e98284dd5b542a (matches the ready marker and the Hub's git blob) |
| Model card | the Hugging Face repository's `README.md` (filled from the run's records by `scripts/fill_model_card.py`) |

Loading command:

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText

model = PeftModel.from_pretrained(
    AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a", dtype="bfloat16"),
    "deanoshuang/qwen3.5-9b-agent-sft-v5ep2-lora", revision="7d40e9cdf8f27d06faf45c6a41334fbc72a7b612")
```

## v6: LoRA SFT adapter

| Field | Value |
|---|---|
| Hugging Face | https://huggingface.co/deanoshuang/qwen3.5-9b-agent-sft-v6-lora (public) |
| Revision | weights `c368bac989fa22284d88c13dafa42fb3a58a59db`; model card `7a9e69e15cf556d3a53a59588874ba81174235c1` (README only) |
| Checkpoint | final, step 323 of 323 (one epoch) |
| run_id | `sft_v6_20261013_174750`; adapter from `/mnt/apex-shared/checkpoints/sft_v6_20261013_174750/final_adapter` |
| Training data and settings | v6r2 (`data/README_v6.md`); LR 2e-5 cosine, 1 epoch, effective batch 16; LoRA r 32, alpha 64, dropout 0.05, 12 target modules |
| Base model | `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, class `Qwen3_5ForConditionalGeneration` |
| Tokenizer and chat template | the base revision's, unchanged (copies in the repository: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`) |
| Quantization | none: bf16 LoRA r 32, alpha 64, 86,556,672 parameters; served merged in bf16 (`/mnt/apex-shared/checkpoints/sft_v6_20261013_174750/final_merged`) |
| sha256 | `adapter_model.safetensors` fdeff47f1e1f00aa266313c3d04544725302e7e2b3ccef96ac06b5a4dc994962 (matches the ready marker and the Hub's LFS hash); `adapter_config.json` eed90a291bd6bfcce7a9efd3817c28022a5e9e4a11a756a3d6b3a34fa5d30b64 (matches the ready marker and the Hub's git blob) |
| Model card | the Hugging Face repository's `README.md` (filled from the run's records by `scripts/fill_model_card.py`) |

Loading command:

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText

model = PeftModel.from_pretrained(
    AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a", dtype="bfloat16"),
    "deanoshuang/qwen3.5-9b-agent-sft-v6-lora", revision="c368bac989fa22284d88c13dafa42fb3a58a59db")
```

## v7, LR 2e-5: LoRA SFT adapter

| Field | Value |
|---|---|
| Hugging Face | https://huggingface.co/deanoshuang/qwen3.5-9b-agent-sft-v7-lora (public) |
| Revision | weights `f846ad9e0992ecb7687da54de2ea0002d58e2066`; model card `92e580caeeaec3d4a50baf71d3bf3fa85801ec9e` (README only) |
| Checkpoint | final, step 298 of 298 (one epoch) |
| run_id | `sft_v7_20261013_221813`; adapter from `/mnt/apex-shared/checkpoints/sft_v7_20261013_221813/final_adapter` |
| Training data and settings | v7r1 (`data/manifests/v7/DATA_CARD_v7.md`); LR 2e-5 cosine, 1 epoch, effective batch 12; LoRA r 32, alpha 64, dropout 0.05, 12 target modules |
| Base model | `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, class `Qwen3_5ForConditionalGeneration` |
| Tokenizer and chat template | the base revision's, unchanged (copies in the repository: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`) |
| Quantization | none: bf16 LoRA r 32, alpha 64, 86,556,672 parameters; served merged in bf16 (`/mnt/apex-shared/checkpoints/sft_v7_20261013_221813/final_merged`) |
| sha256 | `adapter_model.safetensors` 0bcd7b98d2c68219ed3128a82f5e117c26730a02f25c115813efd80c8eaf2003 (matches the ready marker and the Hub's LFS hash); `adapter_config.json` 125edd0b55c6b7d493eb9decf153db0f682f0c797b27b13c7e4632214a673a93 (matches the ready marker and the Hub's git blob) |
| Model card | the Hugging Face repository's `README.md` (filled from the run's records by `scripts/fill_model_card.py`) |

Loading command:

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText

model = PeftModel.from_pretrained(
    AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a", dtype="bfloat16"),
    "deanoshuang/qwen3.5-9b-agent-sft-v7-lora", revision="f846ad9e0992ecb7687da54de2ea0002d58e2066")
```

## v7, LR 5e-5: LoRA SFT adapter (the reported SFT model)

| Field | Value |
|---|---|
| Hugging Face | https://huggingface.co/deanoshuang/qwen3.5-9b-agent-sft-v7-lr5e5-lora (public) |
| Revision | weights `62baeb48e57c835af03e0d267da4d77f00a076e3`; model card `b10582982fee5ad324bd2d9c1594770d5d33258c` (README only) |
| Checkpoint | final, step 223 of 223 (one epoch) |
| run_id | `sft_v7lr5e5_20261013_234355`; adapter from `/mnt/apex-shared/checkpoints/sft_v7lr5e5_20261013_234355/final_adapter` |
| Training data and settings | v7r1 (`data/manifests/v7/DATA_CARD_v7.md`); LR 5e-5 cosine, 1 epoch, effective batch 16; LoRA r 32, alpha 64, dropout 0.05, 12 target modules |
| Base model | `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, class `Qwen3_5ForConditionalGeneration` |
| Tokenizer and chat template | the base revision's, unchanged (copies in the repository: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`) |
| Quantization | none: bf16 LoRA r 32, alpha 64, 86,556,672 parameters; served merged in bf16 (`/mnt/apex-shared/checkpoints/sft_v7lr5e5_20261013_234355/final_merged`) |
| sha256 | `adapter_model.safetensors` 5ae53044f937a4f25f4787d71c534b6dae0d3e30c5cdd03bd388caf7f91b1cd7 (matches the ready marker and the Hub's LFS hash); `adapter_config.json` 3f882a89a01b777b870450cd7021ee17981f4da12c648705a679d8e83e780d9f (matches the ready marker and the Hub's git blob) |
| Model card | the Hugging Face repository's `README.md` (filled from the run's records by `scripts/fill_model_card.py`) |

Loading command:

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText

model = PeftModel.from_pretrained(
    AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a", dtype="bfloat16"),
    "deanoshuang/qwen3.5-9b-agent-sft-v7-lr5e5-lora", revision="62baeb48e57c835af03e0d267da4d77f00a076e3")
```

## Licence

The adapters are released under Apache-2.0, the licence of the base model `Qwen/Qwen3.5-9B`. Training-data sources
and their licences are listed in `data/README.md`; repository code quoted in the training rows keeps its own licence
(MIT, Apache-2.0 or BSD).
