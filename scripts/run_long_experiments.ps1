Param(
  [int]$SftMaxSteps = 800,
  [int]$DpoMaxSteps = 400,
  [int]$SeqLen = 128
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

# Low-disk + HF caches on D:
$env:TEMP='D:\pip-tmp'
$env:TMP='D:\pip-tmp'
$env:HF_HOME='D:\hf-cache'
$env:HF_HUB_CACHE='D:\hf-cache\hub'
$env:TRANSFORMERS_CACHE='D:\hf-cache\transformers'
$env:HF_DATASETS_CACHE='D:\hf-cache\datasets'
$env:PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:64'

$python = '.\.venv312\Scripts\python.exe'

# Real processed splits
$sftTrain = 'data\processed\splits\sft_train.jsonl'
$sftVal   = 'data\processed\splits\sft_val.jsonl'
$dpoTrain = 'data\processed\splits\dpo_train.jsonl'
$dpoVal   = 'data\processed\splits\dpo_val.jsonl'

$ts = (Get-Date).ToString('yyyyMMdd_HHmm')
$sftOut = "checkpoints\qwen3-1.7b-sft-lora_LONG_${ts}"
$dpoOut = "checkpoints\qwen3-1.7b-dpo_LONG_${ts}"

Write-Host "SFT out: $sftOut"
Write-Host "DPO out: $dpoOut"

# Phase 1: SFT (LoRA)
& $python scripts\train_sft_lora.py `
  --model_name_or_path Qwen/Qwen3-1.7B-Base `
  --sft_jsonl $sftTrain `
  --sft_eval_jsonl $sftVal `
  --output_dir $sftOut `
  --max_seq_length $SeqLen `
  --max_steps $SftMaxSteps `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 1 `
  --learning_rate 0.0002 `
  --logging_steps 10 `
  --save_steps 200 `
  --lora_r 4 `
  --lora_alpha 8 `
  --target_modules q_proj v_proj `
  --fp16 --no_bf16 --trust_remote_code

if ($LASTEXITCODE -ne 0) { throw "SFT failed with exit code $LASTEXITCODE" }

# Phase 2: DPO (from SFT adapter)
& $python scripts\train_dpo.py `
  --model_name_or_path $sftOut `
  --dpo_jsonl $dpoTrain `
  --dpo_eval_jsonl $dpoVal `
  --output_dir $dpoOut `
  --max_length $SeqLen `
  --max_steps $DpoMaxSteps `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 1 `
  --learning_rate 1e-5 `
  --logging_steps 10 `
  --save_steps 200

if ($LASTEXITCODE -ne 0) { throw "DPO failed with exit code $LASTEXITCODE" }

Write-Host "DONE: SFT=$sftOut DPO=$dpoOut"