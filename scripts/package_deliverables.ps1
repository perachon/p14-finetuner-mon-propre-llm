Param(
  [Parameter(Mandatory=$true)] [string]$Nom,
  [Parameter(Mandatory=$true)] [string]$Prenom,
  # Format attendu: mmaaaa (ex: 032026)
  [Parameter(Mandatory=$true)] [ValidatePattern('^[0-1][0-9][0-9]{4}$')] [string]$DateDemarrage,
  # Nom du zip final (ex: P14_TriageLLM_Dupont_Jean)
  [string]$ZipTitle = "P14_TriageLLM_${Nom}_${Prenom}",
  [string]$OutDir = "runs\\deliverables_out",
  [switch]$NoZip
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

function New-CleanDir([string]$Path) {
  if (Test-Path $Path) { Remove-Item -Recurse -Force $Path }
  New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Copy-IfExists([string]$Src, [string]$Dst) {
  if (Test-Path $Src) {
    $parent = Split-Path -Parent $Dst
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Copy-Item -Recurse -Force $Src $Dst
  }
}

function Copy-AdapterArtifacts([string]$CheckpointDir, [string]$OutAdapterDir) {
  if (-not (Test-Path $CheckpointDir)) { return }
  New-Item -ItemType Directory -Force -Path $OutAdapterDir | Out-Null

  $names = @(
    'adapter_config.json',
    'adapter_model.safetensors',
    'README.md',
    'training_args.bin',
    'tokenizer.json',
    'tokenizer_config.json',
    'chat_template.jinja'
  )

  foreach ($n in $names) {
    $p = Join-Path $CheckpointDir $n
    if (Test-Path $p) {
      Copy-Item -Force $p (Join-Path $OutAdapterDir $n)
    }
  }
}

# Resolve likely checkpoints
$sftCandidates = @(
  'checkpoints\\qwen3-1.7b-sft-lora_LONG_20260318_1657',
  'checkpoints\\qwen3-1.7b-sft-lora_real_trlcfg',
  'checkpoints\\qwen3-1.7b-sft-lora'
)
$dpoCandidates = @(
  'checkpoints\\qwen3-1.7b-dpo_LONG_20260318_1657',
  'checkpoints\\qwen3-1.7b-dpo_from_sft_lowvram',
  'checkpoints\\qwen3-1.7b-dpo'
)

function First-Existing([string[]]$Paths) {
  foreach ($p in $Paths) {
    if (Test-Path $p) { return $p }
  }
  return $null
}

$sftCkpt = First-Existing $sftCandidates
$dpoCkpt = First-Existing $dpoCandidates

# Output root
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$root = Resolve-Path $OutDir

# Folder names (spec)
$prefix = "${Nom}_${Prenom}"
$d1 = Join-Path $root "${prefix}_1_Dataset_${DateDemarrage}"
$d2 = Join-Path $root "${prefix}_2_Modele_SFT_LoRA_${DateDemarrage}"
$d3 = Join-Path $root "${prefix}_3_Modele_DPO_${DateDemarrage}"
$d4 = Join-Path $root "${prefix}_4_Rapport_${DateDemarrage}"
$d5 = Join-Path $root "${prefix}_5_Deploiement_CICD_${DateDemarrage}"

New-CleanDir $d1
New-CleanDir $d2
New-CleanDir $d3
New-CleanDir $d4
New-CleanDir $d5

# 1) Dataset
Copy-IfExists 'data\\processed' (Join-Path $d1 'data\\processed')
Copy-IfExists 'data\\SOURCES.md' (Join-Path $d1 'data\\SOURCES.md')
Copy-IfExists 'data\\RGPD_PROCESS.md' (Join-Path $d1 'data\\RGPD_PROCESS.md')
Copy-IfExists 'docs\\01_etape1_donnees.md' (Join-Path $d1 'docs\\01_etape1_donnees.md')
Copy-IfExists 'docs\\00_theorie_sft_dpo.md' (Join-Path $d1 'docs\\00_theorie_sft_dpo.md')

# 2) Model SFT
if ($sftCkpt) {
  Copy-AdapterArtifacts $sftCkpt (Join-Path $d2 'adapter')
}
Copy-IfExists 'README.md' (Join-Path $d2 'README.md')

# 3) Model DPO
if ($dpoCkpt) {
  Copy-AdapterArtifacts $dpoCkpt (Join-Path $d3 'adapter')
}
Copy-IfExists 'README.md' (Join-Path $d3 'README.md')

# 4) Report
Copy-IfExists 'report\\rapport_template.md' (Join-Path $d4 'rapport_template.md')
Copy-IfExists 'docs' (Join-Path $d4 'docs')

# 5) Deploy + CI/CD
Copy-IfExists '.github\\workflows\\ci.yml' (Join-Path $d5 '.github\\workflows\\ci.yml')
Copy-IfExists 'docker\\Dockerfile' (Join-Path $d5 'docker\\Dockerfile')
Copy-IfExists 'cloud\\hf_spaces_vllm' (Join-Path $d5 'cloud\\hf_spaces_vllm')
Copy-IfExists 'scripts\\run_api.ps1' (Join-Path $d5 'scripts\\run_api.ps1')
Copy-IfExists 'scripts\\demo_api.ps1' (Join-Path $d5 'scripts\\demo_api.ps1')
Copy-IfExists 'src\\triage_llm\\api' (Join-Path $d5 'src\\triage_llm\\api')
Copy-IfExists 'src\\triage_llm\\schemas.py' (Join-Path $d5 'src\\triage_llm\\schemas.py')

# Write a small index
$index = @()
$index += "Livrables exportés:"
$index += "- 1 Dataset: $d1"
$index += "- 2 Modèle SFT LoRA: $d2"
$index += "- 3 Modèle DPO: $d3"
$index += "- 4 Rapport: $d4"
$index += "- 5 Déploiement + CI/CD: $d5"
$index += ""
$index += "Notes:"
$index += "- Les adapters complets sont également publiés sur Hugging Face: perachon/p14-model"
$index += "- Le dataset complet est sur Hugging Face (privé): perachon/p14-dataset"
$index += ""
Set-Content -Path (Join-Path $root 'INDEX.txt') -Value ($index -join "`r`n") -Encoding UTF8

if (-not $NoZip) {
  $zipPath = Join-Path $root ("${ZipTitle}.zip")
  if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
  Compress-Archive -Path (Join-Path $root '*') -DestinationPath $zipPath
  Write-Host "ZIP created: $zipPath"
} else {
  Write-Host "Export done (no zip). OutDir: $root"
}
