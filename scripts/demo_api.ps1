Param(
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [ValidateSet("fr","en")] [string]$Lang = "fr",
  [string]$Message = "J'ai une douleur thoracique et je suis essoufflé depuis 30 minutes.",
  [switch]$ShowAudit
)

$ErrorActionPreference = 'Stop'

# Ensure UTF-8 output (avoids garbled accents in Windows terminals)
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$triageUrl = "$BaseUrl/triage"
$healthUrl = "$BaseUrl/health"

Write-Host "GET $healthUrl"
$health = Invoke-RestMethod -Method GET -Uri $healthUrl
$health | ConvertTo-Json -Depth 10

$payload = @{
  patient_message = $Message
  lang = $Lang
  context = @{}
}

Write-Host "POST $triageUrl"
$json = ($payload | ConvertTo-Json -Depth 10)
# Windows PowerShell 5.1 may not send string bodies as UTF-8; FastAPI then fails to parse JSON.
$body = [System.Text.Encoding]::UTF8.GetBytes($json)
$resp = Invoke-RestMethod -Method POST -Uri $triageUrl -ContentType "application/json; charset=utf-8" -Body $body
$resp | ConvertTo-Json -Depth 20

if ($ShowAudit) {
  $id = $resp.interaction_id
  if (-not $id) {
    throw "Missing interaction_id in response"
  }
  $auditUrl = "$BaseUrl/audit/$id"
  Write-Host "GET $auditUrl"
  $audit = Invoke-RestMethod -Method GET -Uri $auditUrl
  $audit | ConvertTo-Json -Depth 30
}
