param(
    [switch] $AllowFloatingRevision,
    [switch] $DryRun,
    [switch] $AllowPlaceholderModel,
    [int] $MaxSteps = 0
)

$ErrorActionPreference = "Stop"

function Invoke-TuneScope {
    param([string[]] $Args)
    uv @("run", "tunescope") @Args
}

Invoke-TuneScope @("validate")

if (-not $DryRun) {
    $datasetArgs = @("setup-datasets")
    if ($AllowFloatingRevision) {
        $datasetArgs += "--allow-floating-revision"
    }
    Invoke-TuneScope $datasetArgs
}

New-Item -ItemType Directory -Force reports\run-cards | Out-Null
foreach ($id in @("B0", "Q1", "Q2", "Q3", "Q4", "R1", "R2", "R3", "L1", "D1", "F1")) {
    uv run tunescope run-card $id | Set-Content -Encoding UTF8 "reports\run-cards\$id.md"
}

$common = @()
if ($DryRun) {
    $common += "--dry-run"
}
if ($AllowPlaceholderModel) {
    $common += "--allow-placeholder-model"
}

$trainCommon = @($common)
if ($MaxSteps -gt 0) {
    $trainCommon += @("--max-steps", [string] $MaxSteps)
}

$evalCommon = @($common)
if ($AllowFloatingRevision) {
    $evalCommon += "--allow-floating-revision"
}

Invoke-TuneScope (@("evaluate", "--experiment-id", "B0", "--output-dir", "experiments\results\B0") + $evalCommon)

foreach ($id in @("Q1", "Q2", "Q3", "Q4", "R1", "R3", "L1", "F1")) {
    Invoke-TuneScope (@("train-sft", "--experiment-id", $id, "--output-dir", "experiments\results\$id") + $trainCommon)
    Invoke-TuneScope (@("evaluate", "--experiment-id", $id, "--output-dir", "experiments\results\$id") + $evalCommon)
}

Invoke-TuneScope (@("evaluate", "--experiment-id", "R2", "--reuse-result-from", "Q3", "--output-dir", "experiments\results\Q3") + $evalCommon)
Invoke-TuneScope (@("train-dpo", "--experiment-id", "D1", "--output-dir", "experiments\results\D1") + $trainCommon)
Invoke-TuneScope (@("evaluate", "--experiment-id", "D1", "--output-dir", "experiments\results\D1") + $evalCommon)

Invoke-TuneScope @("report", "--matrix", "experiments\manifests\initial_matrix.yaml", "--results-dir", "experiments\results", "--output", "reports\initial_matrix.md")
