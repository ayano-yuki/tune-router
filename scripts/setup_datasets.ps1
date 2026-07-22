param(
    [string[]] $ExperimentId = @(),
    [switch] $PriorityOnly,
    [switch] $AllowFloatingRevision,
    [switch] $Force
)

$ErrorActionPreference = "Stop"

$argsList = @("run", "tunescope", "setup-datasets")

foreach ($id in $ExperimentId) {
    $argsList += @("--experiment-id", $id)
}

if ($PriorityOnly) {
    $argsList += "--priority-only"
}

if ($AllowFloatingRevision) {
    $argsList += "--allow-floating-revision"
}

if ($Force) {
    $argsList += "--force"
}

uv @argsList

