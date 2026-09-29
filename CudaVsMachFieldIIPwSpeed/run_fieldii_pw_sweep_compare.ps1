param(
    [string]$DataRoot = "E:\Code\mat\y_branch_fieldii_gpu_pd",
    [string]$MachRoot = "E:\Code\Python\mach-main\mach-main",
    [string]$PwValues = "3:16",
    [ValidateSet("on", "off")]
    [string]$PhaseMode = "on",
    [int]$Warmup = 1,
    [int]$Repeat = 5,
    [switch]$SkipBuild,
    [switch]$CudaOnly,
    [switch]$MachOnly,
    [switch]$SkipPlot
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path ".\outputs" | Out-Null

if (-not (Test-Path -LiteralPath $DataRoot)) {
    throw "Data root not found: $DataRoot"
}
if (-not (Test-Path -LiteralPath $MachRoot)) {
    throw "mach root not found: $MachRoot"
}

$phaseTag = "phase_$PhaseMode"
$ourCsv = ".\outputs\benchmark_fieldii_pw_${phaseTag}_our_cuda.csv"
$machCsv = ".\outputs\benchmark_fieldii_pw_${phaseTag}_mach.csv"
$outputPrefix = ".\outputs\benchmark_fieldii_pw_${phaseTag}_cuda_vs_mach"

if (-not $SkipBuild) {
    cmd /c ".\manual_build.cmd"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA benchmark build failed with exit code $LASTEXITCODE"
    }
}

if (-not $MachOnly) {
    & ".\manual_build\FieldIIPwCudaBeamformBench.exe" `
        $DataRoot `
        $ourCsv `
        $PwValues `
        $Warmup `
        $Repeat `
        $PhaseMode
    if ($LASTEXITCODE -ne 0) {
        throw "Our CUDA benchmark failed with exit code $LASTEXITCODE"
    }
}

if (-not $CudaOnly) {
    & py -3.12 ".\benchmark_mach_fieldii_pw_sweep.py" `
        --data-root $DataRoot `
        --mach-root $MachRoot `
        --output-csv $machCsv `
        --pw-values $PwValues `
        --warmup $Warmup `
        --repeat $Repeat `
        --phase-mode $PhaseMode
    if ($LASTEXITCODE -ne 0) {
        throw "mach benchmark failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipPlot -and -not $CudaOnly -and -not $MachOnly) {
    & py -3.12 ".\plot_fieldii_pw_sweep_compare.py" `
        --our-csv $ourCsv `
        --mach-csv $machCsv `
        --output-prefix $outputPrefix
    if ($LASTEXITCODE -ne 0) {
        throw "Plotting failed with exit code $LASTEXITCODE"
    }
}

Write-Host "[Done] Our CUDA CSV: $ourCsv"
Write-Host "[Done] mach CSV    : $machCsv"
if (-not $SkipPlot -and -not $CudaOnly -and -not $MachOnly) {
    Write-Host "[Done] Summary CSV : $outputPrefix.csv"
    Write-Host "[Done] Figure      : $outputPrefix.png"
}
