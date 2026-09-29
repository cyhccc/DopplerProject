param(
    [ValidateSet(1, 75)]
    [int]$MaxFiles = 1,
    [ValidateSet("double", "single")]
    [string]$NumericPrecision = "single",
    [ValidateSet("matlab", "mex")]
    [string]$BeamformerBackend = "matlab",
    [string]$BeamformDumpPath = "",
    [string]$DataBinDir = "E:\Code\Cuda\LinearArrayCPWC_OPT_test\bins_7883227",
    [string]$OutputDir = "D:\DopplerProject\CudaMatlabCpuQuality\outputs\review_20260908_cutoff50_473"
)

$ErrorActionPreference = "Stop"
$matlabExe = "E:\Compiler\Matlab\bin\matlab.exe"
$projectDir = $PSScriptRoot
$cutoffStart = 50
$cutoffEnd = 473
$frameLabel = if ($MaxFiles -eq 1) { "800frames" } else { "60000frames_75x800" }
$backendLabel = if ($BeamformerBackend -eq "mex") { "_mex" } else { "" }
$prefixName = "matlab_cpu_${NumericPrecision}${backendLabel}_${frameLabel}_cutoff50_473"
$outputPrefix = Join-Path $OutputDir $prefixName
$stdoutLog = Join-Path $OutputDir "${prefixName}.out.log"
$stderrLog = Join-Path $OutputDir "${prefixName}.err.log"

if (!(Test-Path -LiteralPath $matlabExe -PathType Leaf)) {
    throw "MATLAB executable not found: $matlabExe"
}
if (!(Test-Path -LiteralPath $DataBinDir -PathType Container)) {
    throw "Input directory not found: $DataBinDir"
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$existing = Get-ChildItem -LiteralPath $OutputDir -Filter "${prefixName}*" -ErrorAction SilentlyContinue
if ($existing) {
    throw "Refusing to overwrite existing review output: $($existing[0].FullName)"
}

$matlabProjectDir = $projectDir.Replace("'", "''")
$matlabDataDir = $DataBinDir.Replace("'", "''")
$matlabOutputPrefix = $outputPrefix.Replace("'", "''")
$matlabBeamformDumpPath = $BeamformDumpPath.Replace("'", "''")
$useParallelFiles = if ($MaxFiles -eq 75 -and $BeamformerBackend -eq "matlab") { "true" } else { "false" }
$batchCommand = "cd('$matlabProjectDir');run_7883227_matlab_cpu_cuda_matched('BinDir','$matlabDataDir','MaxFiles',$MaxFiles,'OutputPrefix','$matlabOutputPrefix','CutoffStart',$cutoffStart,'CutoffEnd',$cutoffEnd,'InterpChunkPixels',2048,'DisplayMinDb',-40,'DisplayMaxDb',0,'NumericPrecision','$NumericPrecision','BeamformerBackend','$BeamformerBackend','BeamformDumpPath','$matlabBeamformDumpPath','UseParallelFiles',$useParallelFiles,'NumWorkers',3,'WorkerNumThreads',2);"

$process = Start-Process -FilePath $matlabExe `
    -ArgumentList @("-batch", $batchCommand) `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

[pscustomobject]@{
    ProcessId = $process.Id
    MaxFiles = $MaxFiles
    EffectiveFrames = 800 * $MaxFiles
    CutoffStart = $cutoffStart
    CutoffEnd = $cutoffEnd
    NumericPrecision = $NumericPrecision
    BeamformerBackend = $BeamformerBackend
    BeamformDumpPath = $BeamformDumpPath
    UseParallelFiles = $MaxFiles -eq 75 -and $BeamformerBackend -eq "matlab"
    NumWorkers = if ($MaxFiles -eq 75 -and $BeamformerBackend -eq "matlab") { 3 } else { 0 }
    OutputPrefix = $outputPrefix
    StdoutLog = $stdoutLog
    StderrLog = $stderrLog
}
