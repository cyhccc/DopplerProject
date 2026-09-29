param(
    [string]$DataBinDir = "E:\Code\Cuda\LinearArrayCPWC_OPT_test\bins_7883227",
    [int]$MaxFiles = 75,
    [int]$CutoffStart = 50,
    [int]$CutoffEnd = 800,
    [ValidateSet("double", "single")]
    [string]$MatlabCpuPrecision = "double",
    [switch]$RunMatlabCpu
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path ".\outputs" | Out-Null

cmd /c ".\manual_build.cmd"

$MaxFilesLabel = "{0:D3}" -f $MaxFiles
$cudaPrefix = ".\outputs\power_doppler_7883227_rf001_to_$($MaxFilesLabel)_cuda_covsvd_cutoff$($CutoffStart)_$($CutoffEnd)_red"
& ".\manual_build\LinearArrayCPWC_OPT_manual.exe" `
    doppler7883227_svd_powdop_binserial `
    $DataBinDir `
    $cudaPrefix `
    $CutoffStart `
    $CutoffEnd `
    $MaxFiles

if ($RunMatlabCpu) {
    $matlabName = if ($MatlabCpuPrecision -eq "single") { "matlab_cpu_single" } else { "matlab_cpu" }
    $matlabPrefix = Join-Path $PSScriptRoot "outputs\power_doppler_7883227_rf001_to_$($MaxFilesLabel)_$matlabName"
    matlab -batch "cd('$PSScriptRoot'); run_7883227_matlab_cpu_cuda_matched('BinDir','$DataBinDir','MaxFiles',$MaxFiles,'OutputPrefix','$matlabPrefix','CutoffStart',$CutoffStart,'CutoffEnd',$CutoffEnd,'InterpChunkPixels',32768,'DisplayMinDb',-40,'DisplayMaxDb',0,'NumericPrecision','$MatlabCpuPrecision');"
} else {
    $matlabName = if ($MatlabCpuPrecision -eq "single") { "matlab_cpu_single" } else { "matlab_cpu" }
    $matlabPrefix = Join-Path $PSScriptRoot "outputs\power_doppler_7883227_rf001_to_$($MaxFilesLabel)_$matlabName"
}

$cudaPower = "$cudaPrefix`_power_sum.bin"
$matlabPower = "$matlabPrefix`_power_sum.bin"
if (!(Test-Path $matlabPower)) {
    throw "MATLAB CPU power_sum not found: $matlabPower. Re-run with -RunMatlabCpu."
}

$comparePrefix = ".\outputs\compare_7883227_rf001_to_$($MaxFilesLabel)_matlab_cpu_vs_cuda_cutoff$($CutoffStart)_$($CutoffEnd)"
if ($MatlabCpuPrecision -eq "single") {
    $comparePrefix = ".\outputs\compare_7883227_rf001_to_$($MaxFilesLabel)_matlab_cpu_single_vs_cuda_cutoff$($CutoffStart)_$($CutoffEnd)"
}
py -3.12 ".\compare_doppler_power_7883227.py" `
    --cuda-power-sum $cudaPower `
    --matlab-power-sum $matlabPower `
    --width 256 `
    --height 180 `
    --display-min-db -30 `
    --display-max-db 0 `
    --diff-min-db -70 `
    --diff-max-db 0 `
    --cuda-label CUDA `
    --matlab-label "MATLAB CPU $MatlabCpuPrecision" `
    --output-prefix $comparePrefix

Write-Host "[Done] Quality comparison report: $comparePrefix.txt"
Write-Host "[Done] Quality comparison figure: $($comparePrefix)_side_by_side.png"
