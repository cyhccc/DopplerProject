param(
    [string]$DataBin = "E:\Code\Cuda\LinearArrayCPWC_OPT_test\bins_7883227\RF_001_7883227_bs100bw_iq_float2.bin",
    [string]$MachRoot = "E:\Code\Python\mach-main\mach-main",
    [string]$FrameCounts = "64,128,256,512,800",
    [int]$Warmup = 1,
    [int]$Repeat = 5
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path ".\outputs" | Out-Null

cmd /c ".\manual_build.cmd"

$ourCsv = ".\outputs\benchmark_rf001_our_cuda_beamforming.csv"
$machCsv = ".\outputs\benchmark_rf001_mach_beamforming.csv"
$outputPrefix = ".\outputs\benchmark_rf001_our_cuda_vs_mach_beamforming"

& ".\manual_build\OurCudaBeamformBench.exe" `
    $DataBin `
    $ourCsv `
    $FrameCounts `
    $Warmup `
    $Repeat

py -3.12 ".\benchmark_mach_beamforming_rf001.py" `
    --rf-bin $DataBin `
    --mach-root $MachRoot `
    --output-csv $machCsv `
    --frame-counts $FrameCounts `
    --warmup $Warmup `
    --repeat $Repeat

py -3.12 ".\plot_beamforming_compare.py" `
    --our-csv $ourCsv `
    --mach-csv $machCsv `
    --output-prefix $outputPrefix

Write-Host "[Done] Beamforming comparison CSV: $outputPrefix.csv"
Write-Host "[Done] Beamforming comparison figure: $outputPrefix.png"
