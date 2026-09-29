param(
    [string]$DataBinDir = "E:\Code\Cuda\LinearArrayCPWC_OPT_test\bins_7883227",
    [int[]]$MaxFilesList = @(1, 10, 30),
    [int]$CutoffStart = 50,
    [int]$CutoffEnd = 800,
    [switch]$RunMatlabCpu
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

foreach ($maxFiles in $MaxFilesList) {
    Write-Host "[Info] Running quality comparison: MaxFiles=$maxFiles cutoff=[$CutoffStart,$CutoffEnd]"
    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", ".\run_quality_compare.ps1",
        "-DataBinDir", $DataBinDir,
        "-MaxFiles", $maxFiles,
        "-CutoffStart", $CutoffStart,
        "-CutoffEnd", $CutoffEnd
    )
    if ($RunMatlabCpu) {
        $args += "-RunMatlabCpu"
    }
    powershell @args
}
