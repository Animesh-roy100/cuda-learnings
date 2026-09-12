<#
.SYNOPSIS
  Full verification: environment, tests, benchmarks, and both profilers.

.DESCRIPTION
  Written to be run AFTER a reboot, when the RmProfilingAdminOnly registry
  change has taken effect and Nsight Compute can read GPU counters.

  Checks, in order:
    1. driver / toolkit / GPU counter permission
    2. cmake configure + build
    3. ctest (110 cases across 9 suites)
    4. all 8 benchmarks, exit codes and output sanity
    5. nsys timelines
    6. ncu metrics -- the step that was blocked before the reboot

.EXAMPLE
  .\scripts\verify_all.ps1
  .\scripts\verify_all.ps1 -SkipBuild      # reuse the existing build
#>
[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [switch]$SkipProfile
)

# 'Continue': under 'Stop', Windows PowerShell turns any native-tool stderr
# line into a terminating error, so a benign nsys warning would abort the run.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$results = [ordered]@{}

function Section($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Pass($t) { Write-Host "  [PASS] $t" -ForegroundColor Green }
function Fail($t) { Write-Host "  [FAIL] $t" -ForegroundColor Red }
function Note($t) { Write-Host "  $t" -ForegroundColor DarkGray }

# ---------------------------------------------------------------------------
Section "1. environment"

$drv = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }).DriverVersion
Note "driver: $drv"

$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin"
if (Test-Path $cudaBin) { $env:PATH = "$cudaBin;$cudaBin\x64;$env:PATH" }
$nvccVer = (& nvcc --version 2>&1 | Select-String "release").ToString().Trim()
Note $nvccVer

$prof = try {
    (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak' `
        -Name RmProfilingAdminOnly -ErrorAction Stop).RmProfilingAdminOnly
} catch { "unset" }
if ($prof -eq 0) { Pass "RmProfilingAdminOnly = 0 (ncu should work)" }
else { Fail "RmProfilingAdminOnly = $prof -- ncu will hit ERR_NVGPUCTRPERM" }
$results["gpu counters enabled"] = ($prof -eq 0)

$pending = (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending')
Note "still reboot-pending: $pending"

# ---------------------------------------------------------------------------
if (-not $SkipBuild) {
    Section "2. build"
    Push-Location $root
    $vc = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    cmd /c "call `"$vc`" >nul 2>&1 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release >nul 2>&1 && cmake --build build 2>&1" |
        Select-String -Pattern "error|FAILED" | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    $built = ($LASTEXITCODE -eq 0)
    Pop-Location
    if ($built) { Pass "build clean" } else { Fail "build reported errors" }
    $results["build"] = $built
}

# ---------------------------------------------------------------------------
Section "3. ctest"
Push-Location (Join-Path $root "build")
$ct = & ctest --output-on-failure 2>&1 | Out-String
Pop-Location
if ($ct -match "100% tests passed") {
    $n = [regex]::Match($ct, "(\d+) tests passed out of (\d+)")
    Pass "all suites passed ($($n.Groups[2].Value) suites)"
    $results["ctest"] = $true
} else {
    Fail "ctest reported failures"
    $ct -split "`n" | Select-String "Failed|\*\*\*" | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    $results["ctest"] = $false
}

# ---------------------------------------------------------------------------
Section "4. benchmarks"
$bin = Join-Path $root "build\bin"
$benches = @('bench_inference','bench_image','bench_hash_kv','bench_spatial',
             'bench_video','bench_audio','bench_mc','bench_graph')
$benchOk = $true
foreach ($b in $benches) {
    $exe = Join-Path $bin "$b.exe"
    if (-not (Test-Path $exe)) { Fail "$b missing"; $benchOk = $false; continue }
    $out = & $exe 2>&1 | Out-String
    $code = $LASTEXITCODE
    # Case-SENSITIVE (-cmatch): these markers are printed in caps by the
    # benchmarks themselves. A case-insensitive match also hits the word
    # "wrong" in ordinary explanatory prose and reports false failures.
    $bad = ($out -cmatch "WRONG|MISMATCH")
    if ($code -eq 0 -and -not $bad) { Pass "$b" }
    else { Fail "$b (exit $code$(if($bad){', bad output'}))"; $benchOk = $false }
}
$results["benchmarks"] = $benchOk

# ---------------------------------------------------------------------------
if (-not $SkipProfile) {
    Section "5. nsys timelines"
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "profile.ps1") -SkipNcu |
        Select-String "wrote|WARNING" | ForEach-Object { Note $_ }
    $reps = @(Get-ChildItem (Join-Path $root "profiles") -Filter *.nsys-rep -ErrorAction SilentlyContinue)
    if ($reps.Count -ge 8) { Pass "$($reps.Count) timelines" } else { Fail "only $($reps.Count) timelines" }
    $results["nsys"] = ($reps.Count -ge 8)

    Section "6. ncu metrics (the post-reboot check)"
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "profile.ps1") -SkipNsys |
        Select-String "ERR_NVGPUCTRPERM|ncu metrics" | ForEach-Object { Note $_ }
    $csvs = @(Get-ChildItem (Join-Path $root "profiles") -Filter *_ncu.csv -ErrorAction SilentlyContinue |
              Where-Object { $_.Length -gt 200 })
    if ($csvs.Count -ge 1) {
        Pass "$($csvs.Count) ncu CSVs with real data"
        $results["ncu"] = $true
    } else {
        Fail "no ncu data -- counters still blocked?"
        $results["ncu"] = $false
    }
}

# ---------------------------------------------------------------------------
Section "summary"
$allOk = $true
foreach ($k in $results.Keys) {
    $v = $results[$k]
    if ($v) { Pass $k } else { Fail $k; $allOk = $false }
}
Write-Host ""
if ($allOk) { Write-Host "EVERYTHING VERIFIED" -ForegroundColor Green }
else { Write-Host "SOME CHECKS FAILED (see above)" -ForegroundColor Red }
