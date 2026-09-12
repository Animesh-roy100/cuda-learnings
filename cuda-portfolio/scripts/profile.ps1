<#
.SYNOPSIS
  Generate Nsight Systems timelines and Nsight Compute roofline data.

.DESCRIPTION
  Writes into profiles/:
    <target>.nsys-rep   timeline: streams, CPU/GPU overlap, transfer gaps
    <target>_ncu.csv    per-kernel roofline + occupancy metrics

.NOTES
  Nsight Compute needs GPU performance counter access. The documented fix is

    reg add "HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" `
        /v RmProfilingAdminOnly /t REG_DWORD /d 0 /f

  followed by a reboot -- but on recent Windows drivers that key is often NOT
  honoured. Verified on this machine: key set to DWORD 0, rebooted, and ncu
  still returned ERR_NVGPUCTRPERM unelevated while succeeding elevated.

  So this script runs ncu elevated instead, in ONE batch, which means a single
  UAC prompt rather than one per target. Nsight Systems needs no elevation.

.EXAMPLE
  .\scripts\profile.ps1                    # everything
  .\scripts\profile.ps1 -Target bench_mc   # one target
  .\scripts\profile.ps1 -SkipNcu           # timelines only, no UAC prompt
#>
[CmdletBinding()]
param(
    [string]$Target = "",
    [string]$BuildDir = "build",
    [switch]$SkipNsys,
    [switch]$SkipNcu
)

# 'Continue', deliberately. Under 'Stop', Windows PowerShell turns ANY stderr
# line from a native tool into a terminating NativeCommandError -- so nsys
# printing a benign "CPU context switch trace requires admin" warning would
# abort the whole run. Native failures are detected via $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$binDir = Join-Path $root "$BuildDir\bin"
$outDir = Join-Path $root "profiles"

if (-not (Test-Path $binDir)) {
    Write-Error "No build at $binDir. Configure and build first."
    exit 1
}
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

function Find-Tool([string]$leaf, [string[]]$globs) {
    $cmd = Get-Command $leaf -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($g in $globs) {
        $hit = Get-ChildItem -Path $g -ErrorAction SilentlyContinue |
               Sort-Object FullName -Descending | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

$nsys = Find-Tool "nsys" @(
    "C:\Program Files\NVIDIA Corporation\Nsight Systems *\target-windows-x64\nsys.exe")
$ncu = Find-Tool "ncu" @(
    "C:\Program Files\NVIDIA Corporation\Nsight Compute *\ncu.bat",
    "C:\Program Files\NVIDIA Corporation\Nsight Compute *\ncu.exe")

$targets = if ($Target) { @($Target) } else {
    @("bench_inference", "bench_image", "bench_hash_kv", "bench_spatial",
      "bench_video", "bench_audio", "bench_mc", "bench_graph")
}

# Roofline needs achieved compute and DRAM traffic; occupancy explains a kernel
# sitting below the roof.
$metrics = @(
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem"
) -join ","

# ---------------------------------------------------------------------------
# Nsight Systems -- no elevation needed
# ---------------------------------------------------------------------------
if (-not $SkipNsys) {
    if (-not $nsys) {
        Write-Warning "nsys not found; skipping timelines"
    } else {
        foreach ($t in $targets) {
            $exe = Join-Path $binDir "$t.exe"
            if (-not (Test-Path $exe)) { Write-Warning "skip $t (not built)"; continue }
            $rep = Join-Path $outDir $t
            # cuda,nvtx only: 'osrt' is a Linux-only trace source and nsys
            # rejects the whole invocation if it is passed on Windows.
            & $nsys profile --force-overwrite true --output $rep `
                --trace cuda,nvtx --sample none $exe 2>$null | Out-Null
            if (Test-Path "$rep.nsys-rep") {
                $kb = [int]((Get-Item "$rep.nsys-rep").Length / 1KB)
                Write-Host "  nsys  $t.nsys-rep ($kb KB)" -ForegroundColor DarkGray
            } else {
                Write-Warning "  nsys produced nothing for $t (exit $LASTEXITCODE)"
            }
        }
    }
}

# ---------------------------------------------------------------------------
# Nsight Compute -- one elevated batch for every target
# ---------------------------------------------------------------------------
if (-not $SkipNcu) {
    if (-not $ncu) {
        Write-Warning "ncu not found; skipping metrics"
    } else {
        $present = @($targets | Where-Object { Test-Path (Join-Path $binDir "$_.exe") })
        if ($present.Count -eq 0) {
            Write-Warning "no built targets to profile"
        } else {
            # Build a helper script and run it once with elevation. Invoking ncu
            # per target with -Verb RunAs would raise one UAC prompt each.
            $helper = Join-Path $env:TEMP "ncu_batch_$PID.ps1"
            $lines = @(
                '$ErrorActionPreference = "Continue"',
                "`$ncu = '$ncu'",
                "`$bin = '$binDir'",
                "`$out = '$outDir'",
                "`$metrics = '$metrics'",
                "`$targets = @('" + ($present -join "','") + "')",
                'foreach ($t in $targets) {',
                '    $exe = Join-Path $bin "$t.exe"',
                '    $csv = Join-Path $out "${t}_ncu.csv"',
                # --launch-count bounds the work: these benchmarks launch the
                # same kernels thousands of times and ncu replays each one.
                '    & $ncu --csv --page raw --target-processes all --launch-count 8 `',
                '        --metrics $metrics $exe 2>&1 | Set-Content -Encoding utf8 $csv',
                '}'
            )
            Set-Content -Path $helper -Value $lines -Encoding UTF8

            Write-Host "  ncu: launching elevated batch for $($present.Count) target(s)" -ForegroundColor Yellow
            Write-Host "       (expect one UAC prompt)" -ForegroundColor Yellow
            $p = Start-Process -FilePath "powershell.exe" `
                    -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File",$helper `
                    -Verb RunAs -Wait -PassThru
            Remove-Item $helper -ErrorAction SilentlyContinue

            foreach ($t in $present) {
                $csv = Join-Path $outDir "${t}_ncu.csv"
                if ((Test-Path $csv) -and (Select-String -Path $csv -Pattern "ERR_NVGPUCTRPERM" -Quiet)) {
                    Write-Warning "  ncu  $t : counters still blocked"
                } elseif ((Test-Path $csv) -and ((Get-Item $csv).Length -gt 200)) {
                    $rows = (Get-Content $csv | Measure-Object -Line).Lines
                    Write-Host "  ncu   ${t}_ncu.csv ($rows rows)" -ForegroundColor DarkGray
                } else {
                    Write-Warning "  ncu  $t : no data (elevated exit $($p.ExitCode))"
                }
            }
        }
    }
}

Write-Host "`nArtifacts in $outDir" -ForegroundColor Green
Write-Host "Open a timeline with:  nsys-ui profiles\<target>.nsys-rep"
