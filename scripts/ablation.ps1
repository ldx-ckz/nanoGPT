param(
    [ValidateSet('A', 'B', 'C', 'D')]
    [string]$Group,
    [string]$Resume,
    [switch]$Run
)

$ErrorActionPreference = 'Stop'

if (($Run -or $Resume) -and -not $Group) {
    throw 'Specify -Group A, B, C or D when using -Run or -Resume.'
}
if ($Run -and -not $IsLinux) {
    throw 'Run the two-GPU NCCL experiment on Linux with PowerShell 7.'
}

$repo = Split-Path -Parent $PSScriptRoot
$python = 'python'
if ($env:OS -eq 'Windows_NT') {
    $python = Join-Path $repo '.venv/Scripts/python.exe'
}
$trainer = Join-Path $repo 'GPT.py'
$dataDir = Join-Path $repo 'wikitext103'
$manifestPath = Join-Path $dataDir 'manifest.json'
$runRoot = Join-Path $repo 'runs/ablation/wikitext-seed1337'

$variants = [ordered]@{
    A = 'gpt2'
    B = 'mha'
    C = 'mla'
    D = 'dsa'
}

$gpuCount = 2
$batchSize = 4
$seqLen = 1024
$globalTokens = 16384
$requestedTokens = 50000000
$dsaTopk = 128
$dsaWarmupSteps = 153
$tokensPerMicroStep = $gpuCount * $batchSize * $seqLen
if ($globalTokens % $tokensPerMicroStep -ne 0) {
    throw 'Global token batch must be divisible by GPUs * batch size * sequence length.'
}
$gradAccumSteps = [int]($globalTokens / $tokensPerMicroStep)
$plannedSteps = [int][math]::Ceiling($requestedTokens / $globalTokens)
$plannedTokens = $plannedSteps * $globalTokens
if ($dsaTopk -ge $seqLen -or $dsaWarmupSteps -ge $plannedSteps) {
    throw 'DSA needs top-k below sequence length and warmup shorter than the run.'
}

$launcherArgs = @('-u', '-m', 'torch.distributed.run', '--standalone', "--nproc-per-node=$gpuCount", $trainer)
$commonArgs = @(
    '--data-dir', $dataDir,
    '--batch-size', "$batchSize",
    '--seq-len', "$seqLen",
    '--total-batch-size', "$globalTokens",
    '--train-tokens', "$requestedTokens",
    '--max-lr', '6e-4',
    '--seed', '1337',
    '--eval-interval', '100',
    '--val-batches', '20',
    '--checkpoint-interval', '500'
)

Write-Host "DDP: $gpuCount GPUs, $batchSize sequences/GPU, $gradAccumSteps accumulation steps."
Write-Host "Budget: $plannedSteps updates, $plannedTokens training tokens per group."
$selectedGroups = if ($Group) { @($Group) } else { $variants.Keys }

foreach ($id in $selectedGroups) {
    $attention = $variants[$id]
    $outputDir = Join-Path $runRoot "$id-$attention"
    $trainArgs = $commonArgs + @(
        '--attention', $attention,
        '--output-dir', $outputDir
    )
    if ($id -eq 'D') {
        $trainArgs += @(
            '--dsa-topk', "$dsaTopk",
            '--dsa-warmup-steps', "$dsaWarmupSteps",
            '--dsa-loss-weight', '1.0'
        )
    }
    if ($Resume) {
        $trainArgs += @('--resume', $Resume)
    }

    $launchArgs = $launcherArgs + $trainArgs
    $quotedArgs = $launchArgs | ForEach-Object { '"{0}"' -f $_ }
    Write-Host ('$env:TORCH_COMPILE_DISABLE = "1"')
    Write-Host ('& "{0}" {1}' -f $python, ($quotedArgs -join ' '))
    if (-not $Run) {
        continue
    }

    if ($Resume -and -not (Test-Path -LiteralPath $Resume -PathType Leaf)) {
        throw "Checkpoint does not exist: $Resume"
    }
    if (-not $Resume -and (Test-Path -LiteralPath $outputDir)) {
        throw "Run directory already exists: $outputDir"
    }

    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    $commit = git -C $repo rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Could not read Git commit.' }
    $gitStatus = @(git -C $repo status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw 'Could not read Git status.' }

    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    $metadataName = if ($Resume) { 'experiment.resume.json' } else { 'experiment.json' }
    [ordered]@{
        group = $id
        attention = $attention
        git_commit = $commit
        git_status = $gitStatus
        dataset = $manifest.dataset
        dataset_revision = $manifest.revision
        tokenizer = $manifest.tokenizer
        manifest_sha256 = (Get-FileHash -LiteralPath $manifestPath).Hash
        processes = $gpuCount
        gradient_accumulation_steps = $gradAccumSteps
        planned_steps = $plannedSteps
        planned_training_tokens = $plannedTokens
        cuda_visible_devices = $env:CUDA_VISIBLE_DEVICES
        compile_disabled = $true
        python = $python
        arguments = $launchArgs
    } | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath (Join-Path $outputDir $metadataName) -Encoding UTF8

    $previousCompileSetting = $env:TORCH_COMPILE_DISABLE
    try {
        $env:TORCH_COMPILE_DISABLE = '1'
        & $python @launchArgs
        if ($LASTEXITCODE -ne 0) { throw "Training failed for group $id." }
    } finally {
        $env:TORCH_COMPILE_DISABLE = $previousCompileSetting
    }
}
