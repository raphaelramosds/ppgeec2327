param(
    [Parameter(Position = 0)]
    [string]$Script
)

if ([string]::IsNullOrWhiteSpace($Script)) {
    Write-Host "Uso: .\matlab.ps1 <arquivo.m>"
    Write-Host "Exemplo: .\matlab.ps1 .\Exercicio4-caps8-9\test_example_CNN.m"
    exit 1
}

if (-not (Test-Path $Script -PathType Leaf)) {
    Write-Error "File not found: $Script"
    exit 1
}

# Escape single quotes for the MATLAB string literal
$path = $Script.Replace("'", "''")

# try/catch guarantees MATLAB exits even if the script errors
$cmd = "try, run('$path'); pause; catch e, disp(getReport(e)); pause; exit(1); end; exit(0);"

# -wait keeps the console attached until MATLAB finishes (Windows launcher returns immediately otherwise)
& matlab -nodisplay -nosplash -nodesktop -wait -r $cmd
exit $LASTEXITCODE
