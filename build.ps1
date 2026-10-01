# Builds function.zip for Lambda (no Docker). Run from the project folder:  .\build.ps1
$ErrorActionPreference = "Stop"

if (Test-Path build) { Remove-Item build -Recurse -Force }
if (Test-Path function.zip) { Remove-Item function.zip -Force }
New-Item -ItemType Directory build | Out-Null

# requests is the only third-party package we bundle (pandas/numpy come from the AWS layer, boto3 is built in)
python -m pip install requests --target build --upgrade

Copy-Item lambda_function.py, groq_client.py, model.py, ses_client.py build\

python -c "import shutil; shutil.make_archive('function', 'zip', 'build')"

Write-Host "Created function.zip"
Get-Item function.zip | Select-Object Name, Length
