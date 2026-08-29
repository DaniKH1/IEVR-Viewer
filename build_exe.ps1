$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
python -m PyInstaller --noconfirm --clean --windowed --onefile --collect-all pyglet --add-data "app_icon.png;." --icon app_icon.ico --name "IEVR Viewer" g4tx_extractor.py
Write-Host "Created: $here\dist\IEVR Viewer.exe"
