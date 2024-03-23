Write-Output "Ending running h2o server processes..."
if(Get-NetTCPConnection -LocalPort "7860" -ErrorAction SilentlyContinue) { 
    Write-Output "Process running on port 7680. Ending Process.... (wait 10 seconds to end)"
    Get-Process -Id (Get-NetTCPConnection -LocalPort "7860").OwningProcess | Stop-Process
    Start-Sleep -Seconds 10
    if(Get-NetTCPConnection -LocalPort "7860") { Write-Error "Failed to stop process."}
}else{
    Write-Output "No processes running on port 7680"
}