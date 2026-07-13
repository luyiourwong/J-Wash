# Stop the J-Wash servers: backend (8381) and the Vite dev server (5173).
$ports = @(8381, 5173)
foreach ($port in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $conns) {
        $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "port ${port}: stopping $($proc.ProcessName) (PID $($proc.Id))"
            Stop-Process -Id $proc.Id -Force -Confirm:$false
        }
    }
    if (-not $conns) { Write-Host "port ${port}: nothing to stop" }
}
