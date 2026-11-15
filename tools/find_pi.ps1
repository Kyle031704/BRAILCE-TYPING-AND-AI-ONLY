# Fast Pi Scanner compatible with Windows PowerShell 5.1 and 7
Write-Host "Scanning local network (192.168.0.x) for Raspberry Pi (excluding router 192.168.0.1)..." -ForegroundColor Cyan

$gateway = "192.168.0.1"
$myIp = "192.168.0.103"
$found = @()
$subnet = "192.168.0."

# Scan 2..254 (skip gateway .1)
$tasks = 2..254 | ForEach-Object {
    $ip = "$subnet$_"
    if ($ip -ne $myIp) {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect($ip, 22, $null, $null)
        [PSCustomObject]@{ IP = $ip; Client = $client; IAR = $iar }
    }
}

# Wait up to 1200ms for Wi-Fi responses
Start-Sleep -Milliseconds 1200

foreach ($t in $tasks) {
    if ($t.IAR.IsCompleted) {
        try {
            $t.Client.EndConnect($t.IAR)
            Write-Host "`n========================================" -ForegroundColor Green
            Write-Host "  >>> YOUR RASPBERRY PI IP IS: $($t.IP) <<<" -ForegroundColor Green
            Write-Host "========================================`n" -ForegroundColor Green
            $found += $t.IP
        } catch {}
    }
    $t.Client.Close()
}

if ($found.Count -eq 0) {
    Write-Host "No other SSH devices found. Showing active devices from ARP table:" -ForegroundColor Yellow
    arp -a | Select-String "192.168.0." | Where-Object { $_ -notmatch "192\.168\.0\.1\s" -and $_ -notmatch "192\.168\.0\.255" }
}

