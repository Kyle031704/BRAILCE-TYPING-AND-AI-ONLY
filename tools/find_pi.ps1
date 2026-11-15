# Auto-detecting Pi Scanner compatible with Windows PowerShell 5.1 and 7
$ErrorActionPreference = "SilentlyContinue"

# Auto-detect IPv4 and Subnet
$ipInfo = ipconfig | Select-String "IPv4 Address"
$ipList = @()
foreach ($line in $ipInfo) {
    if ($line -match '(\d{1,3}\.\d{1,3}\.\d{1,3})\.(\d{1,3})') {
        $ipList += [PSCustomObject]@{
            Subnet = "$($matches[1])."
            FullIP = "$($matches[1]).$($matches[2])"
        }
    }
}

if ($ipList.Count -eq 0) {
    Write-Host "No active IPv4 network interface found." -ForegroundColor Red
    exit
}

$found = @()

foreach ($net in $ipList) {
    $subnet = $net.Subnet
    $myIp = $net.FullIP
    Write-Host "Scanning subnet: ${subnet}x (Your IP: $myIp)..." -ForegroundColor Cyan
    
    $tasks = 1..254 | ForEach-Object {
        $ip = "$subnet$_"
        if ($ip -ne $myIp) {
            $client = New-Object System.Net.Sockets.TcpClient
            $iar = $client.BeginConnect($ip, 22, $null, $null)
            [PSCustomObject]@{ IP = $ip; Client = $client; IAR = $iar }
        }
    }

    Start-Sleep -Milliseconds 1200

    foreach ($t in $tasks) {
        if ($t -and $t.IAR -and $t.IAR.IsCompleted) {
            try {
                $t.Client.EndConnect($t.IAR)
                Write-Host "`n========================================" -ForegroundColor Green
                Write-Host "  >>> YOUR RASPBERRY PI IP IS: $($t.IP) <<<" -ForegroundColor Green
                Write-Host "========================================`n" -ForegroundColor Green
                $found += $t.IP
            } catch {}
        }
        if ($t -and $t.Client) {
            $t.Client.Close()
        }
    }
}

if ($found.Count -eq 0) {
    Write-Host "`nNo SSH port 22 found on any local subnet." -ForegroundColor Yellow
}
