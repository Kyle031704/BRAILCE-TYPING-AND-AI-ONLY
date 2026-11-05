import socket
import concurrent.futures

def check_ip(ip):
    s = socket.socket()
    s.settimeout(0.3)
    try:
        if s.connect_ex((ip, 22)) == 0:
            s.close()
            return ip
    except Exception:
        pass
    s.close()
    return None

def main():
    print("Testing hostnames...")
    for host in ['raspberrypi5.local', 'raspberrypi.local', 'raspberrypi5']:
        try:
            addr = socket.gethostbyname(host)
            print(f"  {host} resolved to {addr}")
            if check_ip(addr):
                print(f"  [SUCCESS] SSH open on {addr}")
        except Exception as e:
            print(f"  {host} resolution failed: {e}")

    print("Scanning hotspot subnet 192.168.137.x for SSH (port 22)...")
    ips_hotspot = [f"192.168.137.{i}" for i in range(2, 255)]
    found = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
        for res in ex.map(check_ip, ips_hotspot):
            if res:
                found.append(res)
                print(f"  [FOUND] SSH on {res}")

    print("Scanning Wi-Fi subnet 192.168.0.x for SSH (port 22)...")
    ips_wifi = [f"192.168.0.{i}" for i in range(2, 255)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
        for res in ex.map(check_ip, ips_wifi):
            if res:
                found.append(res)
                print(f"  [FOUND] SSH on {res}")

    print(f"Total SSH hosts found: {found}")

if __name__ == '__main__':
    main()

