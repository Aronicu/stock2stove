import argparse
import os
import shutil
import socket
import subprocess
import sys


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def get_venv_python(venv_dir):
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def find_npm():
    if os.name == "nt":
        return shutil.which("npm.cmd") or shutil.which("npm")
    return shutil.which("npm")


def get_lan_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def run_cmd(cmd, cwd=None, env=None):
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def ensure_venv(project_root):
    venv_dir = os.path.join(project_root, ".venv")
    venv_python = get_venv_python(venv_dir)
    if not os.path.isfile(venv_python):
        print(f"[INIT] Creating virtual environment at {venv_dir}")
        run_cmd([sys.executable, "-m", "venv", venv_dir], cwd=project_root)
    return venv_python


def ensure_cryptography(venv_python, project_root):
    check = subprocess.run(
        [venv_python, "-c", "import cryptography"],
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check.returncode != 0:
        print("[INIT] Installing cryptography for self-signed certificate generation")
        run_cmd([venv_python, "-m", "pip", "install", "cryptography"], cwd=project_root)


def install_python_deps(project_root, venv_python):
    req_path = os.path.join(project_root, "requirements.txt")
    print("[INIT] Upgrading pip and installing Python dependencies")
    run_cmd([venv_python, "-m", "pip", "install", "--upgrade", "pip"], cwd=project_root)
    if os.path.isfile(req_path):
        run_cmd([venv_python, "-m", "pip", "install", "-r", req_path], cwd=project_root)
    ensure_cryptography(venv_python, project_root)


def install_frontend_deps(project_root):
    npm_bin = find_npm()
    if not npm_bin:
        raise RuntimeError("npm was not found in PATH.")
    frontend_dir = os.path.join(project_root, "frontend")
    print("[INIT] Installing frontend dependencies")
    run_cmd([npm_bin, "install"], cwd=frontend_dir)


def build_frontend(project_root):
    frontend_dir = os.path.join(project_root, "frontend")
    node_modules = os.path.join(frontend_dir, "node_modules")
    if not os.path.isdir(node_modules):
        install_frontend_deps(project_root)
    npm_bin = find_npm()
    if not npm_bin:
        raise RuntimeError("npm was not found in PATH.")
    print("[BUILD] Building frontend")
    run_cmd([npm_bin, "run", "build"], cwd=frontend_dir)


def ensure_self_signed_cert(project_root, venv_python, lan_ip):
    certs_dir = os.path.join(project_root, ".certs")
    cert_path = os.path.join(certs_dir, "cert.pem")
    key_path = os.path.join(certs_dir, "key.pem")
    if os.path.isfile(cert_path) and os.path.isfile(key_path):
        return cert_path, key_path

    os.makedirs(certs_dir, exist_ok=True)
    ensure_cryptography(venv_python, project_root)

    hostname = socket.gethostname()
    cert_script = (
        "import datetime, ipaddress, sys; "
        "from cryptography import x509; "
        "from cryptography.hazmat.primitives import hashes, serialization; "
        "from cryptography.hazmat.primitives.asymmetric import rsa; "
        "from cryptography.x509.oid import NameOID; "
        "cert_path, key_path, lan_ip, hostname = sys.argv[1:5]; "
        "key = rsa.generate_private_key(public_exponent=65537, key_size=2048); "
        "name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'stock2stove-dev')]); "
        "dns_names = ['localhost']; "
        "dns_names.append(hostname) if hostname and hostname not in dns_names else None; "
        "ips = ['127.0.0.1']; "
        "ips.append(lan_ip) if lan_ip and lan_ip not in ips else None; "
        "san_entries = [x509.DNSName(d) for d in dns_names] + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]; "
        "now = datetime.datetime.now(datetime.timezone.utc); "
        "cert = x509.CertificateBuilder()"
        ".subject_name(name)"
        ".issuer_name(name)"
        ".public_key(key.public_key())"
        ".serial_number(x509.random_serial_number())"
        ".not_valid_before(now)"
        ".not_valid_after(now + datetime.timedelta(days=365))"
        ".add_extension(x509.SubjectAlternativeName(san_entries), critical=False)"
        ".sign(key, hashes.SHA256()); "
        "open(key_path, 'wb').write(key.private_bytes("
        "encoding=serialization.Encoding.PEM, "
        "format=serialization.PrivateFormat.TraditionalOpenSSL, "
        "encryption_algorithm=serialization.NoEncryption())); "
        "open(cert_path, 'wb').write(cert.public_bytes(serialization.Encoding.PEM))"
    )
    print(f"[CERT] Generating self-signed development certificate in {certs_dir}")
    run_cmd(
        [venv_python, "-c", cert_script, cert_path, key_path, lan_ip, hostname],
        cwd=project_root,
    )
    return cert_path, key_path


def cmd_init(project_root):
    venv_python = ensure_venv(project_root)
    install_python_deps(project_root, venv_python)
    install_frontend_deps(project_root)
    lan_ip = get_lan_ip()
    ensure_self_signed_cert(project_root, venv_python, lan_ip)
    print("[INIT] Setup complete")


def cmd_build(project_root):
    build_frontend(project_root)
    print("[BUILD] Frontend build complete")


def cmd_run(project_root, host, port):
    venv_dir = os.path.join(project_root, ".venv")
    venv_python = get_venv_python(venv_dir)
    if not os.path.isfile(venv_python):
        venv_python = ensure_venv(project_root)
        install_python_deps(project_root, venv_python)

    dist_index = os.path.join(project_root, "frontend", "dist", "index.html")
    if not os.path.isfile(dist_index):
        build_frontend(project_root)

    lan_ip = get_lan_ip()
    cert_path, key_path = ensure_self_signed_cert(project_root, venv_python, lan_ip)

    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    env["FORWARDED_ALLOW_IPS"] = "*"

    print(f"[SERVER] Local URL: https://localhost:{port}")
    if lan_ip != "127.0.0.1":
        print(f"[SERVER] LAN URL:   https://{lan_ip}:{port}")
    print("[SERVER] Note: Accept the self-signed certificate warning in your browser for camera access.")

    try:
        run_cmd(
            [
                venv_python,
                "-m",
                "uvicorn",
                "src.server.app:app",
                "--host",
                host,
                "--port",
                str(port),
                "--ssl-keyfile",
                key_path,
                "--ssl-certfile",
                cert_path,
                "--proxy-headers",
            ],
            cwd=project_root,
            env=env,
        )
    except KeyboardInterrupt:
        print("\n[SERVER] Stopped")


def main():
    parser = argparse.ArgumentParser(description="Stock2Stove setup, build, and HTTPS LAN runner")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "build", "init"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    project_root = get_project_root()
    if args.command == "init":
        cmd_init(project_root)
    elif args.command == "build":
        cmd_build(project_root)
    else:
        cmd_run(project_root, args.host, args.port)


if __name__ == "__main__":
    main()
