#!/usr/bin/env bash
set -euo pipefail

BIND_HOST="0.0.0.0"
PORT="8000"
SKIP_ADB_REVERSE=0
BUILD_FRONTEND=0
SKIP_FRONTEND=0
DEV_MODE=0

print_usage() {
    echo "Usage: ./run_server.sh [options]"
    echo "  --host <addr>         Bind host (default: 0.0.0.0)"
    echo "  --port <port>         Bind port (default: 8000)"
    echo "  --skip-adb-reverse    Skip ADB USB reverse port forwarding"
    echo "  --build-frontend      Force rebuild of frontend/dist"
    echo "  --skip-frontend       Skip frontend dependency check and build"
    echo "  --dev                 Run FastAPI backend and Vite dev server together"
    echo "  -h, --help            Show this help message"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --host)
            BIND_HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --skip-adb-reverse)
            SKIP_ADB_REVERSE=1
            shift
            ;;
        --build-frontend)
            BUILD_FRONTEND=1
            shift
            ;;
        --skip-frontend)
            SKIP_FRONTEND=1
            shift
            ;;
        --dev)
            DEV_MODE=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
export FORWARDED_ALLOW_IPS="*"

BASE_PYTHON=""
for CANDIDATE in python3.12 python3.11 python3.10 python3.13 python3; do
    if command -v "$CANDIDATE" >/dev/null 2>&1; then
        BASE_PYTHON="$(command -v "$CANDIDATE")"
        break
    fi
done

if [ -z "$BASE_PYTHON" ]; then
    echo "No Python 3 interpreter found." >&2
    exit 1
fi

PYTHON_BIN=""
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
    if [ -x "$PROJECT_ROOT/venv/bin/python" ]; then
        VENV_VER="$("$PROJECT_ROOT/venv/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
        BASE_VER="$("$BASE_PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
        if [ "$VENV_VER" != "$BASE_VER" ] && ! "$PROJECT_ROOT/venv/bin/python" -c "import uvicorn" >/dev/null 2>&1; then
            echo "[SETUP] Recreating virtual environment with Python $BASE_VER"
            rm -rf "$PROJECT_ROOT/venv"
        fi
    fi

    if [ ! -x "$PROJECT_ROOT/venv/bin/python" ]; then
        echo "[SETUP] Creating virtual environment in $PROJECT_ROOT/venv using $BASE_PYTHON"
        "$BASE_PYTHON" -m venv "$PROJECT_ROOT/venv"
    fi
    PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
fi

if ! "$PYTHON_BIN" -c "import uvicorn" >/dev/null 2>&1; then
    echo "[SETUP] Installing Python dependencies from requirements.txt"
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt"
fi

FRONTEND_DIR="$PROJECT_ROOT/frontend"
FRONTEND_PKG="$FRONTEND_DIR/package.json"
FRONTEND_DIST_INDEX="$FRONTEND_DIR/dist/index.html"

if [ "$SKIP_FRONTEND" -eq 0 ] && [ -f "$FRONTEND_PKG" ]; then
    if ! command -v npm >/dev/null 2>&1; then
        if [ "$DEV_MODE" -eq 1 ] || [ ! -f "$FRONTEND_DIST_INDEX" ]; then
            echo "npm is required to build or run the frontend." >&2
            exit 1
        fi
        echo "[FRONTEND] npm not found; using existing frontend/dist/index.html"
    else
        if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
            echo "[FRONTEND] Installing npm dependencies in frontend/"
            (cd "$FRONTEND_DIR" && npm install)
        fi

        if [ "$DEV_MODE" -eq 0 ]; then
            NEED_BUILD=0
            if [ "$BUILD_FRONTEND" -eq 1 ] || [ ! -f "$FRONTEND_DIST_INDEX" ]; then
                NEED_BUILD=1
            else
                if [ -d "$FRONTEND_DIR/src" ] && find "$FRONTEND_DIR/src" -type f -newer "$FRONTEND_DIST_INDEX" | grep -q .; then
                    NEED_BUILD=1
                elif [ -f "$FRONTEND_DIR/index.html" ] && [ "$FRONTEND_DIR/index.html" -nt "$FRONTEND_DIST_INDEX" ]; then
                    NEED_BUILD=1
                elif [ "$FRONTEND_PKG" -nt "$FRONTEND_DIST_INDEX" ]; then
                    NEED_BUILD=1
                elif [ -f "$FRONTEND_DIR/vite.config.js" ] && [ "$FRONTEND_DIR/vite.config.js" -nt "$FRONTEND_DIST_INDEX" ]; then
                    NEED_BUILD=1
                fi
            fi

            if [ "$NEED_BUILD" -eq 1 ]; then
                echo "[FRONTEND] Building React SPA in frontend/dist"
                (cd "$FRONTEND_DIR" && npm run build)
            fi
        fi
    fi
fi

if [ "$SKIP_ADB_REVERSE" -eq 0 ]; then
    ADB_BIN=""
    if command -v adb >/dev/null 2>&1; then
        ADB_BIN="$(command -v adb)"
    elif [ -x "$HOME/Library/Android/sdk/platform-tools/adb" ]; then
        ADB_BIN="$HOME/Library/Android/sdk/platform-tools/adb"
    elif [ -x "$HOME/Android/Sdk/platform-tools/adb" ]; then
        ADB_BIN="$HOME/Android/Sdk/platform-tools/adb"
    fi

    if [ -n "$ADB_BIN" ]; then
        if "$ADB_BIN" reverse "tcp:${PORT}" "tcp:${PORT}" >/dev/null 2>&1; then
            echo "[ADB] USB reverse tunnel active on tcp:${PORT}"
        else
            echo "[ADB] No USB Android device detected; continuing with LAN/localhost mode."
        fi
    fi
fi

echo "[SERVER] Local UI: http://localhost:${PORT}"
echo "[SERVER] Starting Stock-to-Stove FastAPI Server on http://${BIND_HOST}:${PORT}"

if [ "$DEV_MODE" -eq 1 ] && [ "$SKIP_FRONTEND" -eq 0 ] && [ -f "$FRONTEND_PKG" ]; then
    BACKEND_PID=""
    FRONTEND_PID=""
    cleanup() {
        if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
            kill "$FRONTEND_PID" >/dev/null 2>&1 || true
        fi
        if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
            kill "$BACKEND_PID" >/dev/null 2>&1 || true
        fi
    }
    trap cleanup EXIT INT TERM

    "$PYTHON_BIN" -m uvicorn src.server.app:app --host "$BIND_HOST" --port "$PORT" --proxy-headers &
    BACKEND_PID="$!"
    echo "[DEV] Starting Vite SPA dev server in frontend/ (http://localhost:5173)"
    (cd "$FRONTEND_DIR" && npm run dev) &
    FRONTEND_PID="$!"
    wait "$BACKEND_PID" "$FRONTEND_PID"
    exit 0
fi

exec "$PYTHON_BIN" -m uvicorn src.server.app:app --host "$BIND_HOST" --port "$PORT" --proxy-headers
