#!/usr/bin/env bash
# ==============================================================================
# NeuroQuant 3.0 | The Sovereign Quant Launch Script
# Starts FastAPI Backend (Port 8000) and Vite React Frontend (Port 5173)
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "================================================================="
echo "   _  __                     ____                  _   "
echo "  / |/ /__ __ __ ____ ___   / __ \\__ _____ _ ___  / /_ "
echo " /    // // // // __// _ \\ / /_/ // // // _ \`/ _ \\/ __/ "
echo "/_/|_/ \\_,_/ \\_,_/_/  \\___/ \\___\\_\\\\_,_/ \\_,_/_//_/\\__/  "
echo "        Institutional Intelligence & 10k Simulation Lab"
echo "================================================================="

# 1. Python Sanal Ortam Kontrolü
if [ -d ".venv" ]; then
    echo "[✓] Python sanal ortamı (.venv) aktif ediliyor..."
    source .venv/bin/activate
else
    echo "[!] .venv bulunamadı! Lütfen önce python -m venv .venv kurun."
    exit 1
fi

# 2. Port Temizliği (Önceki oturumlardan kalan süreçler varsa)
cleanup_ports() {
    echo "[*] Portlar taranıyor (8000, 5173)..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:5173 | xargs kill -9 2>/dev/null || true
}
cleanup_ports

# Trap ile Çıkışta Süreçleri Sonlandırma
cleanup() {
    echo ""
    echo "[!] NeuroQuant kapatılıyor..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 3. FastAPI Backend Başlatma
echo "[*] FastAPI Asenkron Backend başlatılıyor (http://127.0.0.1:8000)..."
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Backend Sağlık Kontrolü Beklemesi
echo "[*] Backend API hazır olana kadar bekleniyor..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/api/health > /dev/null; then
        echo "[✓] Backend API aktif ve sağlıklı!"
        break
    fi
    sleep 0.5
done

# 4. Vite React Frontend Başlatma
echo "[*] Vite React Frontend başlatılıyor (http://127.0.0.1:5173)..."
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173 &
FRONTEND_PID=$!
cd ..

echo "================================================================="
echo " [✓] NeuroQuant 3.0 Platformu Başarıyla Başlatıldı!"
echo "     - Web Terminali : http://127.0.0.1:5173"
echo "     - Backend API   : http://127.0.0.1:8000"
echo "     - API Dokümanı  : http://127.0.0.1:8000/docs"
echo "================================================================="
echo " Çıkmak için CTRL+C tuşlayınız."

# Süreçleri ayakta tut
wait
