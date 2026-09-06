#!/usr/bin/env bash
# One-time cutover: serial mythd  ->  gateway mythd + batched llama-server.
# Safe: aborts (leaving you to roll back) if llama-server doesn't come up.
set -u

echo "=== install llama-server unit + mythd ordering drop-in ==="
sudo -n cp ~/mythd/deploy/llama-server.service /etc/systemd/system/llama-server.service
sudo -n mkdir -p /etc/systemd/system/mythd.service.d
printf '[Unit]\nAfter=llama-server.service\nWants=llama-server.service\n' \
  | sudo -n tee /etc/systemd/system/mythd.service.d/10-gateway.conf >/dev/null
sudo -n systemctl daemon-reload
sudo -n systemctl enable llama-server >/dev/null 2>&1 || true

echo "=== CUTOVER START — lightning is DOWN from here ==="
sudo -n systemctl stop mythd
sleep 2

echo "=== start batched llama-server (loads 18GB model + KV pool on the B70) ==="
sudo -n systemctl restart llama-server
printf "waiting for llama-server to load"
ready=0
for i in $(seq 1 90); do
  s=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8091/health 2>/dev/null || true)
  if [ "$s" = "200" ]; then ready=1; echo " READY (~$((i*2))s)"; break; fi
  printf "."; sleep 2
done
if [ "$ready" != "1" ]; then
  echo " TIMEOUT — llama-server did not become ready. Last journal lines:"
  sudo -n journalctl -u llama-server --no-pager | tail -30
  echo "!! ABORTING cutover. Roll back with: scripts/rollback-serial.sh"
  exit 1
fi
echo "llama-server health: $(curl -s http://127.0.0.1:8091/health)"

echo "=== start gateway mythd ==="
sudo -n systemctl restart mythd
for i in $(seq 1 15); do
  s=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/health 2>/dev/null || true)
  [ "$s" = "200" ] && break
  sleep 1
done
echo "mythd health: $(curl -s http://127.0.0.1:8090/health)"
echo "=== service states ==="
systemctl is-active llama-server mythd
echo "=== llama-server load summary (KV / slots / VRAM) ==="
sudo -n journalctl -u llama-server --no-pager \
  | grep -iE "n_ctx|n_parallel|KV (self|cache)|buffer size|kv_cache|slots|model loaded|server is listening|graphs" \
  | tail -25
echo "=== CUTOVER DONE ==="
