#!/usr/bin/env bash
# Roll back: gateway mythd + llama-server  ->  the serial embedded mythd.
set -u
echo "=== stop gateway mythd + llama-server (frees VRAM) ==="
sudo -n systemctl stop mythd
sudo -n systemctl stop llama-server
sudo -n systemctl disable llama-server >/dev/null 2>&1 || true
echo "=== restore the serial binary ==="
cp ~/mythd/mythd-serial.bak ~/mythd/target/release/mythd
echo "=== remove the gateway ordering drop-in ==="
sudo -n rm -f /etc/systemd/system/mythd.service.d/10-gateway.conf
sudo -n systemctl daemon-reload
echo "=== restart serial mythd (reloads the model on the B70) ==="
sudo -n systemctl restart mythd
for i in $(seq 1 60); do
  s=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/health 2>/dev/null || true)
  [ "$s" = "200" ] && break
  sleep 2
done
echo "mythd health: $(curl -s http://127.0.0.1:8090/health)"
systemctl is-active mythd
echo "=== ROLLBACK DONE ==="
