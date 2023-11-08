#!/bin/bash -e

while true; do
  http_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "h2oai/h2ogpt-4096-llama2-13b-chat",
      "prompt": "San Francisco is a",
      "max_tokens": 7,
      "temperature": 0
    }')

  if [ "$http_code" -eq 200 ]; then
    echo "Received HTTP 200 status code. Restarting Nginx for h2oGPT"
    ip=$(dig +short myip.opendns.com @resolver1.opendns.com)
    sed "s/<|_SUBST_PUBLIC_IP|>;/$ip;/g" /workspace/temp.conf  > /etc/nginx/conf.d/h2ogpt.conf
    sudo systemctl restart nginx.service
    break
  else
    echo "Received HTTP $http_code status code. Retrying in 5 seconds..."
    sleep 5
  fi
done
