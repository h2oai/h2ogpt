#!/bin/bash
set -o pipefail
set -ex

#
#* Deal with not-thread-safe things in LangChain:
#
sp=`python3.10 -c 'import site; print(site.getsitepackages()[0])'`
sed -i  's/with HiddenPrints():/if True:/g' $sp/langchain_community/utilities/serpapi.py
#sed -i 's/"progress": Status.PROGRESS,/"progress": Status.PROGRESS,\n            "heartbeat": Status.PROGRESS,/g' gradio_client/utils.py
#sed -i 's/async for line in response.aiter_text():/async for line in response.aiter_lines():\n                if len(line) == 0:\n                    continue\n                if line == """{"detail":"Not Found"}""":\n                    continue/g' gradio_client/utils.py


# fix pytube to avoid errors for restricted content
sed -i "s/client='ANDROID_MUSIC'/client='ANDROID'/g" $sp/pytube/innertube.py
# https://github.com/JuanBindez/pytubefix/commit/c0c07b046d8b59574552404931f6ce3c6590137d
sed -i "s/17.31.35/19.08.35/g" $sp/pytube/innertube.py
sed -i "s/17.33.2/19.08.35/g" $sp/pytube/innertube.py
sed -i "s/17.31.35/19.08.35/g" $sp/pytube/innertube.py
sed -i "s/17.33.2/19.08.35/g" $sp/pytube/innertube.py
sed -i "s/5.16.51/6.40.52/g" $sp/pytube/innertube.py
sed -i "s/5.21/6.41/g" $sp/pytube/innertube.py


# fix asyncio same way websockets was fixed, else keep hitting errors in async calls
# https://github.com/python-websockets/websockets/commit/f9fd2cebcd42633ed917cd64e805bea17879c2d7
sed -i "s/except OSError:/except (OSError, RuntimeError):/g" $sp/anyio/_backends/_asyncio.py

# https://github.com/gradio-app/gradio/issues/7086
sed -i 's/while True:/while True:\n            time.sleep(0.001)\n/g' $sp/gradio_client/client.py
