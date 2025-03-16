import requests
import json

headers = {'Content-Type': 'application/json'}

gpt = {"text": "　“是啊，无法预知，所以更要处处小心，少做少错啊，高师弟他们不轻举妄动，就不至于造此劫数。”", "speaker": "jok老师", "streaming": 0}

response = requests.post("http://localhost:9880/", data=json.dumps(gpt), headers=headers)

audio_data = response.content
print(audio_data)
with open(f"post请求测试.wav", "wb") as f:
    f.write(audio_data)
