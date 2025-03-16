import os
import sys

from fastapi import FastAPI, Request, HTTPException
from cosyvoice.cli.cosyvoice import CosyVoice
import torch
import torchaudio
import uvicorn
import asyncio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
app = FastAPI()

# 初始化CosyVoice模型
model_dir = "pretrained_models/CosyVoice2-0.5B"  # 替换为实际的模型目录路径
try:
    cosy_voice = CosyVoice(model_dir)
except Exception:
    try:
        cosy_voice = CosyVoice2(model_dir)
    except Exception:
        raise TypeError('no valid model_type!')


@app.post('/synthesize')
async def synthesize(request: Request):
    """
    语音合成API接口
    """
    data = await request.json()
    tts_text = data.get('text')
    spk_id = data.get('spk_id')
    speed = data.get('speed', 1.0)

    if not tts_text or not spk_id:
        raise HTTPException(status_code=400, detail="Missing required parameters: text or spk_id")

    try:
        # 使用异步执行语音合成，并设置超时时间为10分钟
        audio_data = await asyncio.wait_for(
            asyncio.to_thread(cosy_voice.inference_sft, tts_text, spk_id, speed=speed),
            timeout=600
        )

        # 拼接音频数据
        audio_data = torch.concat([torch.tensor(chunk['tts_speech'].numpy().ravel()) for chunk in audio_data], dim=0)

        # 保存音频文件
        output_path = "output_audio.wav"
        torchaudio.save(output_path, audio_data, cosy_voice.sample_rate)

        return {"message": "Synthesis completed", "output_path": output_path}

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Synthesis process timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
