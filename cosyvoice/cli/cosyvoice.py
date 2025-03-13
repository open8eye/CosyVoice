# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from typing import Generator

import torchaudio
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
grandparent_dir = os.path.dirname(parent_dir)


def ms_to_srt_time(ms):
    """将毫秒转换为SRT字幕格式的时间字符串。

    Args:
        ms (int): 输入的毫秒数。

    Returns:
        str: 格式化的SRT时间字符串（如"00:00:01,234"）。
    """
    N = int(ms)
    hours, remainder = divmod(N, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


class CosyVoice:

    def __init__(self, model_dir: str, load_jit: bool = False, load_trt: bool = False, fp16: bool = False):
        """初始化CosyVoice模型
        
        Args:
            model_dir (str): 模型目录路径，若不存在则自动下载
            load_jit (bool): 是否加载JIT优化模型
            load_trt (bool): 是否加载TensorRT优化模型
            fp16 (bool): 是否使用半精度浮点运算
        
        Notes:
            - 自动检测CUDA环境并调整优化参数
            - 初始化文本前端处理、声学模型和流式模型
        """
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)  # 下载模型文件到本地

        with open(f'{model_dir}/cosyvoice.yaml', 'r') as f:
            configs = load_hyperpyyaml(f)

        # 模型类型校验
        assert get_model_type(configs) != CosyVoice2Model, '请使用CosyVoice2类初始化该模型目录'

        # 初始化文本前端处理模块
        self.frontend = CosyVoiceFrontEnd(
            configs['get_tokenizer'],
            configs['feat_extractor'],
            f'{model_dir}/campplus.onnx',
            f'{model_dir}/speech_tokenizer_v1.onnx',
            f'{model_dir}/spk2info.pt',
            configs['allowed_special']
        )
        self.sample_rate = configs['sample_rate']

        # 根据CUDA环境调整优化参数
        if torch.cuda.is_available() is False and (load_jit or load_trt or fp16):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('未检测到CUDA设备，已关闭JIT/TRT/FP16加速')

        # 初始化声学模型和流式模型
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load(f'{model_dir}/llm.pt', f'{model_dir}/flow.pt', f'{model_dir}/hift.pt')

        # 加载JIT/TRT优化模型
        if load_jit:
            self.model.load_jit(
                f'{model_dir}/llm.text_encoder.{"fp16" if fp16 else "fp32"}.zip',
                f'{model_dir}/llm.llm.{"fp16" if fp16 else "fp32"}.zip',
                f'{model_dir}/flow.encoder.{"fp16" if fp16 else "fp32"}.zip'
            )
        if load_trt:
            self.model.load_trt(
                f'{model_dir}/flow.decoder.estimator.{"fp16" if fp16 else "fp32"}.mygpu.plan',
                f'{model_dir}/flow.decoder.estimator.fp32.onnx',
                fp16
            )
        del configs

    def list_available_spks(self) -> list:
        """获取可用说话人ID列表
        
        Returns:
            list: 说话人ID列表
        """
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text: str, spk_id: str, stream: bool = False, speed: float = 1.0,
                      text_frontend: bool = True, new_dropdown: str = "无", gender: bool = True) -> Generator:
        """SFT模式推理：基于说话人ID的文本到语音合成
        
        Args:
            tts_text (str): 输入文本
            spk_id (str): 目标说话人ID
            stream (bool): 是否启用流式输出
            speed (float): 语速调节系数（0.5-2.0）
            text_frontend (bool): 是否使用文本前端处理
            [('男', False), ('女', True)]
        
        Yields:
            dict: 包含语音数据的模型输出字典
        """
        # 默认说话人列表
        default_voices = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
        # 文本分段合成
        tts_speeches = []
        # 语音合成
        audio_opt = []
        # 语音合成时长
        audio_samples = 0
        # 文本分段合成
        srtlines = []
        # 采样频率 默认 22050
        my_sample_rate = self.sample_rate if self.sample_rate else 22050
        print('采样频率>>>', my_sample_rate)
        print('spk_id>>>', spk_id)
        for text_segment in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if new_dropdown != "无" or spk_id not in default_voices:
                # 加载自定义说话人特征
                if gender == True:
                    model_input = self.frontend.frontend_sft(text_segment, "中文女")
                else:
                    model_input = self.frontend.frontend_sft(text_segment, "中文男")
                voice_path = f'{grandparent_dir}/voices/{new_dropdown}.pt' if new_dropdown != "无" else f'{grandparent_dir}/voices/{spk_id}.pt'
                newspk = torch.load(voice_path)
                # 替换模型输入的说话人嵌入和提示信息
                model_input.update({
                    "flow_embedding": newspk["flow_embedding"],
                    "llm_embedding": newspk["llm_embedding"],
                    "llm_prompt_speech_token": newspk["llm_prompt_speech_token"],
                    "llm_prompt_speech_token_len": newspk["llm_prompt_speech_token_len"],
                    "flow_prompt_speech_token": newspk["flow_prompt_speech_token"],
                    "flow_prompt_speech_token_len": newspk["flow_prompt_speech_token_len"],
                    "prompt_speech_feat_len": newspk["prompt_speech_feat_len"],
                    "prompt_speech_feat": newspk["prompt_speech_feat"],
                    "prompt_text": newspk["prompt_text"],
                    "prompt_text_len": newspk["prompt_text_len"]
                })
            else:
                model_input = self.frontend.frontend_sft(text_segment, spk_id)
            start_time = time.time()
            logging.info(f'synthesis text {text_segment}')
            # 语音合成
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                # 模型分割长度
                speech_len = model_output['tts_speech'].shape[1] / my_sample_rate

                logging.info(f'yield 模型分割长度 {speech_len}, rtf {(time.time() - start_time) / speech_len}')

                # 额外代码  -start
                # 语音合成
                audio = model_output['tts_speech'].numpy().ravel()
                audio_opt.append(audio)
                # 生成字幕文件
                srtline_begin = ms_to_srt_time(audio_samples * 1000.0 / my_sample_rate)
                audio_samples += audio.size
                srtline_end = ms_to_srt_time(audio_samples * 1000.0 / my_sample_rate)

                srtlines.append(f"{len(audio_opt):02d}\n")
                srtlines.append(f"{srtline_begin} --> {srtline_end}\n")
                srtlines.append(f"{text_segment.replace('、。', '')}\n\n")

                tts_speeches.append(model_output['tts_speech'])
                # 额外代码  -end

                yield model_output
                start_time = time.time()
        # 额外代码  -start
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print('合成完成:', date)
        # 先屏蔽合成音频
        # 直接在页面下载文件还小些
        # 虽然小但是会出现不会返回音频的情况，所以还是做个本地音频文件保存
        audio_data = torch.concat(tts_speeches, dim=1)
        torchaudio.save(f"音频输出/output-{date}.wav", audio_data, my_sample_rate)
        with open(f'音频输出/output-{date}.srt', 'w', encoding='utf-8') as f:
            f.writelines(srtlines)
        # 额外代码  -end

    def inference_zero_shot(self, tts_text: str, prompt_text: str, prompt_speech_16k: torch.Tensor,
                            stream: bool = False, speed: float = 1.0, text_frontend: bool = True) -> Generator:
        """零样本推理：基于提示语音的文本到语音合成

        Args:
            tts_text (str): 输入文本
            prompt_text (str): 提示文本
            prompt_speech_16k (torch.Tensor): 16kHz的提示语音张量
            stream (bool): 是否启用流式输出
            speed (float): 语速调节系数（0.5-2.0）
            text_frontend (bool): 是否使用文本前端处理

        Yields:
            dict: 包含语音数据的模型输出字典
        """
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for text_segment in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_zero_shot(text_segment, prompt_text, prompt_speech_16k,
                                                           self.sample_rate)
            start_time = time.time()
            logging.info(f'synthesis text {text_segment}')
            # 保存数据
            torch.save(model_input, 'output.pt')
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info(f'yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}')
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text: str, prompt_speech_16k: torch.Tensor, stream: bool = False,
                                speed: float = 1.0, text_frontend: bool = True) -> Generator:
        """跨语言推理：基于提示语音的跨语言文本到语音合成

        Args:
            tts_text (str): 输入文本（支持多语言）
            prompt_speech_16k (torch.Tensor): 16kHz的提示语音张量
            stream (bool): 是否启用流式输出
            speed (float): 语速调节系数（0.5-2.0）
            text_frontend (bool): 是否使用文本前端处理

        Yields:
            dict: 包含语音数据的模型输出字典
        """
        for text_segment in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(text_segment, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info(f'synthesis text {text_segment}')
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info(f'yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}')
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text: str, spk_id: str, instruct_text: str, stream: bool = False,
                           speed: float = 1.0, text_frontend: bool = True) -> Generator:
        """指令驱动推理：基于指令文本和说话人ID的语音合成

        Args:
            tts_text (str): 输入文本
            spk_id (str): 目标说话人ID
            instruct_text (str): 指令文本
            stream (bool): 是否启用流式输出
            speed (float): 语速调节系数（0.5-2.0）
            text_frontend (bool): 是否使用文本前端处理

        Yields:
            dict: 包含语音数据的模型输出字典
        """
        assert isinstance(self.model, CosyVoiceModel), '仅支持CosyVoice模型'
        if not self.instruct:
            raise ValueError(f'{self.model_dir} 不支持指令推理')

        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for text_segment in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(text_segment, spk_id, instruct_text)
            start_time = time.time()
            logging.info(f'synthesis text {text_segment}')
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info(f'yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}')
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_speech_16k: torch.Tensor, prompt_speech_16k: torch.Tensor, stream: bool = False,
                     speed: float = 1.0) -> Generator:
        """语音转换：将源语音转换为目标说话人风格

        Args:
            source_speech_16k (torch.Tensor): 输入的16kHz源语音张量
            prompt_speech_16k (torch.Tensor): 目标说话人的16kHz提示语音张量
            stream (bool): 是否启用流式输出
            speed (float): 语速调节系数（0.5-2.0）

        Yields:
            dict: 包含转换后语音的模型输出字典
        """
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.vc(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info(f'yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}')
            yield model_output
            start_time = time.time()


class CosyVoice2(CosyVoice):

    def __init__(self, model_dir: str, load_jit: bool = False, load_trt: bool = False, fp16: bool = False):
        """初始化CosyVoice2模型
        
        Args:
            model_dir (str): 模型目录路径，若不存在则自动下载
            load_jit (bool): 是否加载JIT优化模型
            load_trt (bool): 是否加载TensorRT优化模型
            fp16 (bool): 是否使用半精度浮点运算
        
        Notes:
            - 自动检测CUDA环境并调整优化参数
            - 初始化文本前端处理、声学模型和流式模型
        """
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)  # 下载模型文件到本地

        with open(f'{model_dir}/cosyvoice.yaml', 'r') as f:
            configs = load_hyperpyyaml(f,
                                       overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

        # 模型类型校验
        assert get_model_type(configs) == CosyVoice2Model, '请使用CosyVoice类初始化该模型目录'

        # 初始化文本前端处理模块
        self.frontend = CosyVoiceFrontEnd(
            configs['get_tokenizer'],
            configs['feat_extractor'],
            f'{model_dir}/campplus.onnx',
            f'{model_dir}/speech_tokenizer_v2.onnx',
            f'{model_dir}/spk2info.pt',
            configs['allowed_special']
        )
        self.sample_rate = configs['sample_rate']

        # 根据CUDA环境调整优化参数
        if torch.cuda.is_available() is False and (load_jit or load_trt or fp16):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('未检测到CUDA设备，已关闭JIT/TRT/FP16加速')

        # 初始化声学模型和流式模型
        self.model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load(f'{model_dir}/llm.pt', f'{model_dir}/flow.pt', f'{model_dir}/hift.pt')

        # 加载JIT/TRT优化模型
        if load_jit:
            self.model.load_jit(f'{model_dir}/flow.encoder.{"fp16" if fp16 else "fp32"}.zip')
        if load_trt:
            self.model.load_trt(
                f'{model_dir}/flow.decoder.estimator.{"fp16" if fp16 else "fp32"}.mygpu.plan',
                f'{model_dir}/flow.decoder.estimator.fp32.onnx',
                fp16
            )
        del configs

    def inference_instruct(self, *args, **kwargs):
        """指令驱动推理（未实现）
        
        Raises:
            NotImplementedError: CosyVoice2不支持该方法
        """
        raise NotImplementedError('CosyVoice2不支持inference_instruct方法')

    def inference_instruct2(self, tts_text: str, instruct_text: str, prompt_speech_16k: torch.Tensor,
                            stream: bool = False, speed: float = 1.0, text_frontend: bool = True) -> Generator:
        """增强指令驱动推理：基于文本和语音提示的指令合成
        
        Args:
            tts_text (str): 输入文本
            instruct_text (str): 指令文本
            prompt_speech_16k (torch.Tensor): 16kHz的提示语音张量
            stream (bool): 是否启用流式输出
            speed (float): 语速调节系数（0.5-2.0）
            text_frontend (bool): 是否使用文本前端处理
        
        Yields:
            dict: 包含语音数据的模型输出字典
        """
        assert isinstance(self.model, CosyVoice2Model), '仅支持CosyVoice2模型'
        for text_segment in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(text_segment, instruct_text, prompt_speech_16k,
                                                           self.sample_rate)
            start_time = time.time()
            logging.info(f'synthesis text {text_segment}')
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info(f'yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}')
                yield model_output
                start_time = time.time()
