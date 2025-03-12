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

#
#
# 总结
# 初始化 (__init__)：初始化类的属性，包括加载模型和设置设备。
# 文本token提取 (_extract_text_token)：将文本转换为token，并返回token及其长度。
# 文本token生成器 (_extract_text_token_generator)：处理文本生成器并逐个返回token。
# 语音token提取 (_extract_speech_token)：将语音转换为token，并返回token及其长度。
# 扬声器嵌入提取 (_extract_spk_embedding)：从语音中提取扬声器嵌入。
# 语音特征提取 (_extract_speech_feat)：从语音中提取特征，并返回特征及其长度。
# 文本归一化 (text_normalize)：对文本进行归一化处理，包括分词、清理等操作。
# SFT前端处理 (frontend_sft)：准备SFT模式下的模型输入。
# Zero-shot前端处理 (frontend_zero_shot)：准备Zero-shot模式下的模型输入。
# 跨语言前端处理 (frontend_cross_lingual)：准备跨语言模式下的模型输入。
# 指令模式前端处理 (frontend_instruct 和 frontend_instruct2)：准备指令模式下的模型输入。
# 语音转换前端处理 (frontend_vc)：准备语音转换模式下的模型输入。

from functools import partial
from typing import Generator
import json
import onnxruntime
import torch
import numpy as np
import whisper
from typing import Callable
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import os
import re
import inflect

try:
    import ttsfrd

    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer

    use_ttsfrd = False
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, \
    spell_out_number, split_paragraph, is_only_punctuation


class CosyVoiceFrontEnd:
    """
    CosyVoice前端处理类，用于文本和语音的预处理。
    """

    def __init__(self,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 campplus_model: str,
                 speech_tokenizer_model: str,
                 spk2info: str = '',
                 allowed_special: str = 'all'):
        """
        初始化CosyVoice前端处理类。

        :param get_tokenizer: 获取tokenizer的函数。
        :param feat_extractor: 提取语音特征的函数。
        :param campplus_model: Campplus模型路径。
        :param speech_tokenizer_model: 语音tokenizer模型路径。
        :param spk2info: 扬声器信息路径，默认为空字符串。
        :param allowed_special: 允许的特殊字符，默认为'all'。
        """
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option,
                                                             providers=["CPUExecutionProvider"])
        self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option,
                                                                     providers=[
                                                                         "CUDAExecutionProvider" if torch.cuda.is_available() else
                                                                         "CPUExecutionProvider"])
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)
        else:
            self.spk2info = {}
        self.allowed_special = allowed_special
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize('{}/../../pretrained_models/CosyVoice-ttsfrd/resource'.format(ROOT_DIR)) is True, \
                'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyinvg')
        else:
            self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False, overwrite_cache=True)
            self.en_tn_model = EnNormalizer()
            self.inflect_parser = inflect.engine()

    def _extract_text_token(self, text):
        """
        提取文本token。

        :param text: 输入文本或生成器。
        :return: 文本token及其长度。
        """
        if isinstance(text, Generator):
            logging.info('get tts_text generator, will return _extract_text_token_generator!')
            # 添加一个虚拟的text_token_len以保持兼容性
            return self._extract_text_token_generator(text), torch.tensor([0], dtype=torch.int32).to(self.device)
        else:
            text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
            text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
            text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
            return text_token, text_token_len

    def _extract_text_token_generator(self, text_generator):
        """
        提取文本token生成器。

        :param text_generator: 输入文本生成器。
        :return: 文本token生成器。
        """
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i: i + 1]

    def _extract_speech_token(self, speech):
        """
        提取语音token。

        :param speech: 输入语音。
        :return: 语音token及其长度。
        """
        assert speech.shape[1] / 16000 <= 30, '不支持提取超过30秒的音频token'
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None,
                                                         {self.speech_tokenizer_session.get_inputs()[0].name:
                                                              feat.detach().cpu().numpy(),
                                                          self.speech_tokenizer_session.get_inputs()[1].name:
                                                              np.array([feat.shape[2]], dtype=np.int32)})[
            0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        """
        提取扬声器嵌入。

        :param speech: 输入语音。
        :return: 扬声器嵌入。
        """
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None,
                                              {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(
                                                  dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        """
        提取语音特征。

        :param speech: 输入语音。
        :return: 语音特征及其长度。
        """
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def dealing_with_polyphonic_words(self, text):
        """
        处理多音字
        """
        try:
            with open("多音字.txt", 'r', encoding='utf-8') as file:
                t_lines = file.readlines()
            for x in t_lines:
                cleaned_line = x.replace("\n", "")

                cl = cleaned_line.split(' ')

                text = re.sub(cl[0], cl[1], text)
        except Exception as e:
            print(e)
            pass

        return text.strip()

    def text_normalize(self, text, split=True, text_frontend=True):
        """
        对文本进行归一化处理。

        :param text: 输入文本或生成器。
        :param split: 是否分割文本，默认为True。
        :param text_frontend: 是否使用文本前端处理，默认为True。
        :return: 归一化后的文本。
        """
        # 处理多音字
        text = self.dealing_with_polyphonic_words(text)
        if isinstance(text, Generator):
            logging.info('get tts_text generator, will skip text_normalize!')
            return [text]
        if not text_frontend:
            return [text] if split else text
        text = text.strip()
        if self.use_ttsfrd:
            texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
            text = ''.join(texts)
        else:
            if contains_chinese(text):
                text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r'[，,、]+$', '。', text)
                texts = list(
                    split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh",
                                    token_max_n=80,
                                    token_min_n=60, merge_len=20, comma_split=False))
            else:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(
                    split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en",
                                    token_max_n=80,
                                    token_min_n=60, merge_len=20, comma_split=False))
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split else text

    def frontend_sft(self, tts_text, spk_id):
        """
        SFT前端处理。

        :param tts_text: 输入文本。
        :param spk_id: 扬声器ID。
        :return: 模型输入字典。
        """
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]['embedding']
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len, 'llm_embedding': embedding,
                       'flow_embedding': embedding}
        return model_input

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, resample_rate):
        """
        Zero-shot前端处理。

        :param tts_text: 输入文本。
        :param prompt_text: 提示文本。
        :param prompt_speech_16k: 提示语音（16kHz）。
        :param resample_rate: 重采样率。
        :return: 模型输入字典。
        """
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(
            prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        if resample_rate == 24000:
            # 强制speech_feat % speech_token = 2
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {
            'text': tts_text_token,
            'text_len': tts_text_token_len,
            'prompt_text': prompt_text_token,
            'prompt_text_len': prompt_text_token_len,
            'llm_prompt_speech_token': speech_token,
            'llm_prompt_speech_token_len': speech_token_len,
            'flow_prompt_speech_token': speech_token,
            'flow_prompt_speech_token_len': speech_token_len,
            'prompt_speech_feat': speech_feat,
            'prompt_speech_feat_len': speech_feat_len,
            'llm_embedding': embedding,
            'flow_embedding': embedding
        }
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_speech_16k, resample_rate):
        """
        跨语言前端处理。

        :param tts_text: 输入文本。
        :param prompt_speech_16k: 提示语音（16kHz）。
        :param resample_rate: 重采样率。
        :return: 模型输入字典。
        """
        model_input = self.frontend_zero_shot(tts_text, '', prompt_speech_16k, resample_rate)
        # 在跨语言模式下，移除LLM中的提示
        del model_input['prompt_text']
        del model_input['prompt_text_len']
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        """
        指令模式前端处理。

        :param tts_text: 输入文本。
        :param spk_id: 扬声器ID。
        :param instruct_text: 指令文本。
        :return: 模型输入字典。
        """
        model_input = self.frontend_sft(tts_text, spk_id)
        # 在指令模式下，移除LLM中的扬声器嵌入以防止信息泄露
        del model_input['llm_embedding']
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text + '<endofprompt>')
        model_input['prompt_text'] = instruct_text_token
        model_input['prompt_text_len'] = instruct_text_token_len
        return model_input

    def frontend_instruct2(self, tts_text, instruct_text, prompt_speech_16k, resample_rate):
        """
        指令模式2前端处理。

        :param tts_text: 输入文本。
        :param instruct_text: 指令文本。
        :param prompt_speech_16k: 提示语音（16kHz）。
        :param resample_rate: 重采样率。
        :return: 模型输入字典。
        """
        model_input = self.frontend_zero_shot(tts_text, instruct_text + '<|endofprompt|>', prompt_speech_16k,
                                              resample_rate)
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_vc(self, source_speech_16k, prompt_speech_16k, resample_rate):
        """
        语音转换前端处理。

        :param source_speech_16k: 源语音（16kHz）。
        :param prompt_speech_16k: 提示语音（16kHz）。
        :param resample_rate: 重采样率。
        :return: 模型输入字典。
        """
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(prompt_speech_16k)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(
            prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        source_speech_token, source_speech_token_len = self._extract_speech_token(source_speech_16k)
        model_input = {
            'source_speech_token': source_speech_token,
            'source_speech_token_len': source_speech_token_len,
            'flow_prompt_speech_token': prompt_speech_token,
            'flow_prompt_speech_token_len': prompt_speech_token_len,
            'prompt_speech_feat': prompt_speech_feat,
            'prompt_speech_feat_len': prompt_speech_feat_len,
            'flow_embedding': embedding
        }
        return model_input
