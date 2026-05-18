import os
import torch
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_agent import BaseAgent
from utils.logger import logger
from configs.constants import MODELS_DIR

class HFAgent(BaseAgent):
    """
    基于 HuggingFace Transformers 的本地大语言模型代理。
    用于强化学习训练。
    """
    def __init__(self, config: Any):
        super().__init__(config)
        # AgentConfig 必填字段；缺失直接 AttributeError，不再静默兜底。
        self.model_name_or_path = config.model_name_or_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer, self.model = self._load_model_and_tokenizer()
        self.model.eval()  # 推理模式，训练时在RL算法模块会切换为train()
        
    def _load_model_and_tokenizer(self):
        """
        读取模型时优先在本地文件夹里读取（路径为ragen_ward/models），
        如果有就直接本地读，如果没有就从HF云端下载到本地models文件夹里，然后再从本地读
        """
        # 将HF模型名中的斜杠替换为下划线以作为合法文件夹名
        safe_model_name = self.model_name_or_path.replace("/", "_")
        local_model_dir = os.path.join(MODELS_DIR, safe_model_name)
        
        if os.path.exists(local_model_dir):
            logger.info(f"Loading model and tokenizer from local directory: {local_model_dir} on {self.device}")
            tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                local_model_dir,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            ).to(self.device)
        else:
            logger.info(f"Local model not found. Downloading {self.model_name_or_path} from HuggingFace Hub to {local_model_dir}...")
            # 1. 尝试使用 huggingface_hub 的 snapshot_download 带有进度条的下载
            try:
                from huggingface_hub import snapshot_download
                logger.info("Downloading model files with progress bar...")
                # 这会下载整个仓库并显示进度条
                downloaded_path = snapshot_download(
                    repo_id=self.model_name_or_path,
                    local_dir=local_model_dir,
                    local_dir_use_symlinks=False # 直接复制文件而不是软链接
                )
                logger.info(f"Download complete. Loading from {downloaded_path}...")
                
                # 然后从刚才下载的本地路径加载
                tokenizer = AutoTokenizer.from_pretrained(downloaded_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    downloaded_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
            except ImportError:
                logger.warning("huggingface_hub not installed. Falling back to default download (might not show progress bar).")
                logger.info("You can install it via: pip install huggingface_hub")
                # 退回到默认的 from_pretrained 下载方式
                tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
                # 2. 保存到本地models文件夹
                os.makedirs(local_model_dir, exist_ok=True)
                logger.info(f"Saving downloaded model and tokenizer to {local_model_dir}")
                tokenizer.save_pretrained(local_model_dir)
                model.save_pretrained(local_model_dir)
            
            # 3. 将模型加载到指定计算设备
            model = model.to(self.device)
            
        return tokenizer, model
        
    def chat_request(self, messages: List[Dict[str, str]]) -> str:
        """
        生成文本回复
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def batched_chat_request(self, messages_list: List[List[Dict[str, str]]]) -> List[str]:
        """
        Batched 版本的 ``chat_request``：N 条独立 messages 同时调用一次 ``model.generate``，
        相比 N 次单独调用可获得 3~10x throughput（小模型 + 单卡场景下尤其明显，因为
        batch=1 generate 是 memory-bandwidth bound——增大 batch 几乎"免费"地复用 weights fetch）。

        与"N 次串行 ``chat_request``"的数学等价性（重要）：
            - 注意力对 padding 完全屏蔽（``attention_mask`` 正确传入），所以每条 sequence
              的 logits 跟单条单独 generate 的 logits 在数值上等价（仅 BF16 reduction-order
              层面有 ~1e-4 噪声，远小于 sampling temperature 引入的随机性）。
            - 每条 sequence 在自己的 logits 分布上独立 multinomial 采样，**采样分布严格一致**；
              具体采到的 token 因 RNG 流不同而异，这与 N 次串行调用之间的"采样独立性"是同
              一性质，不会引入系统性偏差。
            - 每条 sequence 各自在自己的 EOS / max_new_tokens 截断（HF generate 内部对每条
              独立维护 finished 标志），互不影响。

        Args:
            messages_list: 长度 N 的列表，每个元素是与 ``chat_request`` 入参同结构的 messages。

        Returns:
            List[str]: 长度 N 的 response，顺序与 ``messages_list`` 一一对应。
        """
        if len(messages_list) == 0:
            return []
        if len(messages_list) == 1:
            # 单条退化为非 batched 路径，避免不必要的 padding 开销
            return [self.chat_request(messages_list[0])]

        texts = [
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msgs in messages_list
        ]

        # generate 必须用 left padding（因为新 token 从最右端续写），
        # 临时切换 padding_side 并在结束后恢复，避免对外部状态产生副作用。
        original_padding_side = self.tokenizer.padding_side
        original_pad_token_id = self.tokenizer.pad_token_id
        try:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            model_inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        finally:
            self.tokenizer.padding_side = original_padding_side
            self.tokenizer.pad_token_id = original_pad_token_id

        prompt_len = model_inputs.input_ids.shape[1]

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 因为是 left-padding，所有 prompt 占据 [0, prompt_len)，新生成部分都从 prompt_len 起。
        # skip_special_tokens=True 会自动剥除 EOS 后的 padding，所以无需手动按 EOS 截断。
        new_tokens = generated_ids[:, prompt_len:]
        responses = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return responses

    def get_log_probs(self, messages: List[Dict[str, str]], response: str) -> torch.Tensor:
        """
        获取对话历史+生成的回复对应各个token的log_probs，用于RL算法。
        这部分一般在RL_algo模块计算，这里预留接口。
        """
        pass
