import torch
from typing import Any, Dict, List, Union, Tuple
import copy
import random
from collections import defaultdict
from .base_algo import BaseRLAlgo
from utils.logger import logger

class GRPO(BaseRLAlgo):
    """
    工业级 GRPO (Group Relative Policy Optimization) 算法落地实现。
    严格对齐 RAGEN 论文基线和 DeepSeekMath 思想:
    1. 丢弃 Critic 价值网络，极大降低显存开销。
    2. 基于同一 Prompt 多次生成的轨迹(Group)进行相对回报归一化计算 Advantage。
    3. 完整的 KL 散度约束防止偏离 Reference Model。
    """
    
    def __init__(self, config: Any, agent: Any):
        super().__init__(config, agent)
        self.lr = getattr(config, 'learning_rate', 1e-5)
        self.clip_ratio = getattr(config, 'clip_ratio', 0.2)
        self.grpo_epochs = getattr(config, 'ppo_epochs', 1) 
        self.mini_batch_size = getattr(config, 'mini_batch_size', 2)
        self.ent_coef = getattr(config, 'ent_coef', 0.01)
        self.kl_coef = getattr(config, 'kl_coef', 0.05)
        
        self.device = agent.device
        self.tokenizer = agent.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        if hasattr(self.agent, 'model'):
            self.actor = self.agent.model
            
            logger.info("GRPO: Creating Reference Model (Frozen) for KL penalty...")
            self.ref_model = copy.deepcopy(self.actor)
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)
            
            self.optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.lr)
            logger.info(f"Initialized GRPO algorithm with LR={self.lr}, Epochs={self.grpo_epochs}, MiniBatch={self.mini_batch_size}")
        else:
            logger.warning("Agent does not have a local 'model' attribute. Cannot train locally.")

    def get_action(self, state: Any, evaluate: bool = False) -> Any:
        return self.agent.chat_request(state)

    def _prepare_data(self, batch_data: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """将环境交互轨迹拉平并转为 Tokenizer 编码的张量"""
        data = []
        for traj in batch_data:
            for step in traj:
                prompt_text = self.tokenizer.apply_chat_template(step["messages"], tokenize=False, add_generation_prompt=True)
                response_text = step["response"]
                reward = step["reward"]
                
                prompt_ids = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
                response_ids = self.tokenizer(response_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
                
                input_ids = torch.cat([prompt_ids, response_ids], dim=0)
                response_mask = torch.cat([torch.zeros_like(prompt_ids), torch.ones_like(response_ids)], dim=0)
                
                data.append({
                    "prompt_text": prompt_text,
                    "input_ids": input_ids,
                    "response_mask": response_mask,
                    "reward": float(reward)
                })
        return data

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = [s['input_ids'] for s in batch]
        response_mask = [s['response_mask'] for s in batch]
        
        pad_id = self.tokenizer.pad_token_id
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        padded_response_mask = torch.nn.utils.rnn.pad_sequence(response_mask, batch_first=True, padding_value=0)
        attention_mask = (padded_input_ids != pad_id).long()
        
        return padded_input_ids, attention_mask, padded_response_mask

    def _get_log_probs(self, model, input_ids, attention_mask, response_mask):
        """核心前向计算: 获取每个token的对数概率"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits 
        
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        labels = input_ids[:, 1:]
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
        shifted_response_mask = response_mask[:, 1:]
        return token_log_probs, shifted_response_mask

    def train_step(self, batch_data: List[List[Dict[str, Any]]]) -> Dict[str, Union[float, str]]:
        if not hasattr(self.agent, 'model'):
            return {"error": "Cannot train without a local model."}
            
        data = self._prepare_data(batch_data)
        if len(data) == 0:
            return {}
            
        logger.info(f"Starting GRPO Phase: Calculating Group Advantages for {len(data)} items...")
        
        # 1. 组相对归一化 (Group Relative Advantage)
        # 精确对齐 GRPO 原理：按照同一个 Prompt 分组进行 Reward 归一化
        grouped_data = defaultdict(list)
        for i, d in enumerate(data):
            grouped_data[d["prompt_text"]].append(i)
            
        for prompt, indices in grouped_data.items():
            rewards = torch.tensor([data[i]["reward"] for i in indices], dtype=torch.float32)
            if len(rewards) > 1:
                adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                adv = torch.zeros_like(rewards)
            for idx, a in zip(indices, adv):
                data[idx]["advantage"] = a.item()
                
        # 2. 预计算阶段 (Precomputation)
        self.actor.eval()
        self.ref_model.eval()
        for i in range(0, len(data), self.mini_batch_size):
            batch = data[i:i+self.mini_batch_size]
            input_ids, attention_mask, response_mask = self._collate_fn(batch)
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            response_mask = response_mask.to(self.device)
            
            with torch.no_grad():
                old_log_probs, shifted_response_mask = self._get_log_probs(self.actor, input_ids, attention_mask, response_mask)
                ref_log_probs, _ = self._get_log_probs(self.ref_model, input_ids, attention_mask, response_mask)
            
            for j, item in enumerate(batch):
                seq_len = item["input_ids"].size(0) - 1
                item["old_log_probs"] = old_log_probs[j, :seq_len].cpu()
                item["ref_log_probs"] = ref_log_probs[j, :seq_len].cpu()

        # 3. 训练阶段 (Training)
        self.actor.train()
        total_actor_loss, total_entropy, total_kl = 0.0, 0.0, 0.0
        update_steps = 0
        
        logger.info(f"Starting GRPO Phase: Optimizing Actor for {self.grpo_epochs} epochs...")
        for epoch in range(self.grpo_epochs):
            random.shuffle(data)
            for i in range(0, len(data), self.mini_batch_size):
                batch = data[i:i+self.mini_batch_size]
                input_ids, attention_mask, response_mask = self._collate_fn(batch)
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                response_mask = response_mask.to(self.device)
                
                old_log_probs_batch = torch.nn.utils.rnn.pad_sequence([b["old_log_probs"] for b in batch], batch_first=True, padding_value=0.0).to(self.device)
                ref_log_probs_batch = torch.nn.utils.rnn.pad_sequence([b["ref_log_probs"] for b in batch], batch_first=True, padding_value=0.0).to(self.device)
                
                # GRPO 优势直接由 Turn-level 映射到每个有效的 Token
                adv_batch = torch.tensor([b["advantage"] for b in batch], dtype=torch.float32, device=self.device)
                
                new_log_probs, shifted_response_mask = self._get_log_probs(self.actor, input_ids, attention_mask, response_mask)
                loss_mask = shifted_response_mask.bool()
                
                # KL & Entropy
                kl = torch.exp(ref_log_probs_batch - new_log_probs) - (ref_log_probs_batch - new_log_probs) - 1.0
                entropy = -(torch.exp(new_log_probs) * new_log_probs)
                
                # PPO-Clip
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                adv_seq = adv_batch.unsqueeze(1).expand_as(ratio) # (B, L)
                
                surr1 = ratio * adv_seq
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_seq
                actor_loss = -torch.min(surr1, surr2)
                
                # Masked Mean
                actor_loss_mean = (actor_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1e-8)
                kl_loss_mean = (kl * loss_mask).sum() / loss_mask.sum().clamp(min=1e-8)
                entropy_loss_mean = (entropy * loss_mask).sum() / loss_mask.sum().clamp(min=1e-8)
                
                # Total Loss (No critic loss in GRPO)
                loss = actor_loss_mean + self.kl_coef * kl_loss_mean - self.ent_coef * entropy_loss_mean
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.optimizer.step()
                
                total_actor_loss += actor_loss_mean.item()
                total_kl += kl_loss_mean.item()
                total_entropy += entropy_loss_mean.item()
                update_steps += 1
                
        return {
            "actor_loss": total_actor_loss / max(1, update_steps),
            "kl_penalty": total_kl / max(1, update_steps),
            "entropy": total_entropy / max(1, update_steps)
        }

    def save(self, path: str) -> None:
        if hasattr(self.agent, 'model') and hasattr(self.agent, 'tokenizer'):
            import os
            os.makedirs(path, exist_ok=True)
            self.agent.model.save_pretrained(path)
            self.agent.tokenizer.save_pretrained(path)
            logger.info(f"GRPO actor and tokenizer saved to {path}")

    def load(self, path: str) -> None:
        logger.info(f"GRPO model load logic is handled by HFAgent directly from {path}")
