import torch
import torch.nn as nn
from typing import Any, Dict, List, Union, Tuple
import copy
import random
from collections import defaultdict
from .base_algo import BaseRLAlgo
from utils.logger import logger

class PPO(BaseRLAlgo):
    """
    工业级 PPO (Proximal Policy Optimization) 算法落地实现。
    严格对齐 RAGEN/veRL 的训练逻辑，包含:
    1. 真实大模型的前向 Log Probs 获取。
    2. 完整的 Actor-Critic 架构与 GAE (Generalized Advantage Estimation)。
    3. KL 散度约束 (Reference Model)。
    4. 批处理更新 (Mini-batch) 与 PPO-Clip 截断。
    """
    
    def __init__(self, config: Any, agent: Any):
        super().__init__(config, agent)
        # 从 config 提取 PPO 专属超参数
        self.lr = getattr(config, 'learning_rate', 1e-5)
        self.gamma = getattr(config, 'gamma', 0.99)
        self.lam = getattr(config, 'lam', 0.95)
        self.clip_ratio = getattr(config, 'clip_ratio', 0.2)
        self.ppo_epochs = getattr(config, 'ppo_epochs', 4)
        self.mini_batch_size = getattr(config, 'mini_batch_size', 2)
        self.vf_coef = getattr(config, 'vf_coef', 0.5)
        self.ent_coef = getattr(config, 'ent_coef', 0.01)
        self.kl_coef = getattr(config, 'kl_coef', 0.05)
        
        self.device = agent.device
        self.tokenizer = agent.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        if hasattr(self.agent, 'model'):
            self.actor = self.agent.model
            
            logger.info("PPO: Creating Reference Model (Frozen) for KL penalty...")
            self.ref_model = copy.deepcopy(self.actor)
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)
            
            logger.info("PPO: Creating Value Head (Critic)...")
            hidden_size = self.actor.config.hidden_size
            self.critic = nn.Linear(hidden_size, 1, dtype=self.actor.dtype).to(self.device)
            
            # Optimizer: Update both Actor and Critic
            self.optimizer = torch.optim.AdamW(
                list(self.actor.parameters()) + list(self.critic.parameters()), 
                lr=self.lr
            )
            logger.info(f"Initialized PPO algorithm with LR={self.lr}, Epochs={self.ppo_epochs}, MiniBatch={self.mini_batch_size}")
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

    def _get_log_probs_and_values(self, model, input_ids, attention_mask, response_mask, critic=None):
        """核心前向计算: 获取每个token的对数概率和价值预测"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=critic is not None)
        logits = outputs.logits # (B, L, V)
        
        # Shift logits and labels for next-token prediction
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        labels = input_ids[:, 1:]
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
        shifted_response_mask = response_mask[:, 1:]
        
        values = None
        if critic is not None:
            hidden_states = outputs.hidden_states[-1] # (B, L, H)
            values = self.critic(hidden_states).squeeze(-1) # (B, L)
            values = values[:, :-1]
            
        return token_log_probs, values, shifted_response_mask

    def _compute_gae(self, rewards_seq, values_seq, response_mask):
        """计算广义优势估计 (GAE)"""
        B, L = rewards_seq.shape
        advantages = torch.zeros_like(rewards_seq)
        returns = torch.zeros_like(rewards_seq)
        
        for b in range(B):
            lastgaelam = 0.0
            valid_indices = response_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                continue
            
            for t in reversed(range(len(valid_indices))):
                idx = valid_indices[t]
                next_val = values_seq[b, valid_indices[t+1]] if t + 1 < len(valid_indices) else 0.0
                delta = rewards_seq[b, idx] + self.gamma * next_val - values_seq[b, idx]
                lastgaelam = delta + self.gamma * self.lam * lastgaelam
                advantages[b, idx] = lastgaelam
                returns[b, idx] = lastgaelam + values_seq[b, idx]
                
        return advantages, returns

    def train_step(self, batch_data: List[List[Dict[str, Any]]]) -> Dict[str, Union[float, str]]:
        if not hasattr(self.agent, 'model'):
            return {"error": "Cannot train without a local model."}
            
        data = self._prepare_data(batch_data)
        if len(data) == 0:
            return {}
            
        logger.info(f"Starting PPO Phase: Precomputing LogProbs and Advantages for {len(data)} items...")
        
        self.actor.eval()
        self.ref_model.eval()
        
        # 1. 预计算阶段 (Precomputation)
        for i in range(0, len(data), self.mini_batch_size):
            batch = data[i:i+self.mini_batch_size]
            input_ids, attention_mask, response_mask = self._collate_fn(batch)
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            response_mask = response_mask.to(self.device)
            
            with torch.no_grad():
                old_log_probs, values, shifted_response_mask = self._get_log_probs_and_values(self.actor, input_ids, attention_mask, response_mask, self.critic)
                ref_log_probs, _, _ = self._get_log_probs_and_values(self.ref_model, input_ids, attention_mask, response_mask)
            
            for j, item in enumerate(batch):
                seq_len = item["input_ids"].size(0) - 1
                
                # 设置单步回合奖励(仅在回答的最后一个Token上生效)
                reward_seq = torch.zeros(seq_len)
                resp_idx = shifted_response_mask[j, :seq_len].nonzero(as_tuple=True)[0]
                if len(resp_idx) > 0:
                    reward_seq[resp_idx[-1]] = item["reward"]
                
                val_seq = values[j, :seq_len].cpu()
                adv_seq, ret_seq = self._compute_gae(reward_seq.unsqueeze(0), val_seq.unsqueeze(0), shifted_response_mask[j, :seq_len].cpu().unsqueeze(0))
                
                item["old_log_probs"] = old_log_probs[j, :seq_len].cpu()
                item["ref_log_probs"] = ref_log_probs[j, :seq_len].cpu()
                item["advantages"] = adv_seq.squeeze(0).cpu()
                item["returns"] = ret_seq.squeeze(0).cpu()

        # 2. 训练阶段 (Training)
        self.actor.train()
        total_actor_loss, total_critic_loss, total_entropy, total_kl = 0.0, 0.0, 0.0, 0.0
        update_steps = 0
        
        logger.info(f"Starting PPO Phase: Optimizing Actor & Critic for {self.ppo_epochs} epochs...")
        for epoch in range(self.ppo_epochs):
            random.shuffle(data)
            for i in range(0, len(data), self.mini_batch_size):
                batch = data[i:i+self.mini_batch_size]
                input_ids, attention_mask, response_mask = self._collate_fn(batch)
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                response_mask = response_mask.to(self.device)
                
                old_log_probs_batch = torch.nn.utils.rnn.pad_sequence([b["old_log_probs"] for b in batch], batch_first=True, padding_value=0.0).to(self.device)
                ref_log_probs_batch = torch.nn.utils.rnn.pad_sequence([b["ref_log_probs"] for b in batch], batch_first=True, padding_value=0.0).to(self.device)
                adv_batch = torch.nn.utils.rnn.pad_sequence([b["advantages"] for b in batch], batch_first=True, padding_value=0.0).to(self.device)
                ret_batch = torch.nn.utils.rnn.pad_sequence([b["returns"] for b in batch], batch_first=True, padding_value=0.0).to(self.device)
                
                new_log_probs, values, shifted_response_mask = self._get_log_probs_and_values(self.actor, input_ids, attention_mask, response_mask, self.critic)
                loss_mask = shifted_response_mask.bool()
                
                # Advantage 归一化 (Batch level)
                valid_advs = adv_batch[loss_mask]
                if valid_advs.numel() > 1:
                    adv_batch = (adv_batch - valid_advs.mean()) / (valid_advs.std() + 1e-8)
                
                # PPO-Clip
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_batch
                actor_loss = -torch.min(surr1, surr2)
                
                # Critic Loss
                critic_loss = nn.MSELoss(reduction='none')(values, ret_batch)
                
                # KL & Entropy
                kl = torch.exp(ref_log_probs_batch - new_log_probs) - (ref_log_probs_batch - new_log_probs) - 1.0
                entropy = -(torch.exp(new_log_probs) * new_log_probs)
                
                # Masked Mean
                actor_loss_mean = (actor_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1e-8)
                critic_loss_mean = (critic_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1e-8)
                kl_loss_mean = (kl * loss_mask).sum() / loss_mask.sum().clamp(min=1e-8)
                entropy_loss_mean = (entropy * loss_mask).sum() / loss_mask.sum().clamp(min=1e-8)
                
                # Total Loss
                loss = actor_loss_mean + self.vf_coef * critic_loss_mean + self.kl_coef * kl_loss_mean - self.ent_coef * entropy_loss_mean
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 1.0)
                self.optimizer.step()
                
                total_actor_loss += actor_loss_mean.item()
                total_critic_loss += critic_loss_mean.item()
                total_kl += kl_loss_mean.item()
                total_entropy += entropy_loss_mean.item()
                update_steps += 1
                
        return {
            "actor_loss": total_actor_loss / max(1, update_steps),
            "critic_loss": total_critic_loss / max(1, update_steps),
            "kl_penalty": total_kl / max(1, update_steps),
            "entropy": total_entropy / max(1, update_steps)
        }

    def save(self, path: str) -> None:
        if hasattr(self.agent, 'model') and hasattr(self.agent, 'tokenizer'):
            import os
            os.makedirs(path, exist_ok=True)
            self.agent.model.save_pretrained(path)
            self.agent.tokenizer.save_pretrained(path)
            # 保存 Critic
            torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))
            logger.info(f"PPO actor, critic, and tokenizer saved to {path}")

    def load(self, path: str) -> None:
        logger.info(f"PPO model load logic is handled by HFAgent directly from {path}")
