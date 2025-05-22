def preprocess_function(examples,tokenizer,max_seq_length,sentence1_key,sentence2_key=None):
        # Helper to trim sentences to fit max token length
        def trim_to_max_length(s1, s2=None):
            # Start with original sentences
            s1_words = s1.split()
            s2_words = s2.split() if s2 is not None else None
            while True:
                
                # Remove last word from the longer sentence
                if s2_words is not None and len(s2_words) > 0 and (len(s2_words) >= len(s1_words)):
                    s2_words = s2_words[:-1]
                elif len(s1_words) > 0:
                    s1_words = s1_words[:-1]
                
                max_length = len(s1_words) + len(s2_words) if s2 is not None else len(s1_words)
                if max_length <= max_seq_length-3:
                    break

            if s2_words is not None:
                return ' '.join(s1_words), ' '.join(s2_words)
            else:
                return ' '.join(s1_words), None

        if sentence2_key is not None:
            s1_list = examples[sentence1_key]
            s2_list = examples[sentence2_key]
            trimmed_s1 = []
            trimmed_s2 = []
            for s1, s2 in zip(s1_list, s2_list):
                t1, t2 = trim_to_max_length(s1, s2)
                trimmed_s1.append(t1)
                trimmed_s2.append(t2)
            return tokenizer(
                trimmed_s1,
                trimmed_s2,
                truncation=True,
                max_length=max_seq_length
            )
        else:
            s1_list = examples[sentence1_key]
            trimmed_s1 = []
            for s1 in s1_list:
                t1, _ = trim_to_max_length(s1)
                trimmed_s1.append(t1)
            return tokenizer(
                trimmed_s1,
                truncation=True,
                max_length=max_seq_length
            )


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, List, Optional, Tuple, Union
import logging
from transformers import TrainingArguments

logger = logging.getLogger(__name__)

class TrainingArguments_Distill(TrainingArguments):
    def __init__(self, steps_per_layer=100, learning_rate_final=0.00001, **kwargs):
        super().__init__(**kwargs)
        self.steps_per_layer = steps_per_layer
        self.learning_rate_final = learning_rate_final

class Trainer_Distill(Trainer):
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        *args,
        **kwargs
    ):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()  # Set teacher model to eval mode
        
        # Move models to the same device
        device = next(student_model.parameters()).device
        self.teacher_model = self.teacher_model.to(device)
        
        # Freeze teacher model parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Initialize storage for teacher and student outputs
        self.teacher_outputs = {}
        self.student_outputs = {}
        
        # Register hooks for teacher model
        self._register_teacher_hooks()

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # Call the original log method
        super().log(logs)
        
        # Log custom metrics
        if "loss" in logs:
            logger.info(f"Step {self.state.global_step}: Loss = {logs['loss']}")
        if "learning_rate" in logs:
            logger.info(f"Learning Rate = {logs['learning_rate']}")
        if "eval_loss" in logs:
            logger.info(f"Evaluation Loss = {logs['eval_loss']}")
        if "eval_accuracy" in logs:
            logger.info(f"Evaluation Accuracy = {logs['eval_accuracy']}")

    def _register_teacher_hooks(self):
        """Register hooks to capture teacher model outputs"""
        def get_activation(name, storage):
            def hook(model, input, output):
                storage[name] = output
            return hook
        
        # Register hook for embeddings
        self.teacher_model.bert.embeddings.register_forward_hook(
            get_activation('embeddings', self.teacher_outputs)
        )
        
        # Register hooks for each encoder layer
        for i, layer in enumerate(self.teacher_model.bert.encoder.layer):
            # Register hook for layer output
            layer.register_forward_hook(
                get_activation(f'layer_{i}', self.teacher_outputs)
            )
            
    def _register_student_hooks(self):
        """Register hooks to capture student model outputs"""
        def get_activation(name, storage):
            def hook(model, input, output):
                storage[name] = output
            return hook
        
        # Clear previous student outputs
        self.student_outputs.clear()
        
        # Register hook for embeddings
        self.model.bert.embeddings.register_forward_hook(
            get_activation('embeddings', self.student_outputs)
        )
        
        # Register hooks for each encoder layer
        for i, layer in enumerate(self.model.bert.encoder.layer):
            # Register hook for layer output
            layer.register_forward_hook(
                get_activation(f'layer_{i}', self.student_outputs)
            )
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the distillation loss between teacher and student models
        Includes layer-wise distillation (MSE + Cosine), attention distillation (KL),
        soft label matching, and embedding matching
        """
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Clear previous outputs
        self.teacher_outputs.clear()
        self.student_outputs.clear()
        
        # Register hooks for student model
        if model.training==False:
            student_outputs = model(**inputs)
            task_loss = student_outputs.loss

            return (task_loss, student_outputs) if return_outputs else task_loss

        elif model.training==True:
            self._register_student_hooks()
            
            # Get teacher outputs with attention weights
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs, output_attentions=True)
                
            model.train()
            for param in model.parameters():
                param.requires_grad = True

            # Get student outputs with attention weights
            student_outputs = model(**inputs, output_attentions=True)
            
            # Task loss (e.g. cross entropy for classification)
            task_loss = student_outputs.loss
            
            # Embedding layer matching loss (MSE + Cosine)
            teacher_embeddings = self.teacher_outputs['embeddings']
            student_embeddings = self.student_outputs['embeddings']
            embedding_mse_loss = F.mse_loss(student_embeddings, teacher_embeddings)
            
            # Compute cosine similarity loss for embeddings
            t_emb_flat = teacher_embeddings.view(-1,teacher_embeddings.size(-1))
            s_emb_flat = student_embeddings.view(-1,student_embeddings.size(-1))
            embedding_cos_loss = 1 - F.cosine_similarity(t_emb_flat, s_emb_flat).mean()
            
            # Combine embedding losses
            embedding_loss = embedding_mse_loss + embedding_cos_loss
            
            # Layer-wise distillation loss (MSE + Cosine) and attention distillation
            distill_loss = 0.0
            attention_loss = 0.0
            num_layers = len(self.teacher_model.bert.encoder.layer)

            current_step = self.state.global_step
            steps_per_layer = self.args.steps_per_layer

            layer_loss = 0

            layer_use = min(1+current_step//steps_per_layer,num_layers+1)
            
            
            for i in range(min(layer_use,num_layers)):
                # Get hidden states from teacher and student
                t_hidden = self.teacher_outputs[f'layer_{i}']
                s_hidden = self.student_outputs[f'layer_{i}']
                
                # Handle tuple outputs (BERT returns tuple with hidden states as first element)
                if isinstance(t_hidden, tuple):
                    t_hidden = t_hidden[0]  # Get the hidden states
                if isinstance(s_hidden, tuple):
                    s_hidden = s_hidden[0]  # Get the hidden states
                
                # Compute MSE loss between hidden states
                layer_mse_loss = F.mse_loss(s_hidden, t_hidden)
                
                # Compute cosine similarity loss
                t_hidden_flat = t_hidden.view(-1,t_hidden.size(-1))
                s_hidden_flat = s_hidden.view(-1,s_hidden.size(-1))
                layer_cos_loss = 1 - F.cosine_similarity(t_hidden_flat, s_hidden_flat, dim=-1).mean()
                
                # Get attention weights directly from model outputs
                t_attention = teacher_outputs.attentions[i]  # Shape: (batch_size, num_heads, seq_len, seq_len)
                s_attention = student_outputs.attentions[i]
                
                # Compute KL divergence loss for attention weights
                # Reshape attention weights to 2D for KL divergence
                t_att_flat = t_attention.view(-1,t_attention.size(-1))
                s_att_flat = s_attention.view(-1,s_attention.size(-1))
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-8
                t_att_flat = t_att_flat + epsilon
                s_att_flat = s_att_flat + epsilon
                
                # Compute KL divergence loss
                layer_att_loss = F.kl_div(
                    s_att_flat.log(),
                    t_att_flat
                )
                
                # Combine layer losses
                layer_loss = layer_mse_loss + layer_cos_loss
                distill_loss += layer_loss
                attention_loss += layer_att_loss
            
            # layer_loss = layer_loss/max(layer_use,1)
            attention_loss = attention_loss/max(layer_use,1)
            distill_loss = distill_loss/max(layer_use,1)
            # Soft label distillation loss using KL divergence
            temperature = 1.0
            teacher_logits = teacher_outputs.logits / temperature
            student_logits = student_outputs.logits / temperature

            
            soft_label_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
            ) 

            if layer_use != num_layers+1:
                soft_label_loss = 0
                task_loss = 0
            else:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.args.learning_rate_final
            
            # Combine all losses with appropriate weights
        
            # total_loss = task_loss + soft_label_loss
            p = 1.0
            total_loss = (task_loss + 
                        p * distill_loss + 
                        p * soft_label_loss + 
                        p * embedding_loss +
                        p * attention_loss)
            # return total_loss
            # Log custom metrics
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    "total_loss": total_loss.item(),
                    "current_step": self.state.global_step,
                    "layer_use": layer_use,  
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                })
            return (total_loss, student_outputs) if return_outputs else total_loss
    
    # def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
    #     print("Input batch size:", next(iter(inputs.values())).shape[0])
    #     output = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    #     if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
    #         logits = output[1][0]
    #         print("Logits shape:", logits.shape)
    #     return output
    