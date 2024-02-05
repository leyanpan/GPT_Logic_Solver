import os
from transformers import (Trainer, 
                          get_linear_schedule_with_warmup, 
                          get_cosine_schedule_with_warmup)
from torch.optim import AdamW
import torch

class SATHFTrainer(Trainer):
    
    def create_optimizer(self):
        '''
        TODO
        '''
        self.optimizer = AdamW(self.model.parameters(), 
                               lr=6e-4, 
                               betas=(0.9, 0.95), 
                               weight_decay=0.1)
        return self.optimizer

    
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        '''
        TODO
        '''
        
        if optimizer is None:
            optimizer = self.optimizer
            
        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                            num_warmup_steps=2000, 
                                                            num_training_steps=num_training_steps)
        return self.lr_scheduler

    def training_step(self, 
                      model, 
                      inputs):
        '''
        TODO
        '''
        outputs = super().training_step(model, inputs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        return outputs
    
    def save_vocabulary(self, 
                        save_directory, 
                        filename_prefix=None):
        '''
        TODO
        '''
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token in self.vocab.keys():
                writer.write(token + "\n")

        return (vocab_file,)