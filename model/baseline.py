import math
import torch
import torch.nn as nn
from transformers import BertModel,AutoTokenizer
import torch.nn.functional as F
import copy
import transformers
import numpy as np
class Baseline_Extractor(nn.Module):
    
    def __init__(self, config, load_path=''):
        """
        Initialize networks
        """
        super(Baseline_Extractor, self).__init__()
        # config
        self.config=config
        
        # random seed
        seed = self.config.seed
        torch.manual_seed(seed)           
        torch.cuda.manual_seed(seed)       
        torch.cuda.manual_seed_all(seed)         
        
        #encoder
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_tokenizer)
        self.extractor = BertModel.from_pretrained(self.config.pretrained_model)

        #=============== Decoder =================#
        # Note: input embeddings are concatenated with the sampled latent vector at every step
        self.extract_classifier = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                                nn.Linear(self.config.hidden_dim, 1))

        self.dropout=nn.Dropout(self.config.dropout)

    def forward(self, batch_source_id,
                      batch_source_id_mask,
                      batch_label,
                      batch_label_mask,
                      batch_clss,
                      batch_clss_mask,
                      return_sent_emb=0):
        
        self.batch_clss_mask=batch_clss_mask.float()

        outputs = self.extractor(input_ids=batch_source_id,attention_mask=batch_source_id_mask)
        top_vec = outputs.last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), batch_clss] #(b,s,dim)
        
        sent_len=batch_clss.size()[1]
        p_weight=1
        lossFunc = nn.BCEWithLogitsLoss(weight=batch_clss_mask.float(),pos_weight=(torch.tensor([p_weight]*sent_len)).cuda())

        extractor_output=self.extract_classifier(sents_vec).squeeze(2) 
        sent_len=batch_clss.size()[1]
        extract_loss= lossFunc(extractor_output,batch_label)        
        sigmoid = nn.Sigmoid()
        predicts_score=sigmoid(extractor_output)*batch_clss_mask     
    
        #================ total weighted loss ==========#

        total_loss = extract_loss
        if return_sent_emb ==0:
            return total_loss,predicts_score
        else:
            return total_loss,predicts_score,sents_vec

    def inference(self, batch_source_id,
                      batch_source_id_mask,
                      batch_label,
                      batch_label_mask,
                      batch_clss,
                      batch_clss_mask,
                      return_focus=0):
        
        

        outputs = self.extractor(input_ids=batch_source_id,attention_mask=batch_source_id_mask)
        top_vec = outputs.last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), batch_clss] #(b,s,dim)
        
        sigmoid = nn.Sigmoid()
        
        extractor_output=self.extract_classifier(sents_vec).squeeze(2)    
        predicts_score=sigmoid(extractor_output)*batch_clss_mask
        
        return predicts_score
          
    
    
    
    
    
    def get_params(self):

        other_params = \
            list(self.extractor.parameters()) + \
            list(self.extract_classifier.parameters())
            
        return other_params
    
    def get_params_fewshot(self):
        other_params = \
            list(self.extractor.parameters()) + \
            list(self.extract_classifier.parameters())
            
        return other_params