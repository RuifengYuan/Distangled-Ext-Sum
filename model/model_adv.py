import math
import torch
import torch.nn as nn
from transformers import BertModel,AutoTokenizer
import torch.nn.functional as F
import copy
import transformers
import numpy as np
from collections import OrderedDict
class Disetangle_Extractor_Prompt(nn.Module):
    
    def __init__(self, config, load_path=''):
        """
        Initialize networks
        """
        super(Disetangle_Extractor_Prompt, self).__init__()
        # config
        self.config=config
        
        # random seed
        seed = self.config.seed
        torch.manual_seed(seed)            
        torch.cuda.manual_seed(seed)       
        torch.cuda.manual_seed_all(seed)         
        
        #encoder
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_tokenizer)
        #self.extractor_context = BertModel.from_pretrained(self.config.pretrained_model)
        #self.extractor_pattern = BertModel.from_pretrained(self.config.pretrained_model)
        self.extractor = BertModel.from_pretrained(self.config.pretrained_model)
        self.context_map = nn.Sequential(OrderedDict([
                                          ('MLP1',nn.Linear(self.config.hidden_dim, self.config.hidden_dim)),
                                          ('RELU',nn.LeakyReLU()),
                                          ('MLP2',nn.Linear(self.config.hidden_dim, self.config.hidden_dim))
                                        ]))
        self.pattern_map = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                         nn.LeakyReLU(),
                                          nn.Linear(self.config.hidden_dim, self.config.hidden_dim))        
        #=============== Discriminator/adversary============#
        self.pattern_disc = nn.Sequential(OrderedDict([
                                          ('MLP1',nn.Linear(self.config.hidden_dim, self.config.hidden_dim)),
                                          ('RELU',nn.LeakyReLU()),
                                          ('MLP2',nn.Linear(self.config.hidden_dim, self.config.bow_dim_pattern))
                                          ]))
        
        self.context_disc = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.config.hidden_dim, self.config.bow_dim))
        
        self.position_disc = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                           nn.LeakyReLU(),
                                          nn.Linear(self.config.hidden_dim, self.config.bow_dim_pos))
        
        #=============== Classifier =============#
        self.pattern_classifier = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                                nn.LeakyReLU(),
                                                nn.Linear(self.config.hidden_dim, self.config.bow_dim_pattern))
        self.context_classifier = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                                nn.LeakyReLU(),
                                                nn.Linear(self.config.hidden_dim, self.config.bow_dim))
        self.position_classifier = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                                 nn.LeakyReLU(),
                                                 nn.Linear(self.config.hidden_dim, self.config.bow_dim_pos))
        #=============== Decoder =================#
        # Note: input embeddings are concatenated with the sampled latent vector at every step
        self.extract_classifier = nn.Sequential(nn.Linear(self.config.hidden_dim*2, self.config.hidden_dim*2),
                                                nn.LeakyReLU(),
                                                nn.Linear(self.config.hidden_dim*2, 1))
        self.extract_classifier_context = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(self.config.hidden_dim, 1))
        self.extract_classifier_pattern = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(self.config.hidden_dim, 1))
        self.dropout=nn.Dropout(self.config.dropout)

        self.context_entropy_loss_list=[]
        self.pattern_entropy_loss_list=[]
        self.position_entropy_loss_list=[]
        self.context_mul_loss_list=[]
        self.pattern_mul_loss_list=[]
        self.position_mul_loss_list=[]
        self.extract_loss_list=[]
        self.extract_loss_context_list=[]
        self.extract_loss_pattern_list=[]
        self.context_disc_loss_list=[]
        self.pattern_disc_loss_list=[]
        self.position_disc_loss_list=[]
        
        self.context_entropy_loss_avg='0'
        self.pattern_entropy_loss_avg='0'
        self.position_entropy_loss_avg='0'
        self.context_mul_loss_avg='0'
        self.pattern_mul_loss_avg='0'
        self.position_mul_loss_avg='0'
        self.extract_loss_avg='0'
        self.extract_loss_context_avg='0'
        self.extract_loss_pattern_avg='0'
        self.context_disc_loss_avg='0'
        self.pattern_disc_loss_avg='0'
        self.position_disc_loss_avg='0'
        

    def forward(self, batch_source_id_context,
                      batch_source_id_pattern,
                      batch_source_id_mask,
                      batch_source_id_mask_pattern,
                      batch_label,
                      batch_label_mask,
                      batch_clss,
                      batch_clss_mask,
                      context_bow,
                      pattern_bow,
                      position_bow):
        
        self.batch_clss_mask=batch_clss_mask.float()

        if self.config.add_prompt == 1:
            outputs_context = self.extractor(input_ids=batch_source_id_context,attention_mask=batch_source_id_mask)
            top_vec_context = outputs_context.last_hidden_state
            sents_vec_context = top_vec_context[torch.arange(top_vec_context.size(0)).unsqueeze(1), batch_clss, :] #(b,s,dim)    
            
            sents_vec_context = self.context_map(sents_vec_context)
            
            
            outputs_pattern = self.extractor(input_ids=batch_source_id_pattern,attention_mask=batch_source_id_mask_pattern)
            top_vec_pattern = outputs_pattern.last_hidden_state
            sents_vec_pattern = top_vec_pattern[torch.arange(top_vec_pattern.size(0)).unsqueeze(1), batch_clss, :] #(b,s,dim)    
            
            sents_vec_pattern = self.pattern_map(sents_vec_pattern)
        else:
            outputs = self.extractor(input_ids=batch_source_id_context,attention_mask=batch_source_id_mask)
            top_vec = outputs.last_hidden_state
            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), batch_clss, :] #(b,s,dim)    
            
            sents_vec_context = self.context_map(sents_vec)
            sents_vec_pattern = self.pattern_map(sents_vec)           
                     
        
        
        #=========== Losses on context space =============#
        # Discriminator Loss
        context_disc_preds = self.context_disc(self.dropout(sents_vec_pattern.detach()))
        context_disc_loss = self.get_context_disc_loss(
            context_disc_preds, context_bow)
        # adversarial entropy
        context_entropy_loss = self.get_context_disc_loss(self.context_disc(self.dropout(sents_vec_pattern)),context_bow)
        # Multitask loss
        context_mul_loss = self.get_context_mul_loss(
            sents_vec_context, context_bow)

        #============ Losses on pattern space ================#
        # Discriminator Loss
        pattern_disc_preds = self.pattern_disc(self.dropout(sents_vec_context.detach()))
        pattern_disc_loss = self.get_pattern_disc_loss(
            pattern_disc_preds, pattern_bow)
        # adversarial entropy
        pattern_entropy_loss = self.get_pattern_disc_loss(self.pattern_disc(self.dropout(sents_vec_context)), pattern_bow)
        # Multitask loss
        pattern_mul_loss = self.get_pattern_mul_loss(
            sents_vec_pattern, pattern_bow)
        
        #============ Losses on pattern space-position ================#
        # Discriminator Loss
        position_disc_preds = self.position_disc(self.dropout(sents_vec_context.detach()))
        position_disc_loss = self.get_position_disc_loss(
            position_disc_preds, position_bow)
        # adversarial entropy
        position_entropy_loss = self.get_position_disc_loss(self.position_disc(self.dropout(sents_vec_context)), position_bow)
        # Multitask loss
        position_mul_loss = self.get_position_mul_loss(sents_vec_pattern, position_bow)
        
        #============ Losses on extracted summarization ================#
        sent_len=batch_clss.size()[1]
        p_weight=1
        lossFunc = nn.BCEWithLogitsLoss(weight=batch_clss_mask.float(),pos_weight=(torch.tensor([p_weight]*sent_len)).cuda())
        
        #if self.config.main_detach == 0:
        extractor_output=self.extract_classifier(torch.cat([sents_vec_context,sents_vec_pattern],2)).squeeze(2) 
        #else:
            #extractor_output=self.extract_classifier(torch.cat([sents_vec_context.detach(),sents_vec_pattern.detach()],2)).squeeze(2) 
            
        sigmoid = nn.Sigmoid()
        predicts_score=sigmoid(extractor_output)*batch_clss_mask      
        extract_loss = lossFunc(extractor_output,batch_label) 
        #if self.config.aux_detach == 1:        
        extractor_output_context=self.extract_classifier_context(sents_vec_context.detach()).squeeze(2) 
        extract_loss_context = lossFunc(extractor_output_context,batch_label)         
        
        extractor_output_pattern=self.extract_classifier_pattern(sents_vec_pattern.detach()).squeeze(2) 
        extract_loss_pattern = lossFunc(extractor_output_pattern,batch_label)    
          
        #================ total weighted loss ==========#
        

        total_loss = \
            -self.config.context_adversary_loss_weight * context_entropy_loss + \
            -self.config.pattern_adversary_loss_weight * pattern_entropy_loss + \
            -self.config.position_adversary_loss_weight * position_entropy_loss + \
            self.config.context_multitask_loss_weight * context_mul_loss + \
            self.config.pattern_multitask_loss_weight * pattern_mul_loss + \
            self.config.position_multitask_loss_weight * position_mul_loss + \
            extract_loss*self.config.extract_main_weight+(extract_loss_context+extract_loss_pattern)*self.config.extract_detach_weight

        
        self.context_entropy_loss_list.append(context_entropy_loss.item())
        self.pattern_entropy_loss_list.append(pattern_entropy_loss.item())
        self.position_entropy_loss_list.append(position_entropy_loss.item())
        self.context_mul_loss_list.append(context_mul_loss.item())
        self.pattern_mul_loss_list.append(pattern_mul_loss.item())
        self.position_mul_loss_list.append(position_mul_loss.item())
        self.extract_loss_list.append(extract_loss.item())
        self.extract_loss_context_list.append(extract_loss_context.item())
        self.extract_loss_pattern_list.append(extract_loss_pattern.item())

        
        self.context_entropy_loss_list=self.context_entropy_loss_list[-500:]
        self.pattern_entropy_loss_list=self.pattern_entropy_loss_list[-500:]
        self.position_entropy_loss_list=self.position_entropy_loss_list[-500:]
        self.context_mul_loss_list=self.context_mul_loss_list[-500:]
        self.pattern_mul_loss_list=self.pattern_mul_loss_list[-500:]
        self.position_mul_loss_list=self.position_mul_loss_list[-500:]
        self.extract_loss_list=self.extract_loss_list[-500:]
        self.extract_loss_context_list=self.extract_loss_context_list[-500:]
        self.extract_loss_pattern_list=self.extract_loss_pattern_list[-500:]

        
        def avg(listx):
            s=str(np.mean(listx))
            sp=s.split('e')
            if len(sp)==1:
                return s[:6]
            else:
                return sp[0][:6]+'e'+sp[1]
        
        self.context_entropy_loss_avg=avg(self.context_entropy_loss_list)
        self.pattern_entropy_loss_avg=avg(self.pattern_entropy_loss_list)
        self.position_entropy_loss_avg=avg(self.position_entropy_loss_list)
        self.context_mul_loss_avg=avg(self.context_mul_loss_list)
        self.pattern_mul_loss_avg=avg(self.pattern_mul_loss_list)
        self.position_mul_loss_avg=avg(self.position_mul_loss_list)
        self.extract_loss_avg=avg(self.extract_loss_list)
        self.extract_loss_context_avg=avg(self.extract_loss_context_list)
        self.extract_loss_pattern_avg=avg(self.extract_loss_pattern_list)

        
        
        return context_disc_loss,pattern_disc_loss,position_disc_loss,total_loss,predicts_score
    
    
    
    
    
    
    
    
    def train_disc(self, batch_source_id_context,
                      batch_source_id_pattern,
                      batch_source_id_mask,
                      batch_source_id_mask_pattern,
                      batch_label,
                      batch_label_mask,
                      batch_clss,
                      batch_clss_mask,
                      context_bow,
                      pattern_bow,
                      position_bow):
        
        self.batch_clss_mask=batch_clss_mask.float()
        
        if self.config.add_prompt == 1:
            outputs_context = self.extractor(input_ids=batch_source_id_context,attention_mask=batch_source_id_mask)
            top_vec_context = outputs_context.last_hidden_state
            sents_vec_context = top_vec_context[torch.arange(top_vec_context.size(0)).unsqueeze(1), batch_clss, :] #(b,s,dim)    
            
            sents_vec_context = self.context_map(sents_vec_context)
            
            
            outputs_pattern = self.extractor(input_ids=batch_source_id_pattern,attention_mask=batch_source_id_mask_pattern)
            top_vec_pattern = outputs_pattern.last_hidden_state
            sents_vec_pattern = top_vec_pattern[torch.arange(top_vec_pattern.size(0)).unsqueeze(1), batch_clss, :] #(b,s,dim)    
            
            sents_vec_pattern = self.pattern_map(sents_vec_pattern)
        else:
            outputs = self.extractor(input_ids=batch_source_id_context,attention_mask=batch_source_id_mask)
            top_vec = outputs.last_hidden_state
            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), batch_clss, :] #(b,s,dim)    
            
            sents_vec_context = self.context_map(sents_vec)
            sents_vec_pattern = self.pattern_map(sents_vec)      
        
        
        #=========== Losses on context space =============#
        # Discriminator Loss
        context_disc_preds = self.context_disc(self.dropout(sents_vec_pattern.detach()))
        context_disc_loss = self.get_context_disc_loss(
            context_disc_preds, context_bow)


        #============ Losses on pattern space ================#
        # Discriminator Loss
        pattern_disc_preds = self.pattern_disc(self.dropout(sents_vec_context.detach()))
        pattern_disc_loss = self.get_pattern_disc_loss(
            pattern_disc_preds, pattern_bow)

        #============ Losses on pattern space-position ================#
        # Discriminator Loss
        position_disc_preds = self.position_disc(self.dropout(sents_vec_context.detach()))
        position_disc_loss = self.get_position_disc_loss(
            position_disc_preds, position_bow)


        self.context_disc_loss_list.append(context_disc_loss.item())
        self.pattern_disc_loss_list.append(pattern_disc_loss.item())
        self.position_disc_loss_list.append(position_disc_loss.item())
        
        self.context_disc_loss_list=self.context_disc_loss_list[-500:]
        self.pattern_disc_loss_list=self.pattern_disc_loss_list[-500:]
        self.position_disc_loss_list=self.position_disc_loss_list[-500:]
        
        def avg(listx):
            s=str(np.mean(listx))
            sp=s.split('e')
            if len(sp)==1:
                return s[:6]
            else:
                return sp[0][:6]+'e'+sp[1]

        self.context_disc_loss_avg=avg(self.context_disc_loss_list)
        self.pattern_disc_loss_avg=avg(self.pattern_disc_loss_list)
        self.position_disc_loss_avg=avg(self.position_disc_loss_list)
        
        
        return context_disc_loss,pattern_disc_loss,position_disc_loss 
    
    
    
    
    
        
    def inference(self, batch_source_id_context,
                      batch_source_id_pattern,
                      batch_source_id_mask,
                      batch_source_id_mask_pattern,
                      batch_label,
                      batch_label_mask,
                      batch_clss,
                      batch_clss_mask,                    
                      context_bow='',
                      pattern_bow='',
                      position_bow='',
                      return_representation=0):
        
        if self.config.add_prompt == 1:
            outputs_context = self.extractor(input_ids=batch_source_id_context,attention_mask=batch_source_id_mask)
            top_vec_context = outputs_context.last_hidden_state
            sents_vec_context = top_vec_context[torch.arange(top_vec_context.size(0)).unsqueeze(1), batch_clss, :] #(b,s,dim)    
            
            sents_vec_context = self.context_map(sents_vec_context)
            
            
            outputs_pattern = self.extractor(input_ids=batch_source_id_pattern,attention_mask=batch_source_id_mask_pattern)
            top_vec_pattern = outputs_pattern.last_hidden_state
            sents_vec_pattern = top_vec_pattern[torch.arange(top_vec_pattern.size(0)).unsqueeze(1), batch_clss, :] #(b,s,dim)    
            
            sents_vec_pattern = self.pattern_map(sents_vec_pattern)
        else:
            outputs = self.extractor(input_ids=batch_source_id_context,attention_mask=batch_source_id_mask)
            top_vec = outputs.last_hidden_state
            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), batch_clss, :] #(b,s,dim)    
            
            sents_vec_context = self.context_map(sents_vec)
            sents_vec_pattern = self.pattern_map(sents_vec)   
            
        if return_representation == 1:
            return sents_vec_context,sents_vec_pattern
        
        sigmoid = nn.Sigmoid()
        
        extractor_output=self.extract_classifier(torch.cat([sents_vec_context,sents_vec_pattern],2)).squeeze(2) 
        predicts_score=sigmoid(extractor_output)*batch_clss_mask      

        
        extractor_output_context=self.extract_classifier_context(sents_vec_context).squeeze(2) 
        predicts_score_context=sigmoid(extractor_output_context)*batch_clss_mask           
        
        extractor_output_pattern=self.extract_classifier_pattern(sents_vec_pattern).squeeze(2) 
        predicts_score_pattern=sigmoid(extractor_output_pattern)*batch_clss_mask      
        
        
        return predicts_score, predicts_score_context, predicts_score_pattern
        
    def get_entropy_loss(self, preds):
        """
        Returns the entropy loss: negative of the entropy present in the
        input distribution
        """
        return torch.mean(torch.sum(preds * torch.log(preds + self.config.eps), dim=2)*self.batch_clss_mask)
        
    def get_context_disc_preds(self, pattern_emb, detach = 1):
        """
        Returns predictions about the content using style embedding
        as input
        output shape : [batch_size,content_bow_dim]
        """

        if detach == 1:
            preds = self.context_disc(self.dropout(pattern_emb.detach()))
        else:
            preds = self.context_disc(self.dropout(pattern_emb))           
        return preds
     
    
    

    def get_context_disc_loss(self, context_disc_preds, context_bow):
        """
        It essentially quantifies the amount of information about content
        contained in the style space
        Returns:
        cross entropy loss of content discriminator
        """
        mask=torch.zeros(context_bow.size()).cuda()
        for i in range(self.batch_clss_mask.size()[0]):
            for j in range(self.batch_clss_mask.size()[1]):   
                mask[i][j]=self.batch_clss_mask[i][j]

        # calculate cross entropy loss
        context_disc_loss = nn.BCEWithLogitsLoss(weight=mask,pos_weight=torch.tensor([self.config.label_pos_weight*5]*context_bow.size()[-1]).cuda())(context_disc_preds, context_bow)
        return torch.mean(context_disc_loss)

    def get_pattern_disc_preds(self, context_emb, detach = 1):
        """
        Returns predictions about the content using style embedding
        as input
        output shape : [batch_size,content_bow_dim]
        """

        if detach == 1:
            preds = self.pattern_disc(self.dropout(context_emb.detach()))
        else:
            preds = self.pattern_disc(self.dropout(context_emb))            
        return preds

    def get_pattern_disc_loss(self, pattern_disc_preds, pattern_bow):
        """
        It essentially quantifies the amount of information about content
        contained in the style space
        Returns:
        cross entropy loss of content discriminator
        """
        mask=torch.zeros(pattern_bow.size()).cuda()
        for i in range(self.batch_clss_mask.size()[0]):
            for j in range(self.batch_clss_mask.size()[1]):   
                mask[i][j]=self.batch_clss_mask[i][j]

        pattern_disc_loss = nn.BCEWithLogitsLoss(weight=mask,pos_weight=torch.tensor([1]*pattern_bow.size()[-1]).cuda())(pattern_disc_preds, pattern_bow)

        return torch.mean(pattern_disc_loss)
    
    def get_position_disc_preds(self, context_emb, detach = 1):

        if detach == 1:
            preds = self.position_disc(self.dropout(context_emb.detach()))
        else:
            preds = self.position_disc(self.dropout(context_emb))         
        return preds

    def get_position_disc_loss(self, position_disc_preds, position_bow):
        mask=torch.zeros(position_bow.size()).cuda()
        for i in range(self.batch_clss_mask.size()[0]):
            for j in range(self.batch_clss_mask.size()[1]):   
                mask[i][j]=self.batch_clss_mask[i][j]
        # calculate cross entropy loss
        position_disc_loss = nn.BCEWithLogitsLoss(weight=mask,pos_weight=torch.tensor([self.config.label_pos_weight*1]*position_bow.size()[-1]).cuda())(position_disc_preds, position_bow)

        return torch.mean(position_disc_loss)
    
    def get_context_mul_loss(self, context_emb, context_bow):
        """
        This loss quantifies the amount of content information preserved
        in the content space
        Returns:
        cross entropy loss of the content classifier
        """
        mask=torch.zeros(context_bow.size()).cuda()
        for i in range(self.batch_clss_mask.size()[0]):
            for j in range(self.batch_clss_mask.size()[1]):   
                mask[i][j]=self.batch_clss_mask[i][j]
        # predictions
        preds = self.context_classifier(self.dropout(context_emb))
        # calculate cross entropy loss

        content_mul_loss = nn.BCEWithLogitsLoss(weight=mask,pos_weight=torch.tensor([self.config.label_pos_weight*5]*context_bow.size()[-1]).cuda())(preds, context_bow)

        return torch.mean(content_mul_loss)

    def get_pattern_mul_loss(self, pattern_emb, pattern_bow):
        """
        This loss quantifies the amount of content information preserved
        in the content space
        Returns:
        cross entropy loss of the content classifier
        """
        mask=torch.zeros(pattern_bow.size()).cuda()
        for i in range(self.batch_clss_mask.size()[0]):
            for j in range(self.batch_clss_mask.size()[1]):   
                mask[i][j]=self.batch_clss_mask[i][j]
        # predictions
        preds = self.pattern_classifier(self.dropout(pattern_emb))
        # calculate cross entropy loss

        pattern_mul_loss = nn.BCEWithLogitsLoss(weight=mask,pos_weight=torch.tensor([1]*pattern_bow.size()[-1]).cuda())(preds, pattern_bow)

        return torch.mean(pattern_mul_loss)
    
    def get_position_mul_loss(self, position_emb, position_bow):
        # predictions
        mask=torch.zeros(position_bow.size()).cuda()
        for i in range(self.batch_clss_mask.size()[0]):
            for j in range(self.batch_clss_mask.size()[1]):   
                mask[i][j]=self.batch_clss_mask[i][j]
        preds = self.position_classifier(self.dropout(position_emb))
        # calculate cross entropy loss
        position_mul_loss = nn.BCEWithLogitsLoss(weight=mask,pos_weight=torch.tensor([self.config.label_pos_weight*1]*position_bow.size()[-1]).cuda())(preds, position_bow)

        return torch.mean(position_mul_loss)
    
    def get_params(self):


        context_disc_params = self.context_disc.parameters()
        pattern_disc_params = self.pattern_disc.parameters()
        position_disc_params = self.position_disc.parameters()
        

        other_params =  list(self.extractor.parameters()) + \
                        list(self.extract_classifier.parameters()) + \
                        list(self.context_map.parameters()) + \
                        list(self.pattern_map.parameters()) + \
                        list(self.pattern_classifier.parameters()) + \
                        list(self.context_classifier.parameters()) + \
                        list(self.position_classifier.parameters()) + \
                        list(self.extract_classifier_context.parameters()) + \
                        list(self.extract_classifier_pattern.parameters())
          
        return context_disc_params, pattern_disc_params,position_disc_params, other_params
    
    
    def get_params_fewshot(self):

        params =  list(self.extract_classifier.parameters()) + \
                  list(self.pattern_map.parameters()) 
          
        return params