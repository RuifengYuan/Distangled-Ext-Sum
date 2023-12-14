import math
import torch
import torch.nn as nn
from transformers import BertModel,AutoTokenizer
import torch.nn.functional as F
import copy
import transformers
import numpy as np
import random
from collections import OrderedDict
class Disetangle_Extractor_Prompt(nn.Module):
    
    def __init__(self, config, load_path=''):
        """
        Initialize networks
        """
        super(Disetangle_Extractor_Prompt, self).__init__()
        # config
        self.config=config
        print('add prompt')
        # random seed
        seed = self.config.seed
        torch.manual_seed(seed)            
        torch.cuda.manual_seed(seed)      
        torch.cuda.manual_seed_all(seed)          
        
        #encoder
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_tokenizer)

        self.extractor = BertModel.from_pretrained(self.config.pretrained_model)
        self.context_map = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                         nn.LeakyReLU(),
                                         nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                         nn.LeakyReLU())
        self.pattern_map = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                         nn.LeakyReLU(),
                                         nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                         nn.LeakyReLU())        
        #=============== Discriminator/adversary============#
        self.MI_disc = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                     nn.LeakyReLU())

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


        #=============== Decoder =================#

        self.extract_classifier = nn.Sequential(nn.Linear(self.config.hidden_dim*2, self.config.hidden_dim*2),
                                                nn.LeakyReLU(),
                                                nn.Linear(self.config.hidden_dim*2, 1))

        self.context_classifier = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(self.config.hidden_dim, self.config.bow_dim))
        
        
        self.pattern_classifier = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(self.config.hidden_dim, self.config.bow_dim_pattern))
        
        self.position_classifier = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(self.config.hidden_dim, self.config.bow_dim_pos))
        
        
        self.extract_classifier_context = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(self.config.hidden_dim, 1))
        
        
        self.extract_classifier_pattern = nn.Sequential(nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(self.config.hidden_dim, 1))
        
        self.dropout=nn.Dropout(self.config.dropout)
        

        self.extract_loss_list=[]
        self.extract_loss_context_list=[]
        self.extract_loss_pattern_pred_list=[]
        self.extract_loss_pattern_list=[]
        self.MI_disc_loss_list=[]
        self.MI_loss_list=[]
        self.MI_loss_true_list=[]
        self.MI_loss_false_list=[]

        self.extract_loss_avg='0'
        self.extract_loss_context_avg='0'
        self.extract_loss_pattern_pred_avg='0'
        self.extract_loss_pattern_avg='0'
        self.MI_disc_loss_avg='0'
        self.MI_loss_avg='0'     
        self.MI_loss_true_avg=[]
        self.MI_loss_false_avg=[]
        
        
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
        

        #============ Losses on MI_disc ================#
        kl_loss = nn.KLDivLoss(reduce=False)
        
        disc_loss=kl_loss(F.softmax(self.MI_disc(sents_vec_context.detach()),dim=2).log(), F.softmax(sents_vec_pattern.detach(),dim=2))
        
        #print(torch.sum(F.softmax(self.MI_disc(sents_vec_pattern.detach()),dim=2),dim=2)[0][0])
        
        pred_pattern=self.MI_disc(sents_vec_context)
        pred_loss_true=kl_loss(F.softmax(pred_pattern,dim=2).log(), F.softmax(sents_vec_pattern,dim=2))
        

        shuffle_idx=torch.zeros(batch_label_mask.size()).long()
        #shuffle_b=torch.arange(sents_vec_pattern.size(0))
        for b in range(batch_label_mask.size()[0]):
            #random_b=random.randint(0, sents_vec_pattern.size(0)-1)
            #while random_b == b:
                    #random_b=random.randint(0, sents_vec_pattern.size(0)-1)
            #shuffle_b[b]=random_b
            effective_len=int(torch.sum(batch_label_mask[b]))
            for s in range(batch_label_mask.size()[1]):
                
                if (effective_len-1) >= 1:
                    random_idx=random.randint(0, effective_len-1)
                    while random_idx == s:
                        random_idx=random.randint(0, effective_len-1)
                else:
                    random_idx=0
                    
                shuffle_idx[b][s]=random_idx
        shuffle_idx=shuffle_idx.cuda()        
        #shuffle_b=shuffle_b.cuda()


        sents_vec_context_sample=sents_vec_pattern[torch.arange(sents_vec_context.size(0)).unsqueeze(1),shuffle_idx,:]
        #sents_vec_pattern_sample=sents_vec_pattern[shuffle_b,:,:]
        
        
        pred_pattern_sample=self.MI_disc(sents_vec_context_sample) 
        pred_loss_sample=kl_loss(F.softmax(pred_pattern_sample,dim=2).log(), F.softmax(sents_vec_pattern,dim=2))
        

        disc_loss=torch.mean(disc_loss, 2)*batch_label_mask 
        pred_loss_true=torch.mean(pred_loss_true, 2)*batch_label_mask
        pred_loss_sample=torch.mean(pred_loss_sample, 2)*batch_label_mask   

        disc_loss=torch.mean(disc_loss)
        pred_loss_true=torch.mean(pred_loss_true)
        pred_loss_sample=torch.mean(pred_loss_sample)
        
        MI_loss= torch.min(pred_loss_true - pred_loss_sample,torch.tensor([0.0]).cuda())

        #print(pred_loss_true.item(), pred_loss_sample.item())
        #============ Losses on test_disc ================#
        
        
        mask_context=torch.zeros(context_bow.size()).cuda()
        for i in range(self.batch_clss_mask.size()[0]):
            for j in range(self.batch_clss_mask.size()[1]):   
                mask_context[i][j]=self.batch_clss_mask[i][j]
        # predictions
        disc_on_context = self.context_disc(self.dropout(sents_vec_pattern.detach()))
        # calculate cross entropy loss
        disc_loss_on_context = nn.BCEWithLogitsLoss(weight=mask_context,pos_weight=torch.tensor([self.config.label_pos_weight*5]*context_bow.size()[-1]).cuda())(disc_on_context, context_bow)

        
        
        mask_pattern=torch.zeros(pattern_bow.size()).cuda()
        for i in range(self.batch_clss_mask.size()[0]):
            for j in range(self.batch_clss_mask.size()[1]):   
                mask_pattern[i][j]=self.batch_clss_mask[i][j]
        # predictions
        disc_on_pattern = self.pattern_disc(self.dropout(sents_vec_context.detach()))
        # calculate cross entropy loss
        disc_loss_on_pattern = nn.BCEWithLogitsLoss(weight=mask_pattern,pos_weight=torch.tensor([self.config.label_pos_weight*1]*pattern_bow.size()[-1]).cuda())(disc_on_pattern, pattern_bow)

        
        mask_position=torch.zeros(position_bow.size()).cuda()
        for i in range(self.batch_clss_mask.size()[0]):
            for j in range(self.batch_clss_mask.size()[1]):   
                mask_position[i][j]=self.batch_clss_mask[i][j]
        # predictions
        disc_on_position = self.position_disc(self.dropout(sents_vec_context.detach()))
        # calculate cross entropy loss
        disc_loss_on_position = nn.BCEWithLogitsLoss(weight=mask_position,pos_weight=torch.tensor([self.config.label_pos_weight*2]*position_bow.size()[-1]).cuda())(disc_on_position, position_bow)        
        

        #============ Losses on extracted summarization ================#
        sent_len=batch_clss.size()[1]
        p_weight=1
        lossFunc = nn.BCEWithLogitsLoss(weight=batch_clss_mask.float(),pos_weight=(torch.tensor([p_weight]*sent_len)).cuda())
        
        if self.config.pattern_detach == 0:
            extractor_output=self.extract_classifier(torch.cat([sents_vec_context,sents_vec_pattern],2)).squeeze(2) 
        else:
            extractor_output=self.extract_classifier(torch.cat([sents_vec_context,sents_vec_pattern.detach()],2)).squeeze(2) 
            
        sigmoid = nn.Sigmoid()
        predicts_score=sigmoid(extractor_output)*batch_clss_mask      
        extract_loss = lossFunc(extractor_output,batch_label) 
        
        extractor_output_context=self.extract_classifier_context(sents_vec_context.detach()).squeeze(2) 
        extract_context = lossFunc(extractor_output_context,batch_label)         
        
        extractor_output_pattern=self.extract_classifier_pattern(sents_vec_pattern.detach()).squeeze(2) 
        extract_pattern = lossFunc(extractor_output_pattern,batch_label)  

        
        mask_context=torch.zeros(context_bow.size()).cuda()
        for i in range(self.batch_clss_mask.size()[0]):
            for j in range(self.batch_clss_mask.size()[1]):   
                mask_context[i][j]=self.batch_clss_mask[i][j]
        # predictions
        preds_context = self.context_classifier(self.dropout(sents_vec_context))
        # calculate cross entropy loss
        extract_loss_context = nn.BCEWithLogitsLoss(weight=mask_context,pos_weight=torch.tensor([self.config.label_pos_weight*5]*context_bow.size()[-1]).cuda())(preds_context, context_bow)
        
        
        mask_pattern=torch.zeros(pattern_bow.size()).cuda()
        for i in range(self.batch_clss_mask.size()[0]):
            for j in range(self.batch_clss_mask.size()[1]):   
                mask_pattern[i][j]=self.batch_clss_mask[i][j]
        # predictions
        preds_pattern = self.pattern_classifier(self.dropout(sents_vec_pattern))
        # calculate cross entropy loss
        extract_loss_pattern = nn.BCEWithLogitsLoss(weight=mask_pattern,pos_weight=torch.tensor([self.config.label_pos_weight*1]*pattern_bow.size()[-1]).cuda())(preds_pattern, pattern_bow)
        
        mask_position=torch.zeros(position_bow.size()).cuda()
        for i in range(self.batch_clss_mask.size()[0]):
            for j in range(self.batch_clss_mask.size()[1]):   
                mask_position[i][j]=self.batch_clss_mask[i][j]
        # predictions
        preds_position = self.position_classifier(self.dropout(sents_vec_pattern))
        # calculate cross entropy loss
        extract_loss_position = nn.BCEWithLogitsLoss(weight=mask_position,pos_weight=torch.tensor([self.config.label_pos_weight*2]*position_bow.size()[-1]).cuda())(preds_position, position_bow)        
   

        mask_pattern=torch.zeros(pattern_bow.size()).cuda()
        for i in range(self.batch_clss_mask.size()[0]):
            for j in range(self.batch_clss_mask.size()[1]):   
                mask_pattern[i][j]=self.batch_clss_mask[i][j]
        # predictions
        preds_pattern_pred = nn.Sigmoid()(self.pattern_classifier(self.dropout(pred_pattern)))
        # calculate cross entropy loss
        extract_loss_pattern_pred = nn.BCELoss(weight=mask_pattern)(preds_pattern_pred, pattern_bow)        
        #================ total weighted loss ==========#
        
        extract_loss =  extract_loss*self.config.extract_main_weight+(extract_context+extract_pattern)*self.config.extract_detach_weight

        

        self.extract_loss_list.append(extract_loss.item())
        self.extract_loss_context_list.append(extract_loss_context.item())
        self.extract_loss_pattern_pred_list.append(extract_loss_pattern_pred.item())
        self.extract_loss_pattern_list.append(extract_loss_pattern.item())
        self.MI_loss_list.append(MI_loss.item())
        self.MI_loss_true_list.append(pred_loss_true.item())
        self.MI_loss_false_list.append(pred_loss_sample.item())
        
        self.extract_loss_list=self.extract_loss_list[-500:]
        self.extract_loss_context_list=self.extract_loss_context_list[-500:]
        self.extract_loss_pattern_pred_list=self.extract_loss_pattern_pred_list[-500:]
        self.extract_loss_pattern_list=self.extract_loss_pattern_list[-500:]
        self.MI_loss_list=self.MI_loss_list[-500:]
        self.MI_loss_true_list=self.MI_loss_true_list[-500:]
        self.MI_loss_false_list=self.MI_loss_false_list[-500:]  
        
        def avg(listx):
            s=str(np.mean(listx))
            sp=s.split('e')
            if len(sp)==1:
                return s[:6]
            else:
                return sp[0][:6]+'e'+sp[1]

        self.extract_loss_avg=avg(self.extract_loss_list)
        self.extract_loss_context_avg=avg(self.extract_loss_context_list)
        self.extract_loss_pattern_pred_avg=avg(self.extract_loss_pattern_pred_list)
        self.extract_loss_pattern_avg=avg(self.extract_loss_pattern_list)
        self.MI_loss_avg=avg(self.MI_loss_list)
        self.MI_loss_true_avg=avg(self.MI_loss_true_list)
        self.MI_loss_false_avg=avg(self.MI_loss_false_list)
        return extract_loss, extract_loss_position + extract_loss_context + extract_loss_pattern + disc_loss_on_context+disc_loss_on_pattern+disc_loss_on_position, MI_loss, disc_loss,predicts_score,sents_vec_context,sents_vec_pattern
    
    
    
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
        

        #============ Losses on MI_disc ================#
        kl_loss = nn.KLDivLoss(reduce=False)
        
        disc_loss=kl_loss(F.softmax(self.MI_disc(sents_vec_context.detach()),dim=2).log(), F.softmax(sents_vec_pattern.detach(),dim=2))


        disc_loss=torch.mean(disc_loss, 2)*batch_label_mask 
        disc_loss=torch.mean(disc_loss)

        


        self.MI_disc_loss_list.append(disc_loss.item())

        self.MI_disc_loss_list=self.MI_disc_loss_list[-500:]

        
        def avg(listx):
            s=str(np.mean(listx))
            sp=s.split('e')
            if len(sp)==1:
                return s[:6]
            else:
                return sp[0][:6]+'e'+sp[1]

        self.MI_disc_loss_avg=avg(self.MI_disc_loss_list)

        return disc_loss
        
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
                      position_bow=''):
        
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
        
        sigmoid = nn.Sigmoid()
        
        extractor_output=self.extract_classifier(torch.cat([sents_vec_context,sents_vec_pattern],2)).squeeze(2) 
        predicts_score=sigmoid(extractor_output)*batch_clss_mask      
        
        
        extractor_output_context=self.extract_classifier_context(sents_vec_context).squeeze(2) 
        predicts_score_context=sigmoid(extractor_output_context)*batch_clss_mask           
        
        extractor_output_pattern=self.extract_classifier_pattern(sents_vec_pattern).squeeze(2) 
        predicts_score_pattern=sigmoid(extractor_output_pattern)*batch_clss_mask      
        
        
        return predicts_score, predicts_score_context, predicts_score_pattern        

        

    
    def get_params(self):

        MI_disc_params = self.MI_disc.parameters()
        
        
        other_params =  list(self.extractor.parameters()) + \
                        list(self.context_map.parameters()) + \
                        list(self.pattern_map.parameters()) + \
                        list(self.extract_classifier.parameters()) + \
                        list(self.context_classifier.parameters()) + \
                        list(self.pattern_classifier.parameters()) + \
                        list(self.position_classifier.parameters()) + \
                        list(self.extract_classifier_context.parameters()) + \
                        list(self.extract_classifier_pattern.parameters()) + \
                        list(self.pattern_disc.parameters()) + \
                        list(self.position_disc.parameters()) + \
                        list(self.context_disc.parameters())  
                        
                        
        return MI_disc_params, other_params
    
    
    def get_params_fewshot(self):



        params =  list(self.extract_classifier.parameters()) + \
                  list(self.pattern_map.parameters()) 

          
        return params