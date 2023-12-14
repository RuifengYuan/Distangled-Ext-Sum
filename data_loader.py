import glob
import random
import struct
import json
import re
import torch
import csv
import argparse
from transformers import BartTokenizer
from rouge import Rouge
import numpy as np
import nltk
nltk.download('averaged_perceptron_tagger')

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",

         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"', "\n": '',":":',',"\'":"'",'<s>':'','</s>':''}

def clean(x):
    return re.sub(r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''|\n|:", lambda m: REMAP.get(m.group()), x)

def clean_str(s):
    forbidden=['b"','-lrb-','-rrb-','-','“','"',"'","`",'``',"''","b'",'/','\\','\\n','-','<s>','</s>']
    for i in forbidden:
        s=s.replace(i,'')
    return s

def end_replace(s):
    forbidden=['!', '?',';']

    for i in forbidden:
        s=s.replace(i,'.')
   
    return s






class data_loader():

    def __init__(self, part, config, tokenizer, load_dataset='', load_wrong_context=0, only_pattern=0, add_prompt=0, few_shot=0):

        super(data_loader,self).__init__()  
      
        self.part=part
        self.tokenizer=tokenizer
        self.config=config
        self.raw_rouge=Rouge()
        self.add_prompt=add_prompt
        self.few_shot=few_shot
        
        self.count=0
        self.epoch=0

        self.max_epoch=config.max_epoch
        self.buffer_size=config.buffer_size
        self.batch_size=config.batch_size
        self.true_batch_size=config.true_batch_size
        self.max_article=config.max_article
        self.max_summary=config.max_summary
        self.load_wrong_context=load_wrong_context
        self.only_pattern=only_pattern
        '''
        self.max_epoch=1
        self.buffer_size=128
        self.batch_size=4
        self.max_article=512
        self.max_summary=100
        '''
        self.load_dataset=load_dataset

        if self.load_dataset == 'cnndm':
            
            if self.part == 'train':
                self.cnndm_ext_source = 'data/DMCNN/train_source.txt'
                self.cnndm_ext_target = 'data/DMCNN/train_target.txt'
                self.cnndm_ext_gold = 'data/DMCNN/train_gold.txt'
            if self.part == 'test':
                self.cnndm_ext_source = 'data/DMCNN/test_source.txt'
                self.cnndm_ext_target = 'data/DMCNN/test_target.txt'        
                self.cnndm_ext_gold = 'data/DMCNN/test_gold.txt'
            if self.part == 'val':
                self.cnndm_ext_source = 'data/DMCNN/val_source.txt'
                self.cnndm_ext_target = 'data/DMCNN/val_target.txt' 
                self.cnndm_ext_gold = 'data/DMCNN/val_gold.txt'
 

        if self.load_dataset == 'cnndm':
            self.data_generator=self.next_data_cnndm_ext()
            
        if self.load_dataset == 'cnndm':
            with open('data/pattern_cnndm.txt', 'r') as load_f:
                data=load_f.readlines()
            self.pattern_list=[x.strip('\n').split('<s>') for x in data]
            if self.config.pattern_binary==0:
                self.pattern_list=self.pattern_list[:self.config.bow_dim_pattern]
            else:
                self.pattern_list=self.pattern_list[:500] 
            
        self.batch_generator=self.next_batch()
        
        
    def next_data_cnndm_ext(self):
        buffer=[]
        for epoch in range(self.max_epoch):
            self.epoch=self.epoch+1
            reader_src = open(self.cnndm_ext_source, 'r')
            reader_tar = open(self.cnndm_ext_target, 'r')       
            reader_gold = open(self.cnndm_ext_gold, 'r') 
            src=reader_src.readlines()
            tar=reader_tar.readlines()       
            gold=reader_gold.readlines()
            
            
            sample_list=[]
            
            for i in range(len(src)):
                sample_list.append([src[i],tar[i],gold[i]])
                
            if self.few_shot==1:
                sample_list=sample_list[self.config.sample_start_pos:self.config.sample_start_pos+self.config.sample_num]
                
            if self.part == 'train':
                random.seed(self.config.seed+epoch)
                random.shuffle(sample_list)
            else:
                pass
            
            for data_point in sample_list:

                article_text=data_point[0]
                ext_text=data_point[1]
                gold_text=data_point[2]

                article=article_text.split('<s>')
                ext_summary=ext_text.split('<s>')
                
                article=[s.strip() for s in article]
                ext_summary=[s.strip() for s in ext_summary]
                
                article=[s for s in article if len(s)>5]
                ext_summary=[s for s in ext_summary if len(s)>5]                
                
                ext_index=[]
                for s in article:
                    if s in ext_summary:
                        ext_index.append(1)
                    else:
                        ext_index.append(0)
                        
                article_id=[]
                article_len=0
                for s in article:
                    sent_token=self.tokenizer.encode(s)
                    article_len+=len(sent_token)
                    article_id.append(sent_token)

                if self.part == 'train':
                    if len(article)<3:
                        pass
                    else:
                        buffer.append((article,article_id,article_len,ext_summary,ext_index,gold_text))   
                else:
                    buffer.append((article,article_id,article_len,ext_summary,ext_index,gold_text))   
                        
                if len(buffer) == self.buffer_size:
                    if self.load_wrong_context==1:
                        new_buffer=[]
                        for d in range(len(buffer)):
                            article,article_id,article_len,ext_summary,ext_index,gold_text=buffer[d]
                            wrong_article_id=[]
                            for idx in range(len(ext_index)):
                                rand_d=random.randint(0, len(buffer)-1)
                                rand_s=random.randint(0, len(buffer[rand_d][1])-1)
                                
                                original_len=len(article_id[idx])
                                
                                if ext_index[idx]==0:
                                    wrong_one=buffer[rand_d][1][rand_s]

                                    wrong_len=len(wrong_one)
                                    while(wrong_len<original_len):
                                        rand_d_add=random.randint(0, len(buffer)-1)
                                        rand_s_add=random.randint(0, len(buffer[rand_d_add][1])-1)
                                        wrong_one_add=buffer[rand_d_add][1][rand_s_add]
                                        wrong_one=wrong_one[:-1]+wrong_one_add[1:]
                                        wrong_len=len(wrong_one)
                                    wrong_one=wrong_one[:original_len]

                                    wrong_article_id.append(wrong_one)
                                else:
                                    wrong_article_id.append(article_id[idx])

                            new_buffer.append((article,wrong_article_id,article_len,ext_summary,ext_index,gold_text))   
                        yield new_buffer
                        buffer=[]        
                    else:
                        yield buffer
                        buffer=[]
                        
        print ("data_generator completed reading all datafiles for all epoches. No more data.")                       
        return 0


    def next_batch(self):
        while(True):
            buffer = self.data_generator.__next__()
            buffer.sort(key=self.get_sort)
            
            for batch_idx in range(int(len(buffer)/self.batch_size)):
                batch_data=buffer[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                
                batch_source_id=[]
                batch_source_id_two=[]
                batch_clss=[]
                batch_source=[]
                batch_ext_summary=[]
                batch_abs_summary=[]
                batch_label=[]      
                batch_context_bow=[]
                batch_pattern_bow=[]
                batch_position_bow=[]
                
                for data_point in batch_data:
                    cls_token=self.config.bos_token_id
                    article,article_id,article_len,ext_summary,ext_index,gold_text=data_point
                    source_id=[]
                    source_id_two=[]
                    for s in article_id:
                        if self.add_prompt==0:
                            source_id+=s
                            source_id_two=source_id
                        else:
                            source_id+=[cls_token]+[21,22,23,24]+s[1:]
                            source_id_two+=[cls_token]+[31,32,33,34]+s[1:]
                    source_id=source_id[:self.config.max_article]
                    source_id_two=source_id_two[:self.config.max_article]
                    
                    
                    clss=[]
                    for t_dix,token in enumerate(source_id):
                        if token == cls_token:
                            clss.append(t_dix)
                    
                    label=ext_index[:len(clss)]

                    assert  len(clss) == len(label),'sentence number is not correct between label and input'
                    
                    context_bow,pattern_bow,position_bow=self.get_bow_representations(article[:len(clss)])

                    batch_source_id.append(source_id)
                    batch_source_id_two.append(source_id_two)
                    batch_clss.append(clss)
                    batch_label.append(label)
                    batch_source.append(article)
                    batch_ext_summary.append(ext_summary)
                    batch_abs_summary.append(gold_text)     
                    
                    batch_context_bow.append(context_bow)
                    batch_pattern_bow.append(pattern_bow)
                    batch_position_bow.append(position_bow)
                    
                batch_source_id,batch_source_id_mask=self.pad_with_mask(batch_source_id, pad_id=self.config.pad_token_id)
                batch_source_id=torch.tensor(batch_source_id)
                batch_source_id_mask=torch.tensor(batch_source_id_mask) 
                
                batch_source_id_two,_=self.pad_with_mask(batch_source_id_two, pad_id=self.config.pad_token_id)
                batch_source_id_two=torch.tensor(batch_source_id_two)
                
                batch_source_id_mask_pattern = self.get_mask_for_pattern(batch_source_id, batch_source_id_mask, batch_clss)
                batch_source_id_mask_pattern=batch_source_id_mask_pattern.cuda()
                if self.only_pattern == 1:
                    batch_source_id_mask=batch_source_id_mask_pattern
                batch_clss,batch_clss_mask=self.pad_with_mask(batch_clss, pad_id=self.config.pad_token_id)
                batch_clss=torch.tensor(batch_clss)
                batch_clss_mask=torch.tensor(batch_clss_mask)  
                
                batch_label,batch_label_mask=self.pad_with_mask(batch_label, pad_id=self.config.pad_token_id)
                batch_label=torch.tensor(batch_label)
                batch_label_mask=torch.tensor(batch_label_mask)                          
                
                batch_source_id=batch_source_id.cuda()
                batch_source_id_two=batch_source_id_two.cuda()
                batch_source_id_mask=batch_source_id_mask.cuda()
                batch_clss=batch_clss.long().cuda()
                batch_clss_mask=batch_clss_mask.cuda()
                batch_label=batch_label.float().cuda()
                batch_label_mask=batch_label_mask.cuda()   
                
                batch_context_bow=torch.stack([s[:batch_clss.size()[1],:] for s in batch_context_bow],dim=0).cuda()
                batch_pattern_bow=torch.stack([s[:batch_clss.size()[1],:] for s in batch_pattern_bow],dim=0).cuda()
                batch_position_bow=torch.stack([s[:batch_clss.size()[1],:] for s in batch_position_bow],dim=0).cuda()   

                if self.add_prompt == 0:
                    return_data = [batch_source_id,
                           batch_source_id,
                           batch_source_id_mask,
                           batch_source_id_mask,
                           batch_label,
                           batch_label_mask,
                           batch_clss,
                           batch_clss_mask,
                           batch_source,
                           batch_ext_summary,
                           batch_abs_summary,
                           batch_context_bow,
                           batch_pattern_bow,
                           batch_position_bow] 
                else:
                    return_data = [batch_source_id,
                           batch_source_id_two,
                           batch_source_id_mask,
                           batch_source_id_mask_pattern,
                           batch_label,
                           batch_label_mask,
                           batch_clss,
                           batch_clss_mask,
                           batch_source,
                           batch_ext_summary,
                           batch_abs_summary,
                           batch_context_bow,
                           batch_pattern_bow,
                           batch_position_bow]    
                    
                yield return_data

    def load_data(self):
        self.count=self.count+1
        return self.batch_generator.__next__()
                    
            
    def get_sort(self, x):
        return x[2]


    def pad_with_mask(self, data, pad_id=0, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
            
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        
        pad_mask = [[1] * len(d) + [0] * (width - len(d)) for d in data]
        return rtn_data,pad_mask 

    def word_freq(self, article, only_noun=0):
        allowed_tag=['NN','NNS','NNP','NNPS','FW']
        token=[]
        for s in article:
            text=nltk.word_tokenize(s)
            
            if only_noun == 1:
                text_tag=nltk.pos_tag(text)
                fliter_text=[]
                for w in text_tag:
                    if w[1] in allowed_tag:
                        fliter_text.append(w[0])
            else:
                fliter_text=text
                    
            token+=fliter_text
        token_freq=nltk.FreqDist(token)
        
        stop_list = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
                     'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
                     'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
                     'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
                     'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
                     'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in',
                     'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not',
                     'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i',
                     'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further',
                     'was', 'here', 'than', ',', '.', "'",'"','’','“',"?","!","@","<s>",'</s>',"(",")","{","}","[","]","*","$","#","%",
                     "-","+","=","_",":",";","``","'s","n't",'would','could','should','shall'}
        fliter_token_freq=[]
        fliter_token=[]
        for k,v in token_freq.items():
            if k not in stop_list:
                fliter_token_freq.append((k,v))
                fliter_token.append(k)
            
        return fliter_token_freq,fliter_token

    def get_mask_for_pattern(self, batch_source_id, batch_source_id_mask, clss):
        batch,seq_len=batch_source_id.size()
        
        batch_source_id_mask_pattern=torch.zeros([batch,seq_len,seq_len])
        for i in range(batch):
            one_clss=clss[i]+[int(torch.sum(batch_source_id_mask[i]))]
            for j in range(len(one_clss)-1):
                start=one_clss[j]
                end=one_clss[j+1]
                batch_source_id_mask_pattern[i:i+1, start:end, start:end]=1
            
        return batch_source_id_mask_pattern



    def get_bow_representations(self, article):
        """
        Returns BOW representation of sentence
        """

        context_bow_representation = torch.zeros([100,self.config.bow_dim], dtype=torch.float)
        if self.config.pattern_binary==0:
            pattern_bow_representation = torch.zeros(
                [100,self.config.bow_dim_pattern], dtype=torch.float)
        else:
            pattern_bow_representation = torch.zeros([100,1], dtype=torch.float)     
        position_bow_representation = torch.zeros([100,self.config.bow_dim_pos], dtype=torch.float)        
        
        c_c=0
        c_p=0
        for idx,s in enumerate(article):
            #position representation
            if idx < self.config.bow_dim_pos:
                position_bow_representation[idx][idx]=1
            else:
                position_bow_representation[idx][-1]=1
                
            #context representation
            negbor_freq,negbor_token=self.word_freq(article[max(idx-3,0):min(idx+4,len(article)-1)])
            target_freq,target_token=self.word_freq([article[idx]])
            
            context_freq=[]
            context_token=[]
            for t in negbor_freq:
                if t[1]>1:
                    if t[0] in target_token:
                        context_freq.append(t)
                        context_token.append(t[0])
            context_token_list=[]
            for t in context_freq:
                context_token_list+=[t[0]]*t[1]
            context_token_list=' '.join(context_token_list)
            context_token_list=self.tokenizer.encode(context_token_list)[1:-1]
            for t in context_token_list:
                context_bow_representation[idx][t]=1
                

            #pattern representation
            if self.config.pattern_binary==0:
                tag=0
                for t in range(len(self.pattern_list)):
                    if self.pattern_list[t][0].strip() in s:
                        tag=1
                        pattern_bow_representation[idx][t]=1  
            else:
                tag=0
                for t in range(len(self.pattern_list)):
                    if self.pattern_list[t][0] in s:
                        tag=1
                        pattern_bow_representation[idx][0]=1          
                        break 
                    

        
        
        return context_bow_representation,pattern_bow_representation,position_bow_representation