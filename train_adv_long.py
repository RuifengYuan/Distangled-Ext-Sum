import time,copy
import argparse
import math
import torch
import torch.nn as nn
from rouge import Rouge
from transformers import BertModel,AutoTokenizer
from transformers import AdamW
from torch.optim import *
import numpy as np
from model.model import *
from model.model_adv import *
from model.baseline import *
import data_loader_long 
import data_loader
import os 
from tqdm import tqdm
class Train(object):

    def __init__(self, config):
        self.config = config  
        
        seed = self.config.seed
        torch.manual_seed(seed)           
        torch.cuda.manual_seed(seed)      
        torch.cuda.manual_seed_all(seed)        
        

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_tokenizer)
        self.log = open('log.txt','w')
        self.dataloader=data_loader_long.data_loader('train', self.config, self.tokenizer, 'extractive', load_qmsum=0, load_arxiv=1, only_pattern=0, return_bow=1, add_prompt=self.config.add_prompt)   


        if self.config.mid_start == 0:
            #if self.config.add_prompt == 0:
                #self.generator_raw = Disetangle_Extractor(self.config)
            #else:
            self.generator_raw = Disetangle_Extractor_Prompt(self.config)
        else:    
            x=torch.load('save_model/'+self.config.load_model,map_location='cpu')
            self.generator_raw = x['generator']
            
            
        self.generator_raw.cuda()

        if self.config.multi_gpu==1:
            gpus=[int(gpu) for gpu in self.config.multi_device.split(',')]
            self.generator = nn.DataParallel(self.generator_raw,device_ids=gpus, output_device=gpus[0])
            self.generator.cuda()
        else:
            self.generator=self.generator_raw
            self.generator.cuda()  


        context_disc_params, pattern_disc_params,position_disc_params, main_params = self.generator_raw.get_params()  
        self.context_disc_optimizer = torch.optim.Adam(
            context_disc_params, lr=self.config.lr)
        self.pattern_disc_optimizer = torch.optim.Adam(
            pattern_disc_params, lr=self.config.lr)
        self.position_disc_optimizer = torch.optim.Adam(
            position_disc_params, lr=self.config.lr)
        self.main_optimizer = torch.optim.Adam(
            main_params, lr=self.config.lr)        

        if self.config.use_lr_decay == 1:
            scheduler=lr_scheduler.StepLR(self.optimizer,step_size=1,gamma = self.config.lr_decay)
            



        
    def save_model(self, running_avg_loss,loss_list,rouge1,rouge2,loss_text=''):

        state = {
            'iter': self.dataloader.count/self.config.true_batch_size*self.config.batch_size,
            'ecop': self.dataloader.epoch,
            'generator':self.generator,
            'current_loss': running_avg_loss,
            'loss_list': loss_list,
            'rouge1':rouge1,
            'rouge2':rouge2,
            'config':self.config
        }
        
        #try:
        model_save_path = self.config.save_path+str(self.dataloader.count/self.config.true_batch_size*self.config.batch_size)+'_iter_'+str(self.dataloader.epoch) +'_ep__r_'+str(rouge1)+'_'+str(rouge2)+'__l_'+str(running_avg_loss)[:4]+loss_text
        torch.save(state, model_save_path)
        #except:
            #print('can not save the model!!!')
        
    def train_one_batch(self, data):

        batch_source_id_context, batch_source_id_pattern, batch_source_id_mask,batch_source_id_mask_pattern, batch_clss,batch_clss_mask,batch_label,batch_weight,batch_source,batch_query, batch_context_bow, batch_pattern_bow, batch_position_bow = \
        data
            
            
        if self.config.add_prompt == 0:
            if self.config.pattern_mask == 0:
                batch_source_id_mask_pattern=batch_source_id_mask
            context_disc_loss,pattern_disc_loss,position_disc_loss,total_loss,predicts_score = self.generator(batch_source_id_context,
                                                                                                batch_source_id_context,
                                                                                                batch_source_id_mask,
                                                                                                batch_source_id_mask,
                                                                                                batch_label,
                                                                                                batch_clss_mask,
                                                                                                batch_clss,
                                                                                                batch_clss_mask,
                                                                                                batch_context_bow,
                                                                                                batch_pattern_bow,
                                                                                                batch_position_bow)     
        
        else:
            if self.config.pattern_mask == 0:
                batch_source_id_mask_pattern=batch_source_id_mask
            context_disc_loss,pattern_disc_loss,position_disc_loss,total_loss,predicts_score = self.generator(batch_source_id_context,
                                                                                                batch_source_id_pattern,
                                                                                                batch_source_id_mask,
                                                                                                batch_source_id_mask_pattern,
                                                                                                batch_label,
                                                                                                batch_clss_mask,
                                                                                                batch_clss,
                                                                                                batch_clss_mask,
                                                                                                batch_context_bow,
                                                                                                batch_pattern_bow,
                                                                                                batch_position_bow)     
        

        context_disc_loss=torch.mean(context_disc_loss)*self.config.context_disc_loss_weight*(self.config.batch_size/self.config.true_batch_size)
        pattern_disc_loss=torch.mean(pattern_disc_loss)*self.config.pattern_disc_loss_weight*(self.config.batch_size/self.config.true_batch_size)
        position_disc_loss=torch.mean(position_disc_loss)*self.config.position_disc_loss_weight*(self.config.batch_size/self.config.true_batch_size)
        total_loss=torch.mean(total_loss)*(self.config.batch_size/self.config.true_batch_size)
        #sentence recall
        recall_all=[]
        try:
            recall_all=[]
            for i in range(len(batch_label)):
                
                select_number=int(torch.sum(batch_label[i]))
                gold_top=torch.topk(batch_label[i], select_number).indices
                pred_top=torch.topk(predicts_score[i], select_number).indices
                gold_top=gold_top.tolist()
                pred_top=pred_top.tolist()
                c=0
                for j in gold_top:
                    if j in pred_top:
                        c+=1
                if len(gold_top) == 0:
                    continue
                recall=c/len(gold_top)
                recall_all.append(recall)
        except:
            print("infer fail")
            recall_all=[0]

        total_loss.backward()
        
        #print('total_loss',str(total_loss.item())[:4],'  ','context_disc_loss',str(context_disc_loss.item())[:4],'  ','pattern_disc_loss',str(pattern_disc_loss.item())[:4],'  ','position_disc_loss',str(position_disc_loss.item())[:4])
        
        
        if np.isnan(np.mean(recall_all)) == 0:
            return context_disc_loss.item()+pattern_disc_loss.item()+position_disc_loss.item(),total_loss.item(),np.mean(recall_all), 1
        if np.isnan(np.mean(recall_all)) == 1:
            return context_disc_loss.item()+pattern_disc_loss.item()+position_disc_loss.item(),total_loss.item(),0, 1    
        
    def train_one_disc(self, data):

        batch_source_id_context, batch_source_id_pattern, batch_source_id_mask,batch_source_id_mask_pattern, batch_clss,batch_clss_mask,batch_label,batch_weight,batch_source,batch_query, batch_context_bow, batch_pattern_bow, batch_position_bow = \
        data
            
            
        if self.config.add_prompt == 0:
            if self.config.pattern_mask == 0:
                batch_source_id_mask_pattern=batch_source_id_mask
            context_disc_loss,pattern_disc_loss,position_disc_loss = self.generator.train_disc(batch_source_id_context,
                                                                                                batch_source_id_context,
                                                                                                batch_source_id_mask,
                                                                                                batch_source_id_mask,
                                                                                                batch_label,
                                                                                                batch_clss_mask,
                                                                                                batch_clss,
                                                                                                batch_clss_mask,
                                                                                                batch_context_bow,
                                                                                                batch_pattern_bow,
                                                                                                batch_position_bow)  
        else:
            if self.config.pattern_mask == 0:
                batch_source_id_mask_pattern=batch_source_id_mask
            context_disc_loss,pattern_disc_loss,position_disc_loss = self.generator.train_disc(batch_source_id_context,
                                                                                                batch_source_id_pattern,
                                                                                                batch_source_id_mask,
                                                                                                batch_source_id_mask_pattern,
                                                                                                batch_label,
                                                                                                batch_clss_mask,
                                                                                                batch_clss,
                                                                                                batch_clss_mask,
                                                                                                batch_context_bow,
                                                                                                batch_pattern_bow,
                                                                                                batch_position_bow)     
        

        context_disc_loss=torch.mean(context_disc_loss)*self.config.context_disc_loss_weight*(self.config.batch_size/self.config.true_batch_size)
        pattern_disc_loss=torch.mean(pattern_disc_loss)*self.config.pattern_disc_loss_weight*(self.config.batch_size/self.config.true_batch_size)
        position_disc_loss=torch.mean(position_disc_loss)*self.config.position_disc_loss_weight*(self.config.batch_size/self.config.true_batch_size)



        context_disc_loss.backward()
        pattern_disc_loss.backward()
        position_disc_loss.backward()

        
        #print('total_loss',str(total_loss.item())[:4],'  ','context_disc_loss',str(context_disc_loss.item())[:4],'  ','pattern_disc_loss',str(pattern_disc_loss.item())[:4],'  ','position_disc_loss',str(position_disc_loss.item())[:4])
        
        
        return context_disc_loss.item()+pattern_disc_loss.item()+position_disc_loss.item(), 1
        
    def train_iter(self):
        loss_list=[]
        recall_list=[]

        count=0
        self.generator.train()
        for i in range(self.config.max_epoch*self.config.train_set_len):
            count=count+1
            time_start=time.time()
            
            success=0
            batch_data=[]
            for j in range(int(self.config.true_batch_size/self.config.batch_size)):     
                batch_source_id_context, batch_source_id_pattern, batch_source_id_mask,batch_source_id_mask_pattern, batch_clss,batch_clss_mask,batch_label,batch_weight,batch_source,batch_query, batch_context_bow, batch_pattern_bow, batch_position_bow = \
                self.dataloader.load_data()                
                data=[batch_source_id_context, batch_source_id_pattern, batch_source_id_mask,batch_source_id_mask_pattern, batch_clss,batch_clss_mask,batch_label,batch_weight,batch_source,batch_query, batch_context_bow, batch_pattern_bow, batch_position_bow]
                batch_data.append(data)

    
            self.context_disc_optimizer.zero_grad()
            self.pattern_disc_optimizer.zero_grad()
            self.position_disc_optimizer.zero_grad() 
            #old_w=copy.deepcopy(self.generator.context_map.MLP1.weight)

            for j in range(int(self.config.true_batch_size/self.config.batch_size)):                     
                #try:
                loss,tag = self.train_one_disc(batch_data[j])
                #except:
                    #print('train fail')
                    #loss,tag =0,0
                if tag == 1:
                    loss_list.append(loss)
                    success=success+1
                if tag == 0:
                    print('one mini batch fail')                            
                    continue

            if success == int(self.config.true_batch_size/self.config.batch_size):
                
                #print('training disc')
                #print(self.generator.context_map.MLP1.weight.grad.sum())

                self.context_disc_optimizer.step()
                self.context_disc_optimizer.zero_grad()

                self.pattern_disc_optimizer.step()
                self.pattern_disc_optimizer.zero_grad()
                
                self.position_disc_optimizer.step()
                self.position_disc_optimizer.zero_grad() 
                
            else:
                print('jump one batch')   
                continue
            
            
            self.main_optimizer.zero_grad()                  
            success=0
            for j in range(int(self.config.true_batch_size/self.config.batch_size)):                     
                #try:
                _,loss,recall,tag = self.train_one_batch(batch_data[j])
                #except:
                    #print('train fail')
                    #_,loss,recall,tag =0,0,0,0
                if tag == 1:
                    recall_list.append(recall)
                    success=success+1
                if tag == 0:
                    print('one mini batch fail')                            
                    continue
                
            if success == int(self.config.true_batch_size/self.config.batch_size):
                
                #print('training main')
                #print(self.generator.context_map.MLP1.weight.grad.sum())

                self.main_optimizer.step()
                self.main_optimizer.zero_grad()  

                if self.config.use_lr_decay == 1:
                    if count%self.config.lr_decay_step == 0:
                        self.scheduler.step()         
            else:
                print('jump one batch')   
                continue   
                
            time_end=time.time()                
            #new_w=copy.deepcopy(self.generator.context_map.MLP1.weight)    
            #print(old_w == new_w)            
            
            
            def loss_record(loss_list,window):
                recent_list=loss_list[max(0,len(loss_list)-window*int(self.config.true_batch_size/self.config.batch_size)):]
                return str(np.mean(recent_list))[:8]
            
            if count % self.config.checkfreq == 0:       
                record=str(count)+' iter '+str(self.dataloader.epoch) + \
                ' epoch l:'+loss_record(loss_list,500) + \
                ' recall:'+loss_record(recall_list,500) + \
                ' ext:'+self.generator_raw.extract_loss_avg + \
                ' ext_cn:'+self.generator_raw.extract_loss_context_avg + \
                ' ext_pa:'+self.generator_raw.extract_loss_pattern_avg + \
                ' cn_e:'+self.generator_raw.context_entropy_loss_avg + \
                ' pa_e:'+self.generator_raw.pattern_entropy_loss_avg + \
                ' ps_e:'+self.generator_raw.position_entropy_loss_avg + \
                ' cn_m:'+self.generator_raw.context_mul_loss_avg + \
                ' pa_m:'+self.generator_raw.pattern_mul_loss_avg + \
                ' ps_m:'+self.generator_raw.position_mul_loss_avg + \
                ' cn_d:'+self.generator_raw.context_disc_loss_avg + \
                ' pa_d:'+self.generator_raw.pattern_disc_loss_avg + \
                ' ps_d:'+self.generator_raw.position_disc_loss_avg 


                    
                record+=' -- use time:'+str(time_end-time_start)[:5]
                print(record)
                
                
            if count % self.config.savefreq == 0 and count > self.config.savefreq-100 and count > self.config.startfreq:     
                avg_loss=loss_record(loss_list,500)
                avg_recall=loss_record(recall_list,500)
                print('start val') 
                rouge1,rouge2,additional_text=self.do_val(1000)                
                print(rouge1,rouge2)
                
                self.save_model(avg_recall,recall_list,rouge1,rouge2,loss_text=additional_text) 
                self.generator_raw.train()
                           


    def do_val(self, val_num):

        self.raw_rouge=Rouge()
        self.generator_raw.eval()
        
        #val_config=copy.deepcopy(self.config)

        data_loader_val=data_loader_long.data_loader('test', self.config, self.tokenizer, 'abstractive', load_qmsum=0, load_arxiv=1, add_prompt=self.config.add_prompt,return_bow=1)

        f1=[]
        f2=[]
        
        f1_context=[]
        f2_context=[]
        
        f1_pattern=[]
        f2_pattern=[]
        
        recall_all=[]
        
        c_pos_context=[]
        c_pos_pattern=[]
        c_pan_context=[]
        c_pan_pattern=[]
        c_con_context=[]
        c_con_pattern=[]
                
        for i in tqdm(range(int(val_num)), desc='testing'):      

            #try:
            data_point = \
            data_loader_val.load_data()
            #except:
                #print('load data fail during the evaluation')
                #continue
            
            
            all_score=[]
            all_source=[]
            all_label=[]
            all_score_context=[]
            all_score_pattern=[]
            
    
            for split in data_point:                

                batch_source_id_context, batch_source_id_pattern, batch_source_id_mask,batch_source_id_mask_pattern, batch_clss,batch_clss_mask,batch_label,batch_weight,batch_source,batch_query,batch_summary,batch_context_bow,batch_pattern_bow,batch_position_bow = \
                split 
                
                if self.config.add_prompt == 0:
                    with torch.no_grad():
                        if self.config.pattern_mask == 0:
                            batch_source_id_mask_pattern=batch_source_id_mask
                        predicts_score,predicts_score_context,predicts_score_pattern,accuracy = self.generator_raw.inference(batch_source_id_context,
                                                                                                                    batch_source_id_context,
                                                                                                                    batch_source_id_mask,
                                                                                                                    batch_source_id_mask,
                                                                                                                    batch_label,
                                                                                                                    batch_clss_mask,
                                                                                                                    batch_clss,
                                                                                                                    batch_clss_mask,
                                                                                                                    batch_context_bow,
                                                                                                                    batch_pattern_bow,
                                                                                                                    batch_position_bow)   
                else:
                    with torch.no_grad():
                        if self.config.pattern_mask == 0:
                            batch_source_id_mask_pattern=batch_source_id_mask
                        predicts_score,predicts_score_context,predicts_score_pattern,accuracy = self.generator_raw.inference(batch_source_id_context,
                                                                                                                    batch_source_id_pattern,
                                                                                                                    batch_source_id_mask,
                                                                                                                    batch_source_id_mask_pattern,
                                                                                                                    batch_label,
                                                                                                                    batch_clss_mask,
                                                                                                                    batch_clss,
                                                                                                                    batch_clss_mask,
                                                                                                                    batch_context_bow,
                                                                                                                    batch_pattern_bow,
                                                                                                                    batch_position_bow)       
                if accuracy[0]!=-1:
                    c_pos_context.append(accuracy[0])
                if accuracy[1]!=-1:                        
                    c_pos_pattern.append(accuracy[1])
                if accuracy[2]!=-1:
                    c_pan_context.append(accuracy[2])
                if accuracy[3]!=-1:
                    c_pan_pattern.append(accuracy[3])
                if accuracy[4]!=-1:
                    c_con_context.append(accuracy[4])
                if accuracy[5]!=-1:
                    c_con_pattern.append(accuracy[5])                            
                                            
                all_score.append(predicts_score)
                all_label.append(batch_label)
                all_source+=batch_source[0]

                all_score_context.append(predicts_score_context)
                all_score_pattern.append(predicts_score_pattern)

    
            all_score=torch.cat(all_score,1)    
            all_score=torch.squeeze(all_score)
                             
            all_score_context=torch.cat(all_score_context,1)    
            all_score_context=torch.squeeze(all_score_context)
            all_score_pattern=torch.cat(all_score_pattern,1)    
            all_score_pattern=torch.squeeze(all_score_pattern)
                
            all_label=torch.cat(all_label,1)    
            all_label=torch.squeeze(all_label)
            
            
            try:
                select_number=int(torch.sum(all_label))
                gold_top=torch.topk(all_label, select_number).indices
                pred_top=torch.topk(all_score, self.config.ndoc).indices         
                gold_top=gold_top.tolist()
                pred_top=pred_top.tolist()
            except:
                print('sent number is too small')
                continue
            
            source=[]
            for idx in pred_top:
                source.append(all_source[idx])  
                
            pred=' . '.join(source)
            gold=data_point[0][10][0]
            
            scores = self.raw_rouge.get_scores(pred, gold)            
            f1.append(scores[0]['rouge-1']['f'])
            f2.append(scores[0]['rouge-2']['f'])    
                
            
                                   
            try:
                select_number=int(torch.sum(all_label))
                gold_top=torch.topk(all_label, select_number).indices
                pred_top=torch.topk(all_score_context, self.config.ndoc).indices         
                gold_top=gold_top.tolist()
                pred_top=pred_top.tolist()
            except:
                print('sent context number is too small')
                continue
            source=[]
            for idx in pred_top:
                source.append(all_source[idx])   
                
            pred=' . '.join(source)
            gold=data_point[0][10][0]
    
            scores = self.raw_rouge.get_scores(pred, gold)            
            f1_context.append(scores[0]['rouge-1']['f'])
            f2_context.append(scores[0]['rouge-2']['f'])                                    
        
        
            try:
                select_number=int(torch.sum(all_label))
                gold_top=torch.topk(all_label, select_number).indices
                pred_top=torch.topk(all_score_pattern, self.config.ndoc).indices         
                gold_top=gold_top.tolist()
                pred_top=pred_top.tolist()
            except:
                print('sent pattern number is too small')
                continue
            
            source=[]
            for idx in pred_top:
                source.append(all_source[idx])   
                
            pred=' . '.join(source)
            gold=data_point[0][10][0]
    
            scores = self.raw_rouge.get_scores(pred, gold)            
            f1_pattern.append(scores[0]['rouge-1']['f'])
            f2_pattern.append(scores[0]['rouge-2']['f'])                                   
                                                                        
            if data_loader_val.epoch == 10:
                break 
            
            
            
        additional_text='_self_'+str(np.mean(f1))[:5]+'_con'+str(np.mean(f1_context))[:5]+'_pat'+str(np.mean(f1_pattern))[:5]
        
        print("effective label number", len(c_pos_context))
        additional_text2='_pos2con'+str(np.mean(c_pos_context))[:4]+'_pos2pan'+str(np.mean(c_pos_pattern))[:4]+'_pan2con'+str(np.mean(c_pan_context))[:4]+'_pan2pan'+str(np.mean(c_pan_pattern))[:4]+\
            '_con2con'+str(np.mean(c_con_context))[:4]+'_con2pan'+str(np.mean(c_con_pattern))[:4]

        return np.mean(f1),np.mean(f2),additional_text+additional_text2


def argLoader():

    parser = argparse.ArgumentParser()
    
    
    #device
    
    parser.add_argument('--device', type=int, default=0)    
    
    parser.add_argument('--multi_gpu', type=int, default=0)        

    parser.add_argument('--multi_device', type=str, default='0,1,2,3')       
    # Do What
    
    parser.add_argument('--do_train', action='store_true', help="Whether to run training")

    parser.add_argument('--do_test', action='store_true', help="Whether to run test")
    
    parser.add_argument('--seed', type=int, default=10)

    parser.add_argument('--only_pattern', type=int, default=0)     
    
    parser.add_argument('--add_prompt', type=int, default=0)  
    
    parser.add_argument('--ndoc', type=int, default=6) 
    
    parser.add_argument('--main_detach', type=int, default=0)    
    
    parser.add_argument('--aux_detach', type=int, default=1)  
    
    parser.add_argument('--pattern_mask', type=int, default=0) 
    

    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')    
    
    parser.add_argument('--pretrained_tokenizer', type=str, default='bert-base-uncased') 
    
    parser.add_argument('--bos_token_id', type=int, default=101) 
    
    parser.add_argument('--pad_token_id', type=int, default=0) 
    
    parser.add_argument('--eos_token_id', type=int, default=102)
    #Preprocess Setting
    parser.add_argument('--max_summary', type=int, default=80)

    parser.add_argument('--max_article', type=int, default=500)    
    
    #Model Setting
    parser.add_argument('--hidden_dim', type=int, default=768)
    
    parser.add_argument('--pattern_binary', type=int, default=1)

    parser.add_argument('--bow_dim_pattern', type=int, default=1)
    
    parser.add_argument('--additional_dim', type=int, default=1)

    parser.add_argument('--bow_dim', type=int, default=30522)
    
    parser.add_argument('--bow_dim_pos', type=int, default=100)   
    
    parser.add_argument('--vocab_size', type=int, default=30522)      

    parser.add_argument('--lr', type=float, default=2e-5)     
    
    parser.add_argument('--context_adversary_loss_weight', type=float, default=1)      

    parser.add_argument('--pattern_adversary_loss_weight', type=float, default=1)  
    
    parser.add_argument('--position_adversary_loss_weight', type=float, default=1)  
    
    parser.add_argument('--context_disc_loss_weight', type=float, default=1)      

    parser.add_argument('--pattern_disc_loss_weight', type=float, default=1)  
    
    parser.add_argument('--position_disc_loss_weight', type=float, default=1)  
    
    parser.add_argument('--context_multitask_loss_weight', type=float, default=1)  
    
    parser.add_argument('--pattern_multitask_loss_weight', type=float, default=1)  
    
    parser.add_argument('--position_multitask_loss_weight', type=float, default=1)  
    
    parser.add_argument('--extract_detach_weight', type=float, default=1) 
    
    parser.add_argument('--extract_main_weight', type=float, default=1) 
    
    parser.add_argument('--label_pos_weight', type=float, default=1)
    
    parser.add_argument('--eps', type=float, default=1e-10) 
    
    parser.add_argument('--dropout', type=float, default=0.5) 
        
    parser.add_argument('--batch_size', type=int, default=2)  

    parser.add_argument('--true_batch_size', type=int, default=16)  

    parser.add_argument('--buffer_size', type=int, default=256)      
    
    parser.add_argument('--scale_embedding', type=int, default=0)  
    
    #lr setting

    parser.add_argument('--use_lr_decay', type=int, default=0)  
    
    parser.add_argument('--lr_decay_step', type=int, default=10000)  
    
    parser.add_argument('--lr_decay', type=float, default=1)  

    # Testing setting
    parser.add_argument('--beam_size', type=int, default=4)
    
    parser.add_argument('--max_dec_steps', type=int, default=40)
    
    parser.add_argument('--min_dec_steps', type=int, default=5)
    
    parser.add_argument('--test_model', type=str, default='')   
    
    parser.add_argument('--load_model', type=str, default='')  
    
    parser.add_argument('--save_path', type=str, default='')  #arx_112_bidim_nodet_pos1_3disc_1mi_3label_1extract_
    
    parser.add_argument('--mid_start', type=int, default=0)
   
    # Checkpoint Setting
    parser.add_argument('--max_epoch', type=int, default=50)
    
    parser.add_argument('--train_set_len', type=int, default=4000)
    
    parser.add_argument('--savefreq', type=int, default=2000)

    parser.add_argument('--checkfreq', type=int, default=1)    

    parser.add_argument('--startfreq', type=int, default=1)        
    
    args = parser.parse_args()
    
    return args





def main():
    args = argLoader()
    
    if args.multi_gpu==1:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.multi_device
    else:
        torch.cuda.set_device(args.device)

    print('CUDA', torch.cuda.current_device())
    
    if args.do_train == True:
        x=Train(args)
        x.train_iter()
    if args.do_test == True:    
        x = Test(args)
        x.test()


main()
        