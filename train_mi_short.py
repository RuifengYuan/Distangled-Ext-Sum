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
from model.model_mi import *
from model.baseline import *
import data_loader_long 
import data_loader
import os 
from tqdm import tqdm
import torch.nn.functional as F

# mmd loss

class Train(object):

    def __init__(self, config):
        self.config = config  
        
        seed = self.config.seed
        torch.manual_seed(seed)          
        torch.cuda.manual_seed(seed)       
        torch.cuda.manual_seed_all(seed)         
        

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_tokenizer)
        self.log = open('log.txt','w')

        self.dataloader=data_loader.data_loader('train', self.config, self.tokenizer, load_dataset='cnndm', add_prompt=self.config.add_prompt)
        
        
        #intialize model
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


        disc_params, main_params = self.generator_raw.get_params()  
        self.disc_optimizer = torch.optim.Adam(
            disc_params, lr=self.config.lr*self.config.MI_disc_weight)
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
        
        try:
            model_save_path = self.config.save_path+str(self.dataloader.count/self.config.true_batch_size*self.config.batch_size)+'_iter_'+str(self.dataloader.epoch) +'_ep__r_'+str(rouge1)+'_'+str(rouge2)+'__l_'+str(running_avg_loss)+loss_text
            torch.save(state, model_save_path)
        except:
            print('can not save the model!!!')
        
    def train_one_batch(self,data):
        
        #extraction


        batch_source_id_context, batch_source_id_pattern, batch_source_id_mask, batch_source_id_mask_pattern, batch_label,batch_label_mask,batch_clss,batch_clss_mask,batch_source,batch_ext_summary,batch_abs_summary,context_bow,pattern_bow,position_bow = \
        data
            
            
        if self.config.add_prompt == 0:
            if self.config.pattern_mask == 0:
                batch_source_id_mask_pattern=batch_source_id_mask
            ext_loss, label_loss, MI_loss, disc_loss,predicts_score,sents_vec_context,sents_vec_pattern = self.generator(batch_source_id_context,
                                                                                                batch_source_id_context,
                                                                                                batch_source_id_mask,
                                                                                                batch_source_id_mask,
                                                                                                batch_label,
                                                                                                batch_clss_mask,
                                                                                                batch_clss,
                                                                                                batch_clss_mask,
                                                                                                context_bow,
                                                                                                pattern_bow,
                                                                                                position_bow)     
        else:
            if self.config.pattern_mask == 0:
                batch_source_id_mask_pattern=batch_source_id_mask
            ext_loss, label_loss, MI_loss, disc_loss,predicts_score,sents_vec_context,sents_vec_pattern = self.generator(batch_source_id_context,
                                                                                                batch_source_id_pattern,
                                                                                                batch_source_id_mask,
                                                                                                batch_source_id_mask_pattern,
                                                                                                batch_label,
                                                                                                batch_clss_mask,
                                                                                                batch_clss,
                                                                                                batch_clss_mask,
                                                                                                context_bow,
                                                                                                pattern_bow,
                                                                                                position_bow)            
  
        ext_loss=torch.mean(ext_loss) 
        label_loss=torch.mean(label_loss) 
        
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
            
            

        
        final_loss=ext_loss+label_loss*self.config.label_loss_weight - MI_loss*self.config.MI_loss_weight
        
        #final_loss=final_loss/self.config.true_batch_size*self.config.batch_size
        #disc_loss=disc_loss/self.config.true_batch_size*self.config.batch_size
        
        #disc_loss.backward(retain_graph=True)
        final_loss.backward()
        
        if np.isnan(np.mean(recall_all)) == 0:
            return MI_loss.item(),ext_loss.item(),np.mean(recall_all), 1
        if np.isnan(np.mean(recall_all)) == 1:
            return MI_loss.item(),ext_loss.item(),0, 1    
        
    def train_one_disc(self,data):
        
        #extraction

        batch_source_id_context, batch_source_id_pattern, batch_source_id_mask, batch_source_id_mask_pattern, batch_label,batch_label_mask,batch_clss,batch_clss_mask,batch_source,batch_ext_summary,batch_abs_summary,context_bow,pattern_bow,position_bow = \
        data
            
            
        if self.config.add_prompt == 0:
            if self.config.pattern_mask == 0:
                batch_source_id_mask_pattern=batch_source_id_mask
            disc_loss = self.generator.train_disc(batch_source_id_context,
                                                batch_source_id_context,
                                                batch_source_id_mask,
                                                batch_source_id_mask,
                                                batch_label,
                                                batch_clss_mask,
                                                batch_clss,
                                                batch_clss_mask,
                                                context_bow,
                                                pattern_bow,
                                                position_bow)   
        else:
            if self.config.pattern_mask == 0:
                batch_source_id_mask_pattern=batch_source_id_mask
            disc_loss = self.generator.train_disc(batch_source_id_context,
                                                batch_source_id_pattern,
                                                batch_source_id_mask,
                                                batch_source_id_mask_pattern,
                                                batch_label,
                                                batch_clss_mask,
                                                batch_clss,
                                                batch_clss_mask,
                                                context_bow,
                                                pattern_bow,
                                                position_bow)            
  

        
        disc_loss.backward()

        return disc_loss.item(), 1
        
        
        
    def train_iter(self):
        loss_list=[]
        recall_list=[]

        count=0
        self.generator.train()
        for i in range(self.config.max_epoch*self.config.train_set_len):
            count=count+1
            time_start=time.time()

            
            batch_data=[]
            for j in range(int(self.config.true_batch_size/self.config.batch_size)):     
                batch_source_id_context, batch_source_id_pattern, batch_source_id_mask, batch_source_id_mask_pattern, batch_label,batch_label_mask,batch_clss,batch_clss_mask,batch_source,batch_ext_summary,batch_abs_summary,context_bow,pattern_bow,position_bow = \
                self.dataloader.load_data()              
                data=[batch_source_id_context, batch_source_id_pattern, batch_source_id_mask, batch_source_id_mask_pattern, batch_label,batch_label_mask,batch_clss,batch_clss_mask,batch_source,batch_ext_summary,batch_abs_summary,context_bow,pattern_bow,position_bow]
                batch_data.append(data)            
            
            
            success=0 
            self.disc_optimizer.zero_grad()
            for j in range(int(self.config.true_batch_size/self.config.batch_size)):                     
                #try:
                loss,tag = self.train_one_disc(batch_data[j])
                #except:
                    #print('train fail')
                    #loss,tag =0,0
                if tag == 1:
                    success=success+1
                if tag == 0:
                    print('one mini batch fail')                            
                    continue
                
            if success == int(self.config.true_batch_size/self.config.batch_size):

                self.disc_optimizer.step()
                self.disc_optimizer.zero_grad()
                
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
                    loss_list.append(loss)
                    recall_list.append(recall)
                    success=success+1
                if tag == 0:
                    print('one mini batch fail')                            
                    continue
                
            if success == int(self.config.true_batch_size/self.config.batch_size):

                
                self.main_optimizer.step()
                self.main_optimizer.zero_grad()     
     
            else:
                print('jump one batch')   
                continue
                
            time_end=time.time()                
            
            def loss_record(loss_list,window):
                recent_list=loss_list[max(0,len(loss_list)-window*int(self.config.true_batch_size/self.config.batch_size)):]
                s=str(np.mean(recent_list))
                sp=s.split('e')
                if len(sp)==1:
                    return s[:8]
                else:
                    return sp[0][:6]+'e'+sp[1]

            
            if count % self.config.checkfreq == 0:       
                record=str(count)+' iter '+str(self.dataloader.epoch) + \
                ' epoch l:'+loss_record(loss_list,500) + \
                ' recall:'+loss_record(recall_list,500) + \
                ' ext:'+self.generator_raw.extract_loss_avg + \
                ' ext_cn:'+self.generator_raw.extract_loss_context_avg + \
                ' ext_pa_pred:'+self.generator_raw.extract_loss_pattern_pred_avg + \
                ' ext_pa:'+self.generator_raw.extract_loss_pattern_avg + \
                ' mi_disc:'+self.generator_raw.MI_disc_loss_avg + \
                ' mi_loss:'+self.generator_raw.MI_loss_avg + \
                ' mi_loss_true:'+self.generator_raw.MI_loss_true_avg + \
                ' mi_loss_sample:'+self.generator_raw.MI_loss_false_avg


                    
                record+=' -- use time:'+str(time_end-time_start)[:5]
                print(record)
                
                
            if count % self.config.savefreq == 0 and count > self.config.savefreq-100 and count > self.config.startfreq:     
                avg_loss=loss_record(loss_list,500)
                avg_recall=loss_record(recall_list,500)
                print('start val')
                rouge1,rouge2,additional_text=self.do_val(50)                
                print(rouge1,rouge2)
                
                self.save_model(avg_recall,recall_list,rouge1,rouge2,loss_text=additional_text) 
                self.generator_raw.train()
                           

    def do_val(self, val_num):

        self.raw_rouge=Rouge()
        self.generator_raw.eval()
        
        #val_config=copy.deepcopy(self.config)

        data_loader_val=data_loader.data_loader('val', self.config, self.tokenizer, load_dataset='cnndm', add_prompt=self.config.add_prompt)

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
                
        for i in tqdm(range(int(val_num/self.config.batch_size)), desc='testing'):      
            #try:

            batch_source_id_context, batch_source_id_pattern, batch_source_id_mask,batch_source_id_mask_pattern, batch_label,batch_label_mask,batch_clss,batch_clss_mask,batch_source,batch_ext_summary,batch_abs_summary,context_bow,pattern_bow,position_bow = \
                data_loader_val.load_data()      
                    
            if self.config.add_prompt == 0:
                with torch.no_grad():
                    if self.config.pattern_mask == 0:
                        batch_source_id_mask_pattern=batch_source_id_mask
                    predicts_score,predicts_score_context,predicts_score_pattern,accuracy = self.generator_raw.inference(batch_source_id_context,
                                                                                                                batch_source_id_context,
                                                                                                                      batch_source_id_mask,
                                                                                                                      batch_source_id_mask,
                                                                                                                      batch_label,
                                                                                                                      batch_label_mask,
                                                                                                                      batch_clss,
                                                                                                                      batch_clss_mask,
                                                                                                                      context_bow,
                                                                                                                      pattern_bow,
                                                                                                                      position_bow)  
            else:
                with torch.no_grad():
                    if self.config.pattern_mask == 0:
                        batch_source_id_mask_pattern=batch_source_id_mask
                    predicts_score,predicts_score_context,predicts_score_pattern,accuracy = self.generator_raw.inference(batch_source_id_context,
                                                                                                                batch_source_id_pattern,
                                                                                                                      batch_source_id_mask,
                                                                                                                      batch_source_id_mask_pattern,
                                                                                                                      batch_label,
                                                                                                                      batch_label_mask,
                                                                                                                      batch_clss,
                                                                                                                      batch_clss_mask,
                                                                                                                      context_bow,
                                                                                                                      pattern_bow,
                                                                                                                      position_bow)                
                    
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

            
            for k in range(len(batch_label)):
                pred_top=torch.topk(predicts_score[k], min(3,len(batch_source[k]))).indices
                pred_top=pred_top.tolist()

                gold=batch_abs_summary[k]
                
                pred=''
                for idx in pred_top:
                    pred+=batch_source[k][idx]            

                scores = self.raw_rouge.get_scores(pred, gold)            
                f1.append(scores[0]['rouge-1']['f'])
                f2.append(scores[0]['rouge-2']['f'])    
                
            for k in range(len(batch_label)):
                pred_top=torch.topk(predicts_score_context[k], min(3,len(batch_source[k]))).indices
                pred_top=pred_top.tolist()

                gold=batch_abs_summary[k]
                
                pred=''
                for idx in pred_top:
                    pred+=batch_source[k][idx]            

                scores = self.raw_rouge.get_scores(pred, gold)            
                f1_context.append(scores[0]['rouge-1']['f'])
                f2_context.append(scores[0]['rouge-2']['f'])                        
                
            for k in range(len(batch_label)):
                pred_top=torch.topk(predicts_score_pattern[k], min(3,len(batch_source[k]))).indices
                pred_top=pred_top.tolist()

                gold=batch_abs_summary[k]
                
                pred=''
                for idx in pred_top:
                    pred+=batch_source[k][idx]            

                scores = self.raw_rouge.get_scores(pred, gold)            
                f1_pattern.append(scores[0]['rouge-1']['f'])
                f2_pattern.append(scores[0]['rouge-2']['f'])   
                    
            #except:
                #print('val fail')
                #continue
            
            additional_text='_self_'+str(np.mean(f1))[:5]+'_con'+str(np.mean(f1_context))[:5]+'_pat'+str(np.mean(f1_pattern))[:5]
            
            additional_text2='_pos2con'+str(np.mean(c_pos_context))[:4]+'_pos2pan'+str(np.mean(c_pos_pattern))[:4]+'_pan2con'+str(np.mean(c_pan_context))[:4]+'_pan2pan'+str(np.mean(c_pan_pattern))[:4]+\
                '_con2con'+str(np.mean(c_con_context))[:4]+'_con2pan'+str(np.mean(c_con_pattern))[:4]


            if data_loader_val.epoch == 10:
                break
                    
        if len(f1) != 0 and len(f2) != 0:
            print (np.mean(f1),np.mean(f2),np.mean(recall_all))
            return str(np.mean(f1))[:6],str(np.mean(f2))[:6],additional_text+additional_text2
        else:
            return 0,0,''


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
    
    parser.add_argument('--pattern_detach', type=int, default=0)    
    
    parser.add_argument('--pattern_mask', type=int, default=0) 

    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')    
    
    parser.add_argument('--pretrained_tokenizer', type=str, default='bert-base-uncased') 
    
    parser.add_argument('--bos_token_id', type=int, default=101) 
    
    parser.add_argument('--pad_token_id', type=int, default=0) 
    
    parser.add_argument('--eos_token_id', type=int, default=102)
    #Preprocess Setting
    parser.add_argument('--max_summary', type=int, default=80)

    parser.add_argument('--max_article', type=int, default=510)    
    
    #Model Setting
    parser.add_argument('--hidden_dim', type=int, default=768)
    
    parser.add_argument('--pattern_binary', type=int, default=0)

    parser.add_argument('--bow_dim_pattern', type=int, default=1000)
    
    parser.add_argument('--additional_dim', type=int, default=1000)

    parser.add_argument('--bow_dim', type=int, default=30522)
    
    parser.add_argument('--bow_dim_pos', type=int, default=100)  
    
    parser.add_argument('--vocab_size', type=int, default=30522)      

    parser.add_argument('--lr', type=float, default=2e-5)     
    
    parser.add_argument('--label_loss_weight', type=float, default=1)      

    parser.add_argument('--MI_disc_weight', type=float, default=1)  
    
    parser.add_argument('--MI_loss_weight', type=float, default=1)  

    parser.add_argument('--extract_main_weight', type=float, default=1)      
    
    parser.add_argument('--extract_detach_weight', type=float, default=1) 
    
    parser.add_argument('--label_pos_weight', type=float, default=1)
    
    
    parser.add_argument('--eps', type=float, default=1e-12) 
    
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
    
    parser.add_argument('--load_model_pattern', type=str, default='')  
    
    parser.add_argument('--save_path', type=str, default='')  
    
    parser.add_argument('--mid_start', type=int, default=0)
   
    # Checkpoint Setting
    parser.add_argument('--max_epoch', type=int, default=50)
    
    parser.add_argument('--train_set_len', type=int, default=40000)
    
    parser.add_argument('--savefreq', type=int, default=1000)

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
        