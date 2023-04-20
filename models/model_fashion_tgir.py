
from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,in_size, hidden_size, out_size):
        super().__init__()
        self.hidden_layer = nn.Linear(in_size, hidden_size, bias=True)
        self.drop_out = nn.Dropout(p=0.2)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size,eps=1e-12)
        self.out_layer = nn.Linear(hidden_size, out_size, bias=True)
        
    def forward(self, infeature):
        feature = self.hidden_layer(infeature)
        feature = self.drop_out(feature)
        feature = self.layer_norm(feature)
        feature = self.out_layer(feature)
        
        return feature


class FashionSAP(nn.Module):
    def __init__(self,                 
                 config = None,
                 args = None   
                 ):
        super().__init__()
        
        # self.tokenizer = tokenizer 
        self.config = config
        self.args = args
        embed_dim = config['embed_dim']        
        vision_width = config['vision_width']  
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        self.bert_config = BertConfig.from_json_file(config['bert_config'])   
        self.text_encoder = BertModel(config=self.bert_config, add_pooling_layer=False)      
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.combine_vision_proj = nn.Linear(embed_dim, embed_dim)
        self.combine_text_proj = nn.Linear(embed_dim, embed_dim)


        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        
        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)          
        self.text_encoder_m = BertModel(config=self.bert_config, add_pooling_layer=False)           
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.combine_vision_proj_m = nn.Linear(embed_dim, embed_dim)
        self.combine_text_proj_m = nn.Linear(embed_dim, embed_dim)

        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                            [self.combine_text_proj, self.combine_text_proj_m],
                            [self.combine_vision_proj, self.combine_vision_proj_m]
                           ]
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("fusion_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("img_queue_ptr", torch.zeros(1, dtype=torch.long))  
        self.register_buffer("fusion_queue_ptr", torch.zeros(1, dtype=torch.long))  


        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.fusion_queue = F.normalize(self.fusion_queue, dim=0)
        

    def forward(self, text_input_ids, text_attention_mask, ref_image, tar_image, alpha, ref_id=None, tar_id=None, cap_index=None):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        
        ref_image_embeds = self.visual_encoder(ref_image) 
        ref_image_atts = torch.ones(ref_image_embeds.size()[:-1],dtype=torch.long).to(ref_image.device)
        tar_image_embeds = self.visual_encoder(tar_image)           
        tar_image_feat = F.normalize(self.combine_vision_proj(self.vision_proj(tar_image_embeds[:,0,:])), dim=-1)            
        text_output = self.text_encoder(text_input_ids, attention_mask = text_attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        
        fusion_out = self.text_encoder(
            encoder_embeds =text_embeds,
            attention_mask = text_attention_mask,
            encoder_hidden_states = ref_image_embeds,
            encoder_attention_mask = ref_image_atts,      
            return_dict = True,
            mode = 'fusion',
        )
        fusion_feat = F.normalize(self.combine_text_proj(self.text_proj(fusion_out.last_hidden_state[:,0,:])), dim=-1)

        ####cal cantrastive loss
        with torch.no_grad():
            self._momentum_update()
            ref_image_embeds_m = self.visual_encoder_m(ref_image) 
            ref_image_feat_m = F.normalize(self.combine_vision_proj_m(self.vision_proj_m(ref_image_embeds_m[:,0,:])), dim=-1)
            tar_image_embeds_m = self.visual_encoder_m(tar_image)
            tar_image_feat_m = F.normalize(self.combine_vision_proj_m(self.vision_proj_m(tar_image_embeds_m[:,0,:])), dim=-1)
            tar_image_feat_all = torch.cat([tar_image_feat_m.t(), ref_image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                                         
   
            text_output_m = self.text_encoder_m(text_input_ids, attention_mask = text_attention_mask,             
                                                return_dict = True, mode = 'text')    
            text_embeds_m = text_output_m.last_hidden_state
            fusion_output_m = self.text_encoder_m(
                encoder_embeds =text_embeds_m,
                attention_mask = text_attention_mask,
                encoder_hidden_states = ref_image_embeds_m,
                encoder_attention_mask = ref_image_atts,      
                return_dict = True,
                mode = 'fusion',
            )
            fusion_feat_m = F.normalize(self.combine_text_proj(self.text_proj_m(fusion_output_m.last_hidden_state[:,0,:])), dim=-1)
            fusion_feat_all = torch.cat([fusion_feat_m.t(),self.fusion_queue.clone().detach()],dim=1)
               
            sim_i2f_m = tar_image_feat_m @ fusion_feat_all / self.temp 
            sim_f2i_m = fusion_feat_m @ tar_image_feat_all / self.temp   
            
            sim_targets_f2i = torch.zeros(sim_f2i_m.size()).to(tar_image.device)
            sim_targets_f2i.fill_diagonal_(1)
            sim_targets_i2f = torch.zeros(sim_i2f_m.size()).to(tar_image.device)
            sim_targets_i2f.fill_diagonal_(1)
            

            sim_i2f_targets = alpha * F.softmax(sim_i2f_m, dim=1) + (1 - alpha) * sim_targets_i2f
            sim_f2i_targets = alpha * F.softmax(sim_f2i_m, dim=1) + (1 - alpha) * sim_targets_f2i


        sim_i2f = tar_image_feat @ fusion_feat_all / self.temp 
        sim_f2i = fusion_feat @ tar_image_feat_all / self.temp
 
        loss_i2f = -torch.sum(F.log_softmax(sim_i2f, dim=1)*sim_i2f_targets,dim=1).mean()
        loss_f2i = -torch.sum(F.log_softmax(sim_f2i, dim=1)*sim_f2i_targets,dim=1).mean() 

        loss_ita = (loss_i2f+loss_f2i)/2
        loss = loss_ita
        self._dequeue_and_enqueue_fusion(torch.cat([tar_image_feat_m, ref_image_feat_m]), fusion_feat_m)

        return loss 
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)          
                
    @torch.no_grad()
    def _dequeue_and_enqueue_fusion(self, image_feat, fusion_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        fusion_feats = concat_all_gather(fusion_feat)
        # idxs = concat_all_gather(idx)

        img_batch_size = image_feats.shape[0]
        fusion_batch_size = int(img_batch_size/2)
        # print(batch_size)

        img_ptr = int(self.img_queue_ptr)
        fusion_ptr = int(self.fusion_queue_ptr)
        # print(ptr)
        assert self.queue_size % img_batch_size == 0  # for simplicity
        assert self.queue_size % int(img_batch_size/2) == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, img_ptr:img_ptr + img_batch_size] = image_feats.T
        self.fusion_queue[:, fusion_ptr:fusion_ptr + fusion_batch_size] = fusion_feats.T
        # self.idx_queue[:, img_ptr:img_ptr + img_batch_size] = idxs.T
        img_ptr = (img_ptr + img_batch_size) % self.queue_size  # move pointer
        fusion_ptr = (fusion_ptr + fusion_batch_size) % self.queue_size  # move pointer


        self.img_queue_ptr[0] = img_ptr  
        self.fusion_queue_ptr[0] = fusion_ptr  
        

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


