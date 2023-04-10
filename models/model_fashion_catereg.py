
from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
from models.resnet import resnet50


import torch
from torch import nn


class FashionSAP(nn.Module):
    def __init__(self,                 
                 config = None,
                 args = None   
                 ):
        super().__init__()

        self.config = config
        self.args = args

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    
        # self.visual_encoder = resnet50(pretrained=False)
        self.bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel(config=self.bert_config, add_pooling_layer=False)  
        self.catemap_layer = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size),
            nn.LayerNorm(self.bert_config.hidden_size,eps=self.bert_config.layer_norm_eps),
            nn.Linear(self.bert_config.hidden_size, args.class_num)

        ) 
    def forward(self, image, text_input_ids, text_attention_mask, outpos):       
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)          
        text_output = self.text_encoder(text_input_ids, attention_mask = text_attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state

        output_pos = self.text_encoder(encoder_embeds = text_embeds, 
                                        attention_mask = text_attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )  
        out_state = output_pos.last_hidden_state
        bs,ss,hs = out_state.size()
        out_cate_pos = outpos.unsqueeze(-1).expand(bs,1,hs)
        cate_state = torch.gather(input=out_state,dim=1,index=out_cate_pos).squeeze()
        
        cate_logit = self.catemap_layer(cate_state)

        return cate_logit


