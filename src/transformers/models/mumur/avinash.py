import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config
from modules.transformer import Transformer
from transformers import CLIPModel
import transformers
from transformers import AutoTokenizer, AutoModel
from mclip.multilingual_clip import pt_multilingual_clip

class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.multi_ling_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-B-32')
        config.pooling_type = 'transformer'
        self.pool_frames1 = Transformer(config)
        self.pool_frames2 = Transformer(config)

        # if(config.dataset_name == 'DIDEMO'): #TO REMOVE WITH NEW VERSION
        #     self.max_len = 64
        # elif(config.dataset_name == 'MSRVTT' or config.dataset_name == 'MSRVTT7k'):
        #     self.max_len = 32
        # elif(config.dataset_name == 'AVSD'):
        #     self.max_len = 25
        # else:
        #     self.max_len = 30


    def forward(self, en, video, multi_ling, return_all_frames=False, multilingual=False):
    # def forward(self,data, return_all_frames=False, multilingual=False): ##START FROM HERE
        # batch_size = data['video'].shape[0]
        # text_data = data['text']
        # video_data = data['video']
        # multiling_text_data = data['multi_ling']
        batch_size = video.shape[0]
        text_data = en
        video_data = video
        multiling_text_data = multi_ling
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        #print("CLIP json {} ".format(self.clip.text_model.config))
        #print(text_data['input_ids'].size())
        text_features = self.clip.get_text_features(**text_data)
        video_features = self.clip.get_image_features(video_data)
        multiling_out = self.multi_ling_model(multiling_text_data)

        #multiling_first = F.avg_pool1d(multiling_out[0], kernel_size=2, stride=2).squeeze()
        #multiling_second = F.avg_pool1d(multiling_out[1].reshape(batch_size, 1, -1), kernel_size=2, stride=2).squeeze()

        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        video_features_pooled_en = self.pool_frames1(text_features, video_features)
        video_features_pooled_multi = self.pool_frames1(multiling_out, video_features)
        #multiling_pool = self.pool_frames1(text_features, multiling_first)

        if return_all_frames:
            if(not multilingual):
                #print("Returning English features")
                return text_features, video_features, video_features_pooled_en
            #print("Returning Multilingual features")
            return multiling_out, video_features, video_features_pooled_multi

        return text_features, video_features_pooled_en, multiling_out, video_features_pooled_multi
    
    def encode_query(self, query):
        text_features = self.clip.get_text_features(**query)
        return text_features

    def compute_mktvr_similarity(self, vid_emb_unpooled, query_emb, with_cross_modal=True):
        video_features_pooled_en = []
        device = self.clip.device
        if with_cross_modal:
            for each_vid_emb_unpooled in vid_emb_unpooled:
                video_features_pooled_en.append(self.pool_frames1(query_emb.to(device), each_vid_emb_unpooled.unsqueeze(0).to(device)).squeeze(0).squeeze(0))
            video_features = torch.stack(video_features_pooled_en).cpu()
        else:
            video_features = vid_emb_unpooled.squeeze(1)
        video_features_normed =video_features/video_features.norm(dim=-1, keepdim=True)
        query_emb_normed = query_emb/query_emb.norm(dim=-1, keepdim=True)
        simi = video_features_normed@query_emb_normed.cpu().permute((1,0))
        return simi