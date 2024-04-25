import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .build import MODELS
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from extensions.pointops.functions import pointops
from extensions.pointnet2.pointnet2_modules import PointnetFPModule
import extensions.pointnet2.pointnet2_utils as pt_utils

from .transformer import TransformerEncoder, TransformerDecoder, Group, DummyGroup, Encoder
from .detr.build import build_encoder as build_encoder_3detr, build_preencoder as build_preencoder_3detr

from extensions.loss_functions import focal_loss
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate 
        self.cls_dim = config.cls_dim 
        self.num_heads = config.num_heads 

        self.group_size = config.group_size
        self.num_group = config.num_group

        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_arch = config.get('cls_head_arch', '1x')
        if self.cls_head_arch == '2x':
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )
        else:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()
        
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path, map_location="cpu")
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger = 'Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger = 'Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger = 'Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger = 'Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger = 'Transformer')


    def forward(self, pts, return_feature=False):
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        if return_feature: return x
        concat_f = torch.cat([x[:,0], x[:, 1:].max(1)[0]], dim = -1)
        ret = self.cls_head_finetune(concat_f)
        return ret


class DGCNN_Propagation(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        # print('using group version 2')
        self.k = k
        self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(1024, 384, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 384),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pt_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pt_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, coor, f, coor_q, f_q):
        """ coor, f : B 3 G ; B C G
            coor_q, f_q : B 3 N; B 3 N
        """
        # dgcnn upsample
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q


@MODELS.register_module()
class PointTransformer_seg(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate 
        self.nclasses = config.transformer_config.nclasses 
        self.num_heads = config.transformer_config.num_heads 

        self.group_size = config.transformer_config.group_size
        self.num_group = config.transformer_config.num_group

        self.downsample_targets = config.transformer_config.downsample_targets

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
            finetune=True
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.propogation_2 = PointnetFPModule([self.trans_dim+3, self.trans_dim*4, self.trans_dim])
        self.propogation_1 = PointnetFPModule([self.trans_dim+3, self.trans_dim*4, self.trans_dim])
        self.propogation_0 = PointnetFPModule([self.trans_dim+3+2, self.trans_dim*4, self.trans_dim])

        self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)

        self.seg_head = nn.Sequential(
            nn.Conv1d(self.trans_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.nclasses, 1)
        )

        self.loss_func = self.build_loss_function(gamma=3.0).to(self.device)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def build_loss_function(self, gamma):
        return focal_loss(gamma=gamma)
    
    def get_loss_acc(self, ret, gt, class_weights):
        # loss = F.cross_entropy(ret, gt.long(), weight=class_weights.mean(dim=0))
        loss = self.loss_func(ret, gt)
        pred = ret.argmax(dim=1)
        acc = (pred == gt).sum() / float(gt.size(0)*gt.size(1))
        return loss, acc

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path, map_location="cpu")
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger = 'Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger = 'Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger = 'Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger = 'Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger = 'Transformer')
    

    def load_model_from_ckpt_direct(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path, map_location="cpu")
        base_ckpt = {k.replace('module.', ""): v for k, v in ckpt['base_model'].items()}
        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger = 'Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger = 'Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger = 'Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger = 'Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger = 'Transformer')

    def forward(self, pts, cls_label):
        B, N, _ = pts.shape

        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # add pos embedding
        pos = self.pos_embed(center)
        # transformer
        _, inter_feats = self.blocks(group_input_tokens, pos)
        inter_feats = [self.norm(x).transpose(-1, -2).contiguous() for x in inter_feats]
        # one hot vector for describing upper and lower teeth
        cls_label_one_hot = cls_label.view(B, 2, 1).repeat(1, 1, N)

        center_original = pts
        center_trans = center.transpose(-1, -2).contiguous()
        f_l0 = torch.cat([cls_label_one_hot, center_original.transpose(-1, -2).contiguous()], 1)
        
        # downsample the orginial point cloud
        assert len(inter_feats) == len(self.downsample_targets), \
            "the length of the cardinality and the features should be the same"
        
        center_pts = []
        for i in range(len(inter_feats)):
            center_pts.append(pointops.fps(pts, self.downsample_targets[i]))
        center_pts_trans = [pt.transpose(-1, -2).contiguous() for pt in center_pts]
        
        f_l3 = inter_feats[2]
        f_l2 = self.propogation_2(center_pts[1], center, center_pts_trans[1], inter_feats[1])
        f_l1 = self.propogation_1(center_pts[0], center, center_pts_trans[0], inter_feats[0])

        f_l2 = self.dgcnn_pro_2(center_trans, f_l3, center_pts_trans[1], f_l2)
        f_l1 = self.dgcnn_pro_1(center_pts_trans[1], f_l2, center_pts_trans[0], f_l1)
        
        f_l0 = self.propogation_0(center_original, center_pts[0], f_l0, f_l1)

        logit = self.seg_head(f_l0)

        return logit 


class MaskPointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # encoder parameters
        self.num_group = config.transformer_config.num_group
        self.group_size = str(config.transformer_config.group_size).split('/')
        self.group_size = [int(group_size) for group_size in self.group_size]
        self.encoder_dims = config.transformer_config.encoder_dims
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        self.cls_dim = config.transformer_config.cls_dim

        # generative decoder paramters
        self.gen_queries = config.transformer_config.gen_queries
        self.gen_dec_depth = config.transformer_config.gen_dec_depth
        self.gen_dec_num_heads = config.transformer_config.num_heads
        gen_dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.gen_dec_depth)]

        # discriminative decoder parameters
        extract_layers = config.transformer_config.extract_layers
        self.extract_layers = [int(i) for i in str(extract_layers).split('/')]
        self.dis_dec_depth = config.transformer_config.dis_dec_depth
        self.dis_dec_num_heads = config.transformer_config.num_heads
        self.dis_dec_query_mode = config.transformer_config.dec_query_mode
        self.dis_dec_query_real_num = config.transformer_config.dec_query_real_num
        self.dis_dec_query_fake_num = config.transformer_config.dec_query_fake_num
        self.use_sigmoid = config.transformer_config.use_sigmoid
        self.ambiguous_threshold = config.transformer_config.ambiguous_threshold
        self.ambiguous_dynamic_threshold = config.transformer_config.ambiguous_dynamic_threshold
        print_log(f'[Transformer args] {config.transformer_config}', logger = 'MaskPoint')

        # define the encoder
        self.enc_arch = config.transformer_config.get('enc_arch', 'PointViT')
        if self.enc_arch == '3detr':
            self.encoder = build_preencoder_3detr(num_group=self.num_group, group_size=self.group_size, dim=self.encoder_dims)
        else:
            self.encoder = Encoder(encoder_channel = self.encoder_dims)

        # bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        # define the learnable tokens
        self.mask_token = [nn.Parameter(torch.randn(1, 1, self.trans_dim)).to(self.device) for i in range(len(self.extract_layers))]

        # pos embedding for each patch
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        # define the transformer blocks
        encoder_dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        if self.enc_arch == '3detr':
            self.blocks = build_encoder_3detr(
                ndim=self.trans_dim,
                nhead=self.num_heads,
                nlayers=self.depth
            )
        else:
            self.blocks = TransformerEncoder(
                embed_dim = self.trans_dim,
                depth = self.depth,
                drop_path_rate = encoder_dpr,
                num_heads = self.num_heads,
                extract_layers = self.extract_layers
            )

        self.cls_head = nn.Sequential(
            nn.Linear(self.trans_dim, self.cls_dim),
            nn.GELU(),
            nn.Linear(self.cls_dim, self.cls_dim)
        )

        # generative decoder
        self.gen_decoder = nn.ModuleList([
            TransformerDecoder(
                embed_dim=self.trans_dim,
                depth=self.gen_dec_depth,
                drop_path_rate=gen_dpr,
                num_heads=self.gen_dec_num_heads,
                mode='cross'
            )
            for i in range(len(self.extract_layers))
        ])
        self.gen_head = nn.ModuleList([
            nn.Conv1d(self.trans_dim, 3*group_size, 1)
            for group_size in reversed(self.group_size)
        ])

        # layer norm
        self.norm = nn.ModuleList([
            nn.LayerNorm(self.trans_dim)
            for i in range(len(self.extract_layers))
        ])
        
        # initialize the learnable tokens
        for mask_token in self.mask_token:
            trunc_normal_(mask_token, std=.02)

        self.apply(self._init_weights)
        self.access_count = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _generate_fake_query(self, target):
        B = target.shape[0]
        min_coords, max_coords = torch.min(target, dim=1, keepdim=True)[0], torch.max(target, dim=1, keepdim=True)[0]
        fake_target = torch.rand(B, self.dis_dec_query_fake_num, 3, dtype=target.dtype, device=target.device) * (max_coords - min_coords) + min_coords
        return fake_target

    def _generate_query_xyz(self, points, center, mode='center'):
        if mode == 'center':
            target = center
        elif mode == 'points':
            if self.dis_dec_query_real_num == -1:
                target = points
            else:
                target = pointops.fps(points, self.dis_dec_query_real_num)

        bs, npoints, _ = target.shape
        q, fake_q = target, self._generate_fake_query(target)

        nn_dist = pointops.knn(fake_q, points, 1)[1].squeeze()
        if self.ambiguous_dynamic_threshold > 0:
            assert self.ambiguous_threshold == -1
            if self.ambiguous_dynamic_threshold == self.dis_dec_query_real_num:
                thres_q = q
            else:
                thres_q = pointops.fps(points, self.ambiguous_dynamic_threshold)
            dist_thres = pointops.knn(thres_q, thres_q, 2)[1][..., -1].mean(-1, keepdims=True)
        else:
            assert self.ambiguous_dynamic_threshold == -1
            dist_thres = self.ambiguous_threshold
        queries = torch.cat((q, fake_q), dim=1)
        labels = torch.zeros(bs, queries.shape[1], dtype=torch.long, device=target.device)
        labels[:, :npoints] = 1
        labels[:, npoints:][nn_dist < dist_thres] = -1

        return queries, labels

    def preencoder(self, neighborhood):
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        return group_input_tokens
    
    def minOverlappedMaskedPatches(self, unmasked_centers, masked_centers):
        # unmasked_centers: [B, n_unmask, 3]  masked_centers: [B, n_mask, 3]
        mindist_mask2unmask = torch.min(((masked_centers[:, :, None] - unmasked_centers[:, None])**2).sum(dim=-1), dim=-1)[0]
        maskidxs2unmask = torch.argsort(mindist_mask2unmask, dim=-1, descending=True)
        masktopidxs = maskidxs2unmask[:, :self.gen_queries]
        return masktopidxs

    def forward(self, neighborhood_list, center, only_cls_tokens = False, noaug = False, points_orig=None):
        group_input_tokens = self.preencoder(neighborhood_list[1])
        B, G, _ = center.shape
        mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)

        if not noaug:
            if type(self.mask_ratio) is list:
                assert len(self.mask_ratio) == 2
                mask_ratio = random.uniform(*self.mask_ratio)
                n_mask = int(mask_ratio * G)
            elif self.mask_ratio > 0:
                n_mask = int(self.mask_ratio * G)
            perm = torch.randperm(G)[:n_mask]
            mask[:, perm] = True
        else:
            n_mask = 0
        n_unmask = G - n_mask

        unmasked_input_tokens = group_input_tokens[~mask].view(B, n_unmask, -1)
        unmasked_centers, masked_centers = center[~mask].view(B, n_unmask, -1), center[mask].view(B, n_mask, -1)
        unmasked_pos, masked_pos = self.pos_embed(unmasked_centers), self.pos_embed(masked_centers)

        # encoding
        x, pos = unmasked_input_tokens, unmasked_pos
        feats = self.blocks(x, pos)

        # get sample idxs
        # query_idxs = self.minOverlappedMaskedPatches(unmasked_centers, masked_centers) # B, nsample  B, n_mask, D
        # query_pos = torch.gather(masked_pos, dim=1, index=query_idxs[..., None].repeat_interleave(masked_pos.shape[-1], dim=-1))
        query_pos = masked_pos
        
        # hierarchical decoding
        rec_points_list = []
        for i, feat in enumerate(feats):
            feat = self.norm[i](feat)
            query_token = self.mask_token[i].expand(B, n_mask, -1)            
            x_rec = self.gen_decoder[i](query_token, query_pos, feat, pos)
            rec_points = self.gen_head[i](x_rec.transpose(1, 2)).transpose(1, 2).reshape(B*n_mask, -1, 3)
            rec_points_list.append(rec_points)
        
        gt_points_list = [neighborhood_list[i][mask].view(B, n_mask, -1) for i in reversed(range(len(self.group_size)))]
        gt_points_list = [gt_points.reshape(B*n_mask, -1, 3) for gt_points in gt_points_list]

        return rec_points_list, gt_points_list


@MODELS.register_module()
class MaskPoint(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[MaskPoint] build MaskPoint...', logger ='MaskPoint')
        self.config = config
        self.m = config.m
        self.T = config.T
        self.K = config.K
        
        self.transformer_q = MaskPointTransformer(config)
        
        self.chamfer_type = config.transformer_config.chamfer_type
        self.use_moco_loss = config.transformer_config.use_moco_loss
        self.moco_loss_weight = config.transformer_config.moco_loss_weight
        self.dis_loss_weight = config.transformer_config.dis_loss_weight
        self.gen_loss_weight = config.transformer_config.gen_loss_weight
        self.use_sigmoid = config.transformer_config.use_sigmoid
        self.use_focal_loss = config.transformer_config.use_focal_loss
        self.cur_weight_ratio = config.transformer_config.cur_weight_ratio

        if self.use_focal_loss:
            self.focal_loss_alpha = config.transformer_config.focal_loss_alpha
            self.focal_loss_gamma = config.transformer_config.focal_loss_gamma
        
        self.group_size = str(config.transformer_config.group_size).split('/')
        self.group_size = [int(group_size) for group_size in self.group_size]
        self.num_group = config.transformer_config.num_group

        print_log(f'[MaskPoint Group] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='MaskPoint')
        self.enc_arch = config.transformer_config.get('enc_arch', 'PointViT')
        self.group_divider = (DummyGroup if self.enc_arch == '3detr' else Group)\
            (num_group = self.num_group, group_size = self.group_size, cur_weight_ratio = self.cur_weight_ratio)

        # create the queue
        self.register_buffer("queue", torch.randn(self.transformer_q.cls_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_batch = nn.CrossEntropyLoss(reduction='none')

        # loss
        self.build_loss_func()

    def build_loss_func(self):
        if self.use_sigmoid:
            self.loss_bce_batch = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_ce = nn.CrossEntropyLoss(ignore_index=-1)
            self.loss_ce_batch = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        if self.chamfer_type == 'l1':
            self.gen_loss = ChamferDistanceL1()
        else:
            self.gen_loss = ChamferDistanceL2()

    def forward_eval(self, pts):
        with torch.no_grad():
            neighborhood, center = self.group_divider(pts)
            cls_feature = self.transformer_q(neighborhood, center, only_cls_tokens = True, noaug = True, points_orig = pts)
            return cls_feature

    def loss_focal_bce(self, pred, target):
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.focal_loss_alpha * target + (1 - self.focal_loss_alpha) * (1 - target)) * pt.pow(self.focal_loss_gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        return loss

    def loss_bce(self, preds, labels, reduction='mean'):
        loss_labels = labels.clone()
        loss_labels[labels == -1] = 0
        loss_labels_one_hot = F.one_hot(loss_labels, num_classes=2)
        preds = preds.transpose(1, 2).contiguous()

        if self.use_focal_loss:
            loss = self.loss_focal_bce(preds, loss_labels_one_hot)
        else:
            loss = self.loss_bce_batch(preds, loss_labels_one_hot.float())
        if reduction == 'mean':
            loss = loss[labels != -1].mean()
        return loss

    def forward(self, pts, curvatures, noaug = False, **kwargs):
        if noaug:
            return self.forward_eval(pts)
        else:
            neighborhood_list, center = self.group_divider(pts, curvatures)
            rec_points_list, gt_points_list = self.transformer_q(neighborhood_list, center, points_orig=pts)
            
            gen_loss_list = []
            for (rec_points, gt_points) in zip(rec_points_list, gt_points_list):
                gen_loss_list.append(self.gen_loss(rec_points, gt_points))
            
            return gen_loss_list
