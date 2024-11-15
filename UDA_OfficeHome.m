% =====================================================================
% Code for journal paper:
% Zhu, Chenyang, Qian Wang, Yunxin Xie, and Shoukun Xu. 
% Multiview latent space learning with progressively fine-tuned 
% deep features for unsupervised domain adaptation. Information Sciences
% By Chenyang Zhu, zcy@cczu.edu.cn
% =====================================================================
%% Loading Data:
clear all
addpath('./utils/');
noft_dir = 'F:/multiview/ResNet50/OfficeHome/pretrain/';
soly_dir = 'F:/multiview/ResNet50/OfficeHome/so/';
pseudo_dir = 'F:/multiview/ResNet50/OfficeHome/ts/';
domains = {'Art','Clipart','Product','RealWorld'};

for source_domain_index = 1:length(domains)
    for target_domain_index = 1:length(domains)
        if target_domain_index == source_domain_index
            continue;
        end
        fprintf('Source domain: %s, Target domain: %s\n',domains{source_domain_index},domains{target_domain_index});
        load([noft_dir 'officehome-0.01-noFT-testTransformer-Art-' domains{source_domain_index} '-resnet50-noft.mat']);
        pre_feat_S = L2Norm(resnet50_features);
        pre_lb_S = labels+1;
        load([noft_dir 'officehome-0.01-noFT-testTransformer-Art-' domains{target_domain_index} '-resnet50-noft.mat']);
        pre_feat_T = L2Norm(resnet50_features);
        pre_lb_T = labels+1;
        load([soly_dir 'officehome-0.01-sourceOnlyFT-testTransforms-' domains{source_domain_index} '-' domains{source_domain_index} '-resnet50-noft.mat']);
        so_feat_S = L2Norm(features);
        so_lb_S = labels+1;
        load([soly_dir 'officehome-0.01-sourceOnlyFT-testTransforms-' domains{source_domain_index} '-' domains{target_domain_index} '-resnet50-noft.mat']);
        so_feat_T = L2Norm(features);
        so_lb_T = labels+1;
        load([pseudo_dir 'officehome-0.01-100.0-9-sourcePseudoTargetFT-testTransforms-source-' domains{source_domain_index} '-' domains{target_domain_index} '-resnet50-noft.mat']);
        pseudo_feat_S = L2Norm(resnet50_features);
        pseudo_lb_S = labels+1;
        load([pseudo_dir 'officehome-0.01-100.0-9-sourcePseudoTargetFT-testTransforms-' domains{source_domain_index} '-' domains{target_domain_index} '-resnet50-noft.mat']);
        pseudo_feat_T = L2Norm(resnet50_features);
        pseudo_lb_T = labels+1;
        opts.ReducedDim = 1024;
        num_class = length(unique(pre_lb_T));
        classifierType='1nn';
        acc_per_class = DA_LPP_MV(pre_feat_S,pre_lb_S,pre_feat_T,pre_lb_T,so_feat_S,so_lb_S, so_feat_T,so_lb_T, pseudo_feat_S, pseudo_lb_S, pseudo_feat_T, pseudo_lb_T);
    end
end
