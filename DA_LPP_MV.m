function acc = DA_LPP(pre_feat_S,pre_lb_S,pre_feat_T,pre_lb_T,so_feat_S,so_lb_S, so_feat_T,so_lb_T, pseudo_feat_S, pseudo_lb_S, pseudo_feat_T, pseudo_lb_T)
    num_iter = 10;
    options.NeighborMode='KNN';
    options.WeightMode = 'HeatKernel';
    options.k = 30;
    options.t = 1;
    options.ReducedDim = 128;
    options.alpha = 1;
    num_class = length(unique(pre_lb_S));
    src_feat_index = size(pre_feat_S,1);
    tgt_feat_index = size(pre_feat_T,1);
    domainS_features = [pre_feat_S;so_feat_S;pseudo_feat_S];
    domainT_features = [pre_feat_T;so_feat_T;pseudo_feat_T];
    domainS_labels = [pre_lb_S so_lb_S pseudo_lb_S];
    domainT_labels = [pre_lb_T so_lb_T pseudo_lb_T];
    W_all = zeros(size(domainS_features,1)+size(domainT_features,1));
    W_s = constructW1(domainS_labels);
    W = W_all;
    W(1:size(W_s,1),1:size(W_s,2)) =  W_s;
    % looping
    p = 1;
    fprintf('d=%d\n',options.ReducedDim);
    for iter = 1:num_iter
        P = LPP([domainS_features;domainT_features],W,options);
        %P = LPP(domainS_features,W_s,options);
        domainS_proj = domainS_features*P;
        domainT_proj = domainT_features*P;
        proj_S_pre  = domainS_proj(1:src_feat_index,:);
        proj_S_source  = domainS_proj(src_feat_index+1:src_feat_index*2,:);
        proj_S_pseudo  = domainS_proj(src_feat_index*2+1:src_feat_index*3,:);
        proj_T_pre = domainT_proj(1:tgt_feat_index,:);
        proj_T_source = domainT_proj(tgt_feat_index+1:tgt_feat_index*2,:);
        proj_T_pseudo = domainT_proj(tgt_feat_index*2+1:tgt_feat_index*3,:);
        % proj_mean = mean([domainS_proj;domainT_proj]);
        % domainS_proj = domainS_proj - repmat(proj_mean,[size(domainS_proj,1) 1 ]);
        % domainT_proj = domainT_proj - repmat(proj_mean,[size(domainT_proj,1) 1 ]);
        % domainS_proj = L2Norm(domainS_proj);
        % domainT_proj = L2Norm(domainT_proj);
        % distances = EuDist2(domainT_proj,domainS_proj);
        % %% distance to class means
        % classMeans = zeros(num_class,options.ReducedDim);
        % for i = 1:num_class
        %     classMeans(i,:) = mean(domainS_proj(domainS_labels==i,:));
        % end
        % classMeans = L2Norm(classMeans);
        % distClassMeans = EuDist2(domainT_proj,classMeans);
        [dist1,dist4] = distmeans(proj_S_pre,proj_T_pre,pre_lb_S,num_class);
        [dist2,dist5] = distmeans(proj_S_source,proj_T_source,so_lb_S,num_class);
        [dist3,dist6] = distmeans(proj_S_pseudo,proj_T_pseudo,pseudo_lb_S, num_class);
        expMatrix = exp(-dist1);
        expMatrix2 = exp(-dist4);
        probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
        probMatrix2 = expMatrix2./repmat(sum(expMatrix2,2),[1 num_class]);
        probMatrix_1 = max(probMatrix,probMatrix2);
        expMatrix = exp(-dist2);
        expMatrix2 = exp(-dist5);
        probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
        probMatrix2 = expMatrix2./repmat(sum(expMatrix2,2),[1 num_class]);
        probMatrix_2 = max(probMatrix,probMatrix2);
        expMatrix = exp(-dist3);
        expMatrix2 = exp(-dist6);
        probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
        probMatrix2 = expMatrix2./repmat(sum(expMatrix2,2),[1 num_class]);
        probMatrix_3 = max(probMatrix,probMatrix2);
        probMatrix = 0.2*probMatrix_1+0.3*probMatrix_2+0.5*probMatrix_3;
        [prob,predLabels] = max(probMatrix');
        p=1-iter/num_iter;
        p = max(p,0);
        [sortedProb,index] = sort(prob);
        sortedPredLabels = predLabels(index);
        trustable = zeros(1,length(prob));
        for i = 1:num_class
            thisClassProb = sortedProb(sortedPredLabels==i);
            if length(thisClassProb)>0
                trustable = trustable+ (prob>thisClassProb(floor(length(thisClassProb)*p)+1)).*(predLabels==i);
            end
        end
        pseudoLabels = predLabels;
        pseudoLabels(~trustable) = -1;
        W = constructW1([domainS_labels,pseudoLabels,pseudoLabels,pseudoLabels]);
        %% calculate ACC
        acc = sum(predLabels==pre_lb_T)/length(pre_lb_T);
        % for i = 1:num_class
        %     acc_per_class(i) = sum((predLabels == domainT_labels).*(domainT_labels==i))/sum(domainT_labels==i);
        % end
%         fprintf('Iteration=%d, Acc:%0.3f\n', iter, acc);
        if sum(trustable)>=length(prob)
            break;
        end
    end
    fprintf('Iteration=%d, Acc:%0.3f\n', iter, acc);