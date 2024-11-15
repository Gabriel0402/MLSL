function [distClassMeans,distClusterMeans] = distmeans(domainS_proj,domainT_proj,domainS_labels,num_class)
    options.ReducedDim = 128;
    proj_mean = mean([domainS_proj;domainT_proj]);
    domainS_proj = domainS_proj - repmat(proj_mean,[size(domainS_proj,1) 1 ]);
    domainT_proj = domainT_proj - repmat(proj_mean,[size(domainT_proj,1) 1 ]);
    domainS_proj = L2Norm(domainS_proj);
    domainT_proj = L2Norm(domainT_proj);
    distances = EuDist2(domainT_proj,domainS_proj);
    %% distance to class means
    classMeans = zeros(num_class,options.ReducedDim);
    for i = 1:num_class
        classMeans(i,:) = mean(domainS_proj(domainS_labels==i,:));
    end
    classMeans = L2Norm(classMeans);
    distClassMeans = EuDist2(domainT_proj,classMeans);
    targetClusterMeans = vgg_kmeans(double(domainT_proj'), num_class, classMeans')';
    targetClusterMeans = L2Norm(targetClusterMeans);
    distClusterMeans = EuDist2(domainT_proj,targetClusterMeans);