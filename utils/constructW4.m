function W = constructW4(feat,label)
D = pdist2(feat, feat).^2;
sigma = 0.1;  % Scaling parameter
W = exp(-D/(2*sigma^2));
W = W .* (label == label');
D_sqrt = diag(sqrt(sum(W, 2)));
W = D_sqrt \ W / D_sqrt;
