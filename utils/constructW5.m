function W = constructW1(label)
n=max(label(:));
D_labels = pdist(label);
D_labels = squareform(D_labels);
W=exp(-D_labels.^2);
W(1:n+1:end) = 0;