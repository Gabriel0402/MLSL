function P = LPP(data,W,options)
data = double(data);
W = double(W);
D = diag(sum(W,2));
L = D - W;
Sl = data'*L*data;%data'=data的转置
Sd = data'*D*data;
Sl = (Sl+Sl')/2;
Sd = (Sd+Sd')/2;

Sl = Sl + options.alpha*eye(size(Sl,2));%eye是单位矩阵
opts.disp = 0;
[P,Diag] = eigs(double(Sd),double(Sl),options.ReducedDim,'la',options);%计算特征向量和特征值
for i = 1:size(P,2)
    if (P(1,i)<0)
        P(:,i) = P(:,i)*-1;
    end
end