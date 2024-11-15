function Pc = eigvec(Sd,Sl,dim)
%EIGVEC Summary of this function goes here
%   Detailed explanation goes here
[Pc,Diag] = eigs(double(Sd),double(Sl),dim,'lm');
for i = 1:size(Pc,2)
    if (Pc(1,i)<0)
        Pc(:,i) = Pc(:,i)*-1;
    end
end

