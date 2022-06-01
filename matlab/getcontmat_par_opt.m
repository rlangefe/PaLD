function A3 = getcontmat_par_opt(D)
  bet=1;
  h=0.5;
  b=0;
  n=size(D,1);
  A3=zeros(n,n);
  parfor x=1:n
    dx=ones(n,1)*D(x,:);
    Uxy=((dx<=bet*D(x,:)') | (D<=bet*D(:,x)));
    Uxy(x,:)=0;
    wx = 1*(dx(Uxy)<D(Uxy))+h*((dx(Uxy)==D(Uxy)));
    tmp=zeros(n,n);
    tmp(Uxy)=wx;
    Uxy=sum(Uxy,2)';
    ind = find(Uxy>0);
    Uxy(ind) = 1./Uxy(ind);
    A3(x,:)=(Uxy*tmp)';
  end
  A3=A3+(eye(n)*b);
  A3 = A3/(n-(b==0));
  return;
end