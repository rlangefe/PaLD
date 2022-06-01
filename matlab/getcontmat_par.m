function A3 = getcontmat_par(D)
  bet=1;
  h=0.5;
  b=0;
  n=size(D,1);
  A3=zeros(n,n);
  parfor x=1:n
    dx=D(x,:); 
    for y=1:n
      if x ~= y
        dy=D(y,:);
        Uxy=((dx<=bet*D(x,y)) | (dy<=bet*D(y,x))); %the reaching set
        wx = 1*(dx(Uxy)<dy(Uxy))+h*((dx(Uxy)==dy(Uxy)));
        tmp = zeros(1,n);
        tmp(Uxy) = wx;
        A3(x,:)=A3(x,:) + 1/(sum(Uxy, 'all'))*tmp;
      end
    end
  end
  A3=A3+(eye(n)*b);
  A3 = A3/(n-(b==0));
  return;
end

