function A3 = getcontmat_seq(D)
  bet=1;
  h=0.5;
  b=0;
  n=size(D,1);
  A3=zeros(n,n);
  for x=1:(n-1)
    dx=D(x,1:end); 
    %dxt=D(:,x);
    for y=(x+1):n
      dy=D(y,1:end);
      %dyt=D(:,y);
      Uxy=((dx<=bet*D(x,y)) | (dy<=bet*D(y,x))); %the reaching set
      wx = 1*(dx(Uxy)<dy(Uxy))+h*((dx(Uxy)==dy(Uxy)));
      wy = 1*(dy(Uxy)<dx(Uxy))+h*((dx(Uxy)==dy(Uxy)));
      A3(x,Uxy)=A3(x,Uxy)+1/(sum(Uxy, 'all'))*wx;
      A3(y,Uxy)=A3(y,Uxy)+1/(sum(Uxy, 'all'))*wy;
    end
  end
  A3=A3+(eye(n)*b);
  A3 = A3/(n-(b==0));
  return;
end
