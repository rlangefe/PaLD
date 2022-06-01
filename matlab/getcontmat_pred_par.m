function A3 = getcontmat_pred_par(D,N)
  bet=1;
  h=0.5;
  b=0;
  n=size(D,1);
  A3=zeros(n-N,N);
  parfor x=(N+1):n
    dx=D(x,:);
    for y=1:N
      if x ~= y
        dy=D(y,:);
        Uxy = ((dx<=bet*D(x,y)) | (dy<=bet*D(y,x))); %the reaching set
        wx = 1*(dx(Uxy)<dy(Uxy))+h*((dx(Uxy)==dy(Uxy)));
        tmp = zeros(1,n);
        tmp(Uxy) = wx;
        A3(x-N,:) = A3(x-N,:) + 1/(sum(Uxy(:,1:N), 'all'))*tmp(:,1:N);
      end
    end
  end
  %I=eye(n-N);
  %A3=A3+(I((N+1):end, :)*b);
  A3 = A3/(N+1-(b==0));
  return;
end

