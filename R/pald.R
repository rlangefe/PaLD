getcontmat<-function(D, b=0,h=.5,bet=1,cr=1:dim(D)[1]){
      D<-round(D,15)
      L<-NULL
      n=dim(D)[1]
      A3=matrix(0,n,n)
      for(x in 1:(n-1)){
        dx=D[x,]; dxt=D[,x]
        for(y in (x+1):n){
          dy=D[y,]; dyt=D[,y]
          Uxy=(which((dx<=bet*D[x,y]) | (dy<=bet*D[y,x]))) #the reaching set
          print("")
          print(Uxy)
          print("")
          print(dx[Uxy])
          print("")
          wx<-1*(dx[Uxy]<dy[Uxy])+h*((dx[Uxy]==dy[Uxy]))
          wy<-1*(dy[Uxy]<dx[Uxy])+h*((dx[Uxy]==dy[Uxy]))
          A3[x,Uxy]=A3[x,Uxy]+1/(length(Uxy))*wx
          A3[y,Uxy]=A3[y,Uxy]+1/(length(Uxy))*wy
        }
      }
      diag(A3)<-diag(A3)+b
      rownames(A3)=1:n
      colnames(A3)=1:n
      return(A3/(n-(b==0)))
    }
print("Reading files")
#D<-read.csv(file="generated.csv", header=FALSE)
D<-read.csv(file="10pts.csv", header=FALSE)
D<-apply(D,2,as.numeric)
print("Modifying rownames")
rownames(D)<-c(1:length(D))#D[,1]
#D<-D[,-1]
print("Running PaLD")
D<-as.matrix(dist(D))
C<-getcontmat(D)
bd<-mean(diag(C))/2
diag(C)<-0
C<-pmin(C,t(C))
T<-which((C>=bd)&(row(C)<col(C)),arr.ind=TRUE)
T<-cbind(T-1,C[T])
colnames(T)<-c("rows","cols", "vals")
print("Outputting")
write.csv(T,file="r-results.csv", row.names = FALSE)
