library(igraph)

data<-read.csv("results.csv")
A<-cbind(data$rows + 1,data$cols + 1)
w<-data$vals

w[data$rows == data$cols] <- 0
g<-graph_from_edgelist(A, directed=FALSE)#, weighted = TRUE)
E(g)$weights<-w
g<-simplify(g)
cl<-clusters(g)$membership

pdf("graph.pdf") 

plot(g,vertex.color=cl,vertex.size=5,edge.color="grey90",vertex.label="")

dev.off()