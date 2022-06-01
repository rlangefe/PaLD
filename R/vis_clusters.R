library(igraph)
library(RColorBrewer)
paldPlotC<-
function(C,lw=1,layout=NULL,lab=FALSE,gt=NULL) {
L<-layout  
C<-as.matrix(C)
C[,1:2] = apply(C[,1:2], 2, as.character)
C <- rbind(C,C[,c(2,1,3)])
n<-nrow(C)
if(is.null(rownames(C))){rownames(C)<-1:n}
if (length(lab)==1){
      if (lab==TRUE) {laby<-rownames(C)} else {laby<-rep("",dim(C)[1])}}
    else {laby<-lab}
    color<-c( brewer.pal(n = 8, name = "Dark2"), brewer.pal(n=8, name="Set2"))    
    Acut<-C
    #g<-graph.adjacency(Acut,weighted=TRUE,mode="undirected")
    g<-graph.edgelist(C[,1:2], directed=FALSE)
    E(g)$weight <- as.numeric(C[,3])
    g<-simplify(as.undirected(g))
    e<-as.numeric(get.edgelist(g)[, 1])
    u<-igraph::clusters(g)$membership
    u<-u[order(as.numeric(names(u)))]
    edge_colors<-color[u[e]]      
    edge_widths<-E(g)$weight
       
    lab<-laby[as.numeric(V(g)$name)]
    V(g)$lab<-lab
    if (is.null(L)) {g$layout<-layout_with_kk(g)} else {g$layout<-as.matrix(L)[as.numeric(V(g)$name),]}

    plot(g,   ylim=c(-1, 1),xlim=c(-1, 1),
           vertex.size=4, vertex.label.cex=1,
           vertex.color=color[igraph::clusters(g)$membership],
           #           vertex.label.color=color[igraph::clusters(g)$membership],  
           vertex.label.color="black",
           vertex.label=lab,
           vertex.label.dist = 1, edge.width=lw*100*(edge_widths),
           edge.color=edge_colors, asp=0,
           main="")
    if(!is.null(gt[1])){dev.new()
      plot(g,   ylim=c(-1, 1),xlim=c(-1, 1),
           vertex.size=4, vertex.label.cex=1.2,
           vertex.label.color=gt[as.numeric(V(g)$name)],
           vertex.color=gt[as.numeric(V(g)$name)],
           vertex.label=lab[as.numeric(V(g)$name)],
           vertex.label.dist = 1, edge.width=lw*100*(edge_widths),
           edge.color=edge_colors, asp=0,main="gt")}
}