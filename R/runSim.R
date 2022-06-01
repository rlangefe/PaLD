library(pald)

args = commandArgs(trailingOnly=TRUE)

loop_count = 0

t <- 5

for(dist1 in c('normal', 'exponential', 'chisquare')){
    for(mean1 in c(1)){
        for(std1 in seq(1, 80, 20)){
            for(dim in seq(1,22,10)){
                for(dist2 in c('normal', 'exponential', 'chisquare')){
                    for(mean2 in seq(1,200,75)){
                        for(std2 in seq(1, 80, 20)){
                            for(n2 in seq(10, 1001, 100)){
                                for(n1 in seq(10, 1001, 100)){
                                    for(num_times in 1:t){
                                        loop_count = loop_count+1
                                        if(loop_count != as.numeric(args[1])){
                                            next
                                        }
                                        data = matrix(NA, n1+n2, dim)

                                        for(d in 1:dim){
                                            if(dist1 == 'normal'){
                                                data[1:n1,d] = rnorm(n1, mean1, std1)
                                            } else if(dist1 == 'exponential'){
                                                data[1:n1,d] = rexp(n1, mean1)
                                                std1 = mean1
                                            } else if(dist1 == 'chisquare'){
                                                data[1:n1,d] = rchisq(n1, mean1)
                                                std1 = sqrt(2*mean1)
                                            }
                                        }

                                        for(d in 1:dim){
                                            if(dist2 == 'normal'){
                                                data[(n1+1):(n1+n2),d] = rnorm(n2, mean2, std2)
                                            } else if(dist2 == 'exponential'){
                                                data[(n1+1):(n1+n2),d] = rexp(n2, mean2)
                                                std2 = mean2
                                            } else if(dist2 == 'chisquare'){
                                                data[(n1+1):(n1+n2),d] = rchisq(n2, mean2)
                                                std2 = sqrt(2*mean2)
                                            }
                                        }

                                        D <- dist(data)
                                        start.time <- Sys.time()     
                                        cohesions <- cohesion_matrix(D)
                                        end.time <- Sys.time()
                                        time.taken <- difftime(end.time, start.time, units = "sec")

                                        results <- data.frame(indexVal = c(loop_count),
                                                            mean1 = c(mean1),
                                                            std1 = c(std1),
                                                            dist1 = c(dist1),
                                                            n1 = c(n1),
                                                            dim1 = c(dim),
                                                            mean2 = c(mean2),
                                                            std2 = c(std2),
                                                            dist2 = c(dist2),
                                                            n2 = c(n2),
                                                            dim2 = c(dim),
                                                            time_pald_R = c(time.taken),
                                                            bound_pald_R = c(mean(diag(cohesions))/2))

                                        file_name <- paste0("sim_output_files/full_run_results_R_", loop_count, ".csv")
                                        write.table(results, 
                                                    file=file_name, 
                                                    append=file.exists(file_name), 
                                                    row.names=FALSE,
                                                    col.names=FALSE)
                                    }
                                }
                            }
                        
                            if(dist2 %in% c('exponential', 'chisquare')){
                                break
                            }
                        }
                    }
                }
            }
        
            if(dist1 %in% c('exponential', 'chisquare')){
                break
            }
        }
    }
}