mean1_arr = [];
std1_arr = [];
dist1_arr = [];
n1_arr = [];
dim1_arr = [];
mean2_arr = [];
std2_arr = [];
dist2_arr = [];
n2_arr = [];
dim2_arr = [];
matlab_time_arr = [];
matlab_bound_arr = [];
indexVal = [];
nprocs = [];


rng(42)
t = 5;
i=0;

dist1 = "normal";
mean1 = [1];
std1 = 20;
dim = 4;
dist2 = "normal";
mean2 = 80;
std2 = 40;
n2 = 1000;
n1 = 1000;
for num_times = 1:t
    D = zeros(n1+n2, dim);
    if strcmp(dist1, 'normal')
        D(1:n1,:) = normrnd(mean1, std1, n1, dim);
    elseif strcmp(dist1, 'exponential')
        D(1:n1,:) = exprnd(mean1, n1, dim);
        std1 = mean1;
    elseif strcmp(dist1, 'chisquare')
        D(1:n1,:) = chi2rnd(mean1, n1, dim);
        std1 = sqrt(2*mean1);
    end

    if strcmp(dist2, 'normal')
        D(n1+1:end,:) = normrnd(mean2, std2, n2, dim);
    elseif strcmp(dist2, 'exponential')
        D(n1+1:end,:) = exprnd(mean2, n2, dim);
        std2 = mean2;
    elseif strcmp(dist2, 'chisquare')
        D(n1+1:end,:) = chi2rnd(mean2, n2, dim);
        std2 = sqrt(2*mean2);
    end

    D=dist(D');
    tic;
    [C]=getcontmat_par(D);
    matlab_time = toc;

    bd=trace(C)/(size(D,1)*2);

    if ~isfile(strcat("scaling_runs/full_run_results_matlab_", num2str(curr_procs), ".csv"))
        nprocs = [nprocs curr_procs];
        mean1_arr = [mean1_arr mean1];
        std1_arr = [std1_arr std1];
        dist1_arr = [dist1_arr dist1];
        n1_arr = [n1_arr n1];
        dim1_arr = [dim1_arr dim];
        mean2_arr = [mean2_arr mean2];
        std2_arr = [std2_arr std2];
        dist2_arr = [dist2_arr dist2];
        n2_arr = [n2_arr n2];
        dim2_arr = [dim2_arr dim];
        matlab_time_arr = [matlab_time_arr matlab_time];
        matlab_bound_arr = [matlab_bound_arr bd];

        curr_table = table(nprocs', mean1_arr', std1_arr', dist1_arr', n1_arr', dim1_arr', mean2_arr', std2_arr', dist2_arr', n2_arr', dim2_arr', matlab_time_arr', matlab_bound_arr', 'VariableNames', {'nprocs' 'mean1' 'std1' 'dist1' 'n1' 'dim1' 'mean2' 'std2' 'dist2' 'n2' 'dim2' 'matlab_time' 'matlab_bound'});
        writetable(curr_table, strcat("scaling_runs/full_run_results_matlab_", num2str(curr_procs), ".csv"), 'WriteVariableNames',false,'WriteRowNames',true);

    else
        nprocs = [curr_procs];
        mean1_arr = [mean1];
        std1_arr = [std1];
        dist1_arr = [dist1];
        n1_arr = [n1];
        dim1_arr = [dim];
        mean2_arr = [mean2];
        std2_arr = [std2];
        dist2_arr = [dist2];
        n2_arr = [n2];
        dim2_arr = [dim];
        matlab_time_arr = [matlab_time];
        matlab_bound_arr = [bd];

        curr_table = table(nprocs', mean1_arr', std1_arr', dist1_arr', n1_arr', dim1_arr', mean2_arr', std2_arr', dist2_arr', n2_arr', dim2_arr', matlab_time_arr', matlab_bound_arr', 'VariableNames', {'nprocs' 'mean1' 'std1' 'dist1' 'n1' 'dim1' 'mean2' 'std2' 'dist2' 'n2' 'dim2' 'matlab_time' 'matlab_bound'});
        writetable(curr_table, strcat("scaling_runs/full_run_results_matlab_", num2str(curr_procs), ".csv"),'WriteMode','Append','WriteVariableNames',false,'WriteRowNames',true);
    end
end

