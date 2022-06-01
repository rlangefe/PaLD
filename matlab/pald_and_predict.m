warning('off')
disp("Reading files");
D=readmatrix("../data/sim/data.csv", "Delimiter",",");
disp("Running PaLD");
D=dist(D');
%D=D(2:end, 2:end);
%[C]=getcontmat_par_opt(D);
[C]=getcontmat_par(D);
%[C]=getcontmat_seq(D);
bd=trace(C)/(size(D,1)*2);
C=C-(C.*eye(size(C,1)));
C=min(C,C');
[rows,cols] = find(C>=bd);
vals = zeros(length(rows),1);

parfor i=1:length(rows)
    vals(i) = C(rows(i), cols(i));
end

[r] = find(rows<cols);

bd
T=table(rows(r),cols(r),vals(r));
T.Properties.VariableNames(1:3) = {'rows','cols','vals'};
writetable(T, "matlab-results.csv");


disp("Running Prediction");
disp("Reading files");
D=readmatrix("../data/sim/data.csv", "Delimiter",",");
G=readmatrix("matlab-results.csv", "Delimiter",",");
P=readmatrix("../data/sim/test_data.csv", "Delimiter",",");
%D=[D(unique(reshape(G(:,1:2),size(G,1)*2,1)),:);P];
D=[D;P];
N=size(D,1)-size(P,1);
disp("Running PaLD");
D=dist(D');
%D=D(2:end, 2:end);
%[C]=getcontmat_par_opt(D,N);
[C]=getcontmat_pred_par(D,N);
%[C]=getcontmat_seq(D);
%C=min(C);
[rows,cols] = find(C>=bd);
vals = zeros(length(rows),1);

parfor i=1:length(rows)
    vals(i) = C(rows(i), cols(i));
end

[r] = find(rows<cols);

T=table(rows(r),cols(r),vals(r));
T.Properties.VariableNames(1:3) = {'rows','cols','vals'};
writetable(T, "predicted-cohesions.csv");

exit;