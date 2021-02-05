%%%%%%%%%%%%
% This function splits the data into two groups: one for training and one
% for testing. The split factor is given as input to the function.
%
% USE: [xTraining, yTraining, xTest, yTest] = splitTrainingData(data, y, p)
%
% Input:
%   data: data matrix with dimension M x N. M instances and N features
%   y: labels in form of an array of dimension M x 1
%   p: split factor 0...1
%
% Output:
%   xTraining: data matrix of dimension p*M x N. p*M instances and N
%     features
%   yTraining: vetor with labels of dimension p*M x 1
%   xTest: data matrix of dimension (1-p)*M x N. (1-p)*M instances and N
%     features
%   yTest: vetor with labels of dimension (1-p)*M x 1
%
% Marcelo Bacher, 2014
%%%%%%%%%%%%
function [xTraining, yTraining, xTest, yTest, i_train, i_test] = splitTrainingData(data, y, p)

labels = unique(y);
nLabels = length(labels);

xTraining = cell(nLabels,1);
yTraining = cell(nLabels,1);
xTest     = cell(nLabels,1);
yTest     = cell(nLabels,1);
i_train = [];
i_test = [];

for c=1:nLabels
    indexdata = find(y == labels(c));
    select = boolean(zeros(length(indexdata),1));
    mx = length(indexdata);
    select(randsample(mx, floor(p*mx))) = 1;
    xTraining{c} = data(indexdata(select),:);
    yTraining{c} = y(indexdata(select));
    xTest{c}     = data(indexdata(~select),:);
    yTest{c}     = y(indexdata(~select));
    i_train = [i_train; indexdata(select)];
    i_test = [i_test; indexdata(~select)];
end
% convert cells into matrices
xTraining = cell2mat(xTraining);
yTraining = cell2mat(yTraining);
xTest     = cell2mat(xTest);
yTest     = cell2mat(yTest);