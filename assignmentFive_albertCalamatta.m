%clear workspace
clear all;

%load data
load fisheriris;

%randomize indices
rng(1);
indices = randsample(150,150);

%Training
spec = species(indices(1:60), :);
%Test
testSpec = species(indices(61:100), :);
%training
data = meas(indices(1:60), :);
%testing
testData = meas(indices(61:100), :);

%bayesian classifier
bayesClassifier = fitcnb(data,spec);

pred = bayesClassifier.predict(testData);

confMatBayesian = confusionmat(testSpec,pred);

%Nearest Neighbor 

nearestNeighbor = fitcknn(data,spec);

predict = nearestNeighbor.predict(testData);

confMatNearest = confusionmat(testSpec,predict);

%display results of confusion matrices
disp(confMatBayesian);
disp(confMatNearest);


