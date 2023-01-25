% Copyright 2017 The MathWorks, Inc.

function I = readFunctionTrain(filename)
% Resize the flowers images to the size required by the network.
I = imread(filename);
I = imresize(I, [227 227]);
%I = cat(3, I, I, I); %combines 3 same images of grayscale to 3 channels

%combines 3 same images of grayscale to 3 channels



