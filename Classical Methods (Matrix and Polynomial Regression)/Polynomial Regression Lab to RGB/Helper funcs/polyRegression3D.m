function [coeffMatrix, transformedData] = polyRegression3D(inputData, outputData, degree, lambda)
    % Number of samples
    numSamples = size(inputData, 1);

    if size(inputData, 2) ~= 3 || size(outputData, 2) ~= 3
        error('Input and output data must be Nx3 matrices');
    end
    
    if size(outputData, 1) ~= numSamples
        error('Input and output data must have the same number of samples');
    end
    
    % Generate polynomial basis functions
    X = generatePolynomialFeatures(inputData, degree);
    
     % Add Ridge Regularization term (lambda * I)
    lambdaI = lambda * eye(size(X, 2));
    lambdaI(1, 1) = 0; % Do not regularize the bias term 

    % Perform regression for each output dimension
    coeffMatrix = zeros(size(X, 2), 3);
    for dim = 1:3
        coeffMatrix(:, dim) = (X' * X + lambdaI) \ (X' * outputData(:, dim));
    end

    transformedData = X * coeffMatrix;
end




