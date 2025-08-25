function predictionOutput = applyPolynomialMapping(inputData, coeffMatrix, degree)
    % Generate polynomial features for the input data
    X = generatePolynomialFeatures(inputData, degree);
    predictionOutput = X * coeffMatrix;
end