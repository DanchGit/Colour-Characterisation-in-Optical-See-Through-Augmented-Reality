function X = generatePolynomialFeatures(inputData, degree)
    numSamples = size(inputData, 1);

    x = inputData(:, 1);
    y = inputData(:, 2);
    z = inputData(:, 3);
    
    numFeatures = nchoosek(degree + 3, 3);
    
    X = ones(numSamples, numFeatures);
    
    % First column is already filled with ones (constant term)
    featureIdx = 2;
    
    % Generate all polynomial terms up to degree
    for d = 1:degree
        for px = 0:d
            for py = 0:d-px
                pz = d - px - py;
                X(:, featureIdx) = (x.^px) .* (y.^py) .* (z.^pz);
                featureIdx = featureIdx + 1;
            end
        end
    end
end
