clc;clear;

%%
load('Calibation_Unity2_CS2000_Hololens2_22_04_2025_WhiteWall_Quadrant2.mat');
black_level = 1;

primaries(1, :) = [Red];
primaries(2, :) = [Green];
primaries(3, :) = [Blue];
primaries(4, :) = [Gray];

white = [White]; 

x = (0:5:255)./255;
N = length(x);

if black_level
    allblacks_xyz = zeros(size(primaries, 1),3);
    for i=1:size(primaries, 1)
        allblacks_xyz(i, :) = primaries(i, 1).color.XYZ;
    end
    black_levelxyz = mean(allblacks_xyz);
end

%% Define xyY, xyz, and spectra
for i = 1:size(primaries, 1)
    for j = 1:size(primaries, 2)
        aux0 = [primaries(i, j).color.xyY];
        aux1 = [primaries(i, j).color.XYZ];
        Ys(j, i) = aux0(3);
        xs(j, i) = aux0(1);
        ys(j, i) = aux0(2);
        Xs(j, i) = aux1(1);
        Zs(j, i) = aux1(3);
        SPECTRA((i-1)*size(primaries, 2) + j,:) = primaries(i, j).radiance.value;
    end
end

%% Prepare the RGB and XYZ data for Polynomial Regression
RGB_train = zeros(N*4, 3);

% Fill in the RGB values for Red, Green, Blue primaries
RGB_train(1:N, 1) = x';  % Red channel for primaries
RGB_train(1:N, 2) = 0;   % Green channel for primaries
RGB_train(1:N, 3) = 0;   % Blue channel for primaries

RGB_train(N+1:2*N, 1) = 0;   % Red channel for primaries
RGB_train(N+1:2*N, 2) = x';  % Green channel for primaries
RGB_train(N+1:2*N, 3) = 0;   % Blue channel for primaries

RGB_train(2*N+1:3*N, 1) = 0;   % Red channel for primaries
RGB_train(2*N+1:3*N, 2) = 0;   % Green channel for primaries
RGB_train(2*N+1:3*N, 3) = x';  % Blue channel for primaries

% For grayscale, the R, G, and B values are identical
RGB_train(3*N+1:4*N, 1) = x';  % Grayscale, Red channel
RGB_train(3*N+1:4*N, 2) = x';  % Grayscale, Green channel
RGB_train(3*N+1:4*N, 3) = x';  % Grayscale, Blue channel

%% Prepare XYZ training data for each channel
XYZ_train = zeros(N*4, 3);
for i = 1:N
    % Assign XYZ values for Red, Green, Blue, Grayscale
    XYZ_train(i, :) = [Xs(i, 1), Ys(i, 1), Zs(i, 1)];     % Red
    XYZ_train(N + i, :) = [Xs(i, 2), Ys(i, 2), Zs(i, 2)]; % Green
    XYZ_train(2*N + i, :) = [Xs(i, 3), Ys(i, 3), Zs(i, 3)]; % Blue
    XYZ_train(3*N + i, :) = [Xs(i, 4), Ys(i, 4), Zs(i, 4)]; % Grayscale
end

XYZ_train_norm = XYZ_train - black_levelxyz;  % Optionally normalize by white XYZ
CIELab_train = xyz2lab(XYZ_train_norm, ...
    "WhitePoint", XYZ_train_norm(end, :));

% Set polynomial degree
degree = 2;

lambda = 1e-3;

%% The polynomial regression model
[coeffMatrix, predictedRGB] = polyRegression3D(CIELab_train, ...
    RGB_train, degree, lambda);

% Calculate training error
trainingError = mean(sqrt(sum((predictedRGB - RGB_train).^2, 2)));
fprintf('Mean training error: %f\n', trainingError);


RGB2lab_train = rgb2lab(RGB_train);
RGB2lab_predicted = rgb2lab(predictedRGB);
for i = 1:size(RGB_train, 1)
    deltaEtrain(i) = deltaE00(RGB2lab_train(i, :)', RGB2lab_predicted(i, :)');
end
stat_train = ...
    [mean(deltaEtrain) median(deltaEtrain) std(deltaEtrain) min(deltaEtrain) max(deltaEtrain)];
disp('Statistics on train data points:')
disp(stat_train)

% For new CIELab values, predict RGB values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:100
    CIELab_test(i, :) = ...
    xyz2lab(Validation_rand(i).color.XYZ'-black_levelxyz, ...
    'WhitePoint', XYZ_train(end, :));  % Get the corresponding Lab values
end
predictedRGB_test = ...
    applyPolynomialMapping(CIELab_test, coeffMatrix, degree);
RGB_test = PredefinedRGB ./255;
% Evaluate ΔE (color difference)
euclidist = zeros(size(CIELab_test, 1), 1);
RGB2lab_test = rgb2lab(RGB_test);
RGB2lab_testpred = rgb2lab(predictedRGB_test);
for i = 1:size(PredefinedRGB, 1)
    euclidist(i) = sqrt(sum((RGB_test(i,:) - predictedRGB_test(i,:)).^2));
    deltaEtest(i) = deltaE00(RGB2lab_test(i, :)', RGB2lab_testpred(i, :)');
end
stat_validation = [mean(deltaEtest) median(deltaEtest) std(deltaEtest) min(deltaEtest) max(deltaEtest)];
disp('Statistics on 100 validation points:')
disp(stat_validation)

% Visualize results
figure;
subplot(1,2,1);
scatter3(RGB_train(:,1), RGB_train(:,2), RGB_train(:,3), 20, 'b', 'filled');
hold on;
scatter3(predictedRGB(:,1), predictedRGB(:,2), predictedRGB(:,3), 20, 'r', 'filled');
title('Training Data: Original (blue) vs Predicted (red)');
xlabel('R'); ylabel('G'); zlabel('B');
legend('Original Lab', 'Predicted Lab');
grid on;

subplot(1,2,2);
scatter3(RGB_test(:,1), RGB_test(:,2), RGB_test(:,3), 20, 'b', 'filled');
hold on;
scatter3(predictedRGB_test(:,1), predictedRGB_test(:,2), predictedRGB_test(:,3), 20, 'r', 'filled');
title('Test Data: Original (blue) vs Predicted (red)');
xlabel('R'); ylabel('G'); zlabel('B');
legend('Original Lab', 'Predicted Lab');
grid on;








%% the validation data
%for i = 1:100
%     XYZmeas(i, :) = Validation_rand(i).color.XYZ';
%end
%XYZcorr = XYZmeas - black_levelxyz;

