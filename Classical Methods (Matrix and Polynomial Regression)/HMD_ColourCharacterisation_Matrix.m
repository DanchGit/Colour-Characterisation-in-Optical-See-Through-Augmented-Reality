clc;close all;clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% HMD colour characterisation
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% To get RGB to XYZ values using the model below
load('ForPsychophysical_CC24_Q2_RGBfromPolynomial.mat');
uselater_RGB = RGB_reflectance;
clear RGB_reflectance;

%%
load('Calibation_Unity6_CS2000_Hololens2_14_04_2025_dark.mat');
%save_filename = 'Hololens2_06052025_model_b0.mat';
black_level = 1;
%%
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

%%
%Radiances

% Plot
Red(52).radiance.plot;
% Update font sizes
ylim([0 0.4])
xlabel('Wavelength (nm)', 'FontSize', 14);
ylabel('Radiance', 'FontSize', 14);
title('Radiance for Red', 'FontSize', 16);
set(gca, 'FontSize', 12);
lines = findobj(gca, 'Type', 'Line');
set(lines, 'Color', [1 0 0], 'LineWidth', 2); 


Green(52).radiance.plot;
% Update font sizes
ylim([0 0.4])
xlabel('Wavelength (nm)', 'FontSize', 14);
ylabel('Radiance', 'FontSize', 14);
title('Radiance for Green', 'FontSize', 16);
set(gca, 'FontSize', 12);
lines = findobj(gca, 'Type', 'Line');
set(lines, 'Color', [0 1 0], 'LineWidth', 2); 


Blue(52).radiance.plot;
% Update font sizes
ylim([0 0.4])
xlabel('Wavelength (nm)', 'FontSize', 14);
ylabel('Radiance', 'FontSize', 14);
title('Radiance for Blue', 'FontSize', 16);
set(gca, 'FontSize', 12);
lines = findobj(gca, 'Type', 'Line');
set(lines, 'Color', [0 0 1], 'LineWidth', 2); 


Gray(52).radiance.plot;
% Update font sizes
ylim([0 0.4])
xlabel('Wavelength (nm)', 'FontSize', 14);
ylabel('Radiance', 'FontSize', 14);
title('Radiance for White', 'FontSize', 16);
set(gca, 'FontSize', 12);
lines = findobj(gca, 'Type', 'Line');
set(lines, 'Color', [0.6 0.6 0.6], 'LineWidth', 2); 



%%
% Create figure
figure('Name', 'Combined Radiance Plot', 'NumberTitle', 'off');
hold on;

% Plot Red
plot(Red(52).radiance.wavelength, Red(52).radiance.value, ...
     'Color', [1 0 0], 'LineWidth', 2, 'LineStyle','-.');

% Plot Green
plot(Green(52).radiance.wavelength, Green(52).radiance.value, ...
     'Color', [0 1 0], 'LineWidth', 2, 'LineStyle','-.');

% Plot Blue
plot(Blue(52).radiance.wavelength, Blue(52).radiance.value, ...
     'Color', [0 0 1], 'LineWidth', 2, 'LineStyle','-.');

% Plot Gray (White)
plot(Gray(52).radiance.wavelength, Gray(52).radiance.value, ...
     'Color', [0.6 0.6 0.6], 'LineWidth', 2, 'LineStyle',':');

% Formatting
ylim([0 0.4]);
xlabel('Wavelength (nm)', 'FontSize', 14);
ylabel('Radiance', 'FontSize', 14);
title('Radiance for RGB and White', 'FontSize', 16);
set(gca, 'FontSize', 12);
legend({'Red', 'Green', 'Blue', 'White'});

drawnow;


%% Defin xyY, xyz, and spectra %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:size(primaries, 1)
    
    for j=1:size(primaries, 2)
        aux0 = [primaries(i, j).color.xyY];
        aux1 = [primaries(i, j).color.XYZ];
        Ys(j, i) = aux0(3) ;
        xs(j, i) = aux0(1) ;
        ys(j, i) = aux0(2) ;
        Xs(j, i) = aux1(1) ;
        Zs(j, i) = aux1(3) ;
        SPECTRA((i-1)*size(primaries, 2) + j,:) = ...
            primaries(i, j).radiance.value;
    end
    
end

%% Plot measurements %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot xy chromaticities of primaries
figure;
plotChromaticity();hold on
cols = {'r', 'g', 'b', 'w'};
for i=1:size(primaries, 1)
    if black_level
        xyz_aux = [Xs(:, i) Ys(:, i) Zs(:, i)] - black_levelxyz;
        xyY_aux = XYZToxyY(xyz_aux');

        plot(xyY_aux(1, :), xyY_aux(2, :), [cols{i}, 'o'], 'MarkerSize', 12, ...
        'MarkerEdgeColor', 'k', 'MarkerFaceColor', cols{i}, ...
        'LineWidth', .3);
    else
        plot(xs(:, i), ys(:, i), [cols{i}, 'o'], 'MarkerSize', 12, ...
        'MarkerEdgeColor', 'k', 'MarkerFaceColor', cols{i}, ...
        'LineWidth', .3);
    end
end
yticks([0 0.2 0.4 0.6 0.8])
xticks([0 0.2 0.4 0.6 0.8])

set(gca,  'FontSize', 20, 'fontname','Times New Roman',...
    'Color', [1. 1. 1.]);
grid on
set(gcf,'renderer','Painters');
%%
% Plot Y luminance and additivity using tiled layout
figure;
tiledlayout(1, 4, 'Padding', 'compact', 'TileSpacing', 'compact'); % Better layout

cols = {'r', 'g', 'b', 'k'};
channel_labels = {'Red', 'Green', 'Blue', 'Additivity (R+G+B)'};

for i = 1:4
    nexttile;
    hold on;

    % Plot individual channels or combined
    if i < 4
        if black_level
            y_vals = Ys(:, i) - black_levelxyz(2);
        else
            y_vals = Ys(:, i);
        end
        plot(x, y_vals, [cols{i}, '-o'], ...
            'MarkerSize', 6, ...
            'MarkerEdgeColor', 'k', 'MarkerFaceColor', cols{i}, ...
            'LineWidth', 0.8);
    else
        if black_level
            y_vals = sum(Ys(:, 1:3) - black_levelxyz(2), 2);
        else
            y_vals = sum(Ys(:, 1:3), 2);
        end
        plot(x, y_vals, [cols{i}, '--o'], ...
            'MarkerSize', 6, ...
            'MarkerEdgeColor', 'k', 'MarkerFaceColor', cols{i}, ...
            'LineWidth', 0.8);
    end

    % Axis config
    xticks(0:0.2:1);
    yticks(0:10:200);
    xlabel('Input Level');
    if i == 1
        ylabel('Luminance (Y)');
    end
    title(channel_labels{i});
    grid on;
    axis tight;
end

set(gcf, 'Renderer', 'Painters');

%%
set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultTextFontSize', 14);
figure('Position', [100, 100, 600, 400]); % Reasonable size
hold on;

cols = {'r', 'g', 'b', 'k'};
channel_labels = {'Red', 'Green', 'Blue', 'White'};

for i = 1:4
    % if i < 4
    if black_level
        y_vals = Ys(:, i) - black_levelxyz(2);
    else
        y_vals = Ys(:, i);
    end
    style = '-o';
    % else
    %     if black_level
    %         y_vals = sum(Ys(:, 1:3) - black_levelxyz(2), 2);
    %     else
    %         y_vals = sum(Ys(:, 1:3), 2);
    %     end
    %     style = '-o';
    %end
    if i == 4
        plot(x, y_vals, style, 'Color', [0.7, 0.7, 0.7], 'MarkerSize', 4, ...
            'MarkerEdgeColor', 'k', 'MarkerFaceColor', [0.7, 0.7, 0.7], ...
            'LineWidth', 1.2, 'DisplayName', 'White');
    else
        plot(x, y_vals, [cols{i}, style], ...
            'MarkerSize', 4, ...
            'MarkerEdgeColor', 'k', ...
            'MarkerFaceColor', cols{i}, ...
            'LineWidth', 1.2, ...
            'DisplayName', channel_labels{i});
    end
end

% Add the R+G+B combined line in gray and dotted
if black_level
    combined_y_vals = sum(Ys(:, 1:3) - black_levelxyz(2), 2);
else
    combined_y_vals = sum(Ys(:, 1:3), 2);
end
plot(x, combined_y_vals, 'o-', 'Color', [0.1, 0.1, 0.1], 'MarkerFaceColor', [0,0,0], 'MarkerSize', 4 ,'LineWidth', 0.5, 'DisplayName', 'R+G+B');

xlabel('Input Level (Normalized)', 'FontSize',14);
ylabel('Luminance (Y)', 'FontSize',14);
title('Luminance vs Input Level','FontSize',16);
xticks(0:0.2:1.0);
legend('Location', 'northwest','FontSize',14);
grid on;
axis tight;
set(gcf, 'Renderer', 'Painters');


%%

% Plot spectra of all primary ranges
figure;
for i=1:size(primaries, 1)
    subplot(1, 4, i)
    
    for j=1:size(primaries, 2)
        
        plot(380:780,primaries(i, j).radiance.value, cols{i}); hold on
        
    end
    xlabel('Wavelength (nm)', 'Interpreter','latex');
    ylabel('Power', 'Interpreter','latex');
    set(gca,  'FontSize', 24, 'fontname','TeXGyreTermes');
    grid on
    set(gcf,'renderer','Painters');
end

% Plot normalize spectra in a single plot (only maximum intensities)
figure
for i=1:size(primaries, 1)
    plot(380:780,primaries(i, end).radiance.value ./ ...
    max(primaries(i, end).radiance.value), [cols{i}, '-.'],...
    'LineWidth',2); hold on
end
set(gca,  'FontSize', 15, 'fontname','Times New Roman');
set(gcf,'renderer','Painters');
legend('Red primary','Green primary','Blue primary', ...
    'Interpreter','latex','Location','northeast','FontSize',12);
legend('boxoff')
xlabel('Wavelength (nm)', 'Interpreter','latex');
ylabel('Normalized power', 'Interpreter','latex');
axis([380 780 0 1])
xticks([400 500 600 700])

% Plot spectra in a single plot  (only maximum intensities)
figure
for i=1:size(primaries, 1)
    plot(380:780,primaries(i, end).radiance.value, [cols{i}, '-.'],...
    'LineWidth',2); hold on
end
set(gca,  'FontSize', 15, 'fontname','Times New Roman');
set(gcf,'renderer','Painters');
legend('Red primary','Green primary','Blue primary', ...
    'Interpreter','latex','Location','northeast','FontSize',12);
legend('boxoff')
xlabel('Wavelength (nm)', 'Interpreter','latex');
ylabel('Power', 'Interpreter','latex');
axis([380 780 0 Inf])
xticks([400 500 600 700])


%% Define the model: matrix + LUTs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Matrix
for ch=1:3
    if black_level
        monXYZ(ch,:) = ...
        [Xs(end, ch) Ys(end, ch) Zs(end, ch)] - black_levelxyz;
    else
         monXYZ(ch,:) = ...
        [Xs(end, ch) Ys(end, ch) Zs(end, ch)];
    end
end

%% LUT with non-linearities
if black_level
    radiometric = ([Xs(:, 4) Ys(:, 4) Zs(:, 4)] - black_levelxyz)*...             % 
        inv(monXYZ);
else
    radiometric = ([Xs(:, 4) Ys(:, 4) Zs(:, 4)])* inv(monXYZ);
end

%% Perform the validation of  the model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOOK AT TEST COLORS
load PredefinedRGB;

RGBStest = PredefinedRGB./255;
aux  = [Validation_rand];
for i=1:length(aux)
    XYZmeas(i, :) = aux(i).color.XYZ;
end

for ch = 1:3
    RGBStestLinear(:, ch) = ...
        interp1(x, radiometric(:, ch), RGBStest(:, ch));
    RGBSwhite(:, ch) = interp1(x, radiometric(:, ch), 1);
end

if black_level
    XYZ = RGBStestLinear * monXYZ + black_levelxyz;
    XYZwhite = RGBSwhite * monXYZ + black_levelxyz;
else
    XYZ = RGBStestLinear * monXYZ;
    XYZwhite = RGBSwhite * monXYZ;
end

xyY = XYZToxyY(XYZ')';
xyYmeas = XYZToxyY(XYZmeas')';

%% Compute deltae2000
lab_meas = xyz2lab(XYZmeas, 'whitepoint', white.color.XYZ'); 

lab_est  = xyz2lab(XYZ,  'whitepoint', XYZwhite);
lab_nocalib  = rgb2lab(RGBStest, 'whitepoint', [1 1 1], ...
    'ColorSpace','linear-rgb');

dE = deltaE00(lab_meas', lab_est');
dE_nocalib = deltaE00(lab_meas', lab_nocalib');

%% Plot validation results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;% xy chromaticity diagram
plotChromaticity();hold on
plot(xyY(:,1),xyY(:,2),'ko','MarkerSize',10,'LineWidth',2);
plot(xyYmeas(:,1),xyYmeas(:,2),'kx','markersize',12,'linewidth',2)
set(gca,'FontSize',15,'LineWidth',2)
box off
xlabel('x','FontSize',15)
ylabel('y','FontSize',15)
set(gca,  'FontSize', 30, 'fontname','TeXGyreTermes');
grid on
set(gcf,'renderer','Painters');


msize=15;
figure;% Lab colour space
subplot 131;
for i = 2:size(lab_est, 1)
    plot(lab_est(i, 2), lab_est(i, 3), 'o', 'color', RGBStest(i, :), ...
        'markerfacecolor', RGBStest(i, :), 'markersize', msize);hold on
    
    plot(lab_meas(i, 2), lab_meas(i, 3), 'kx', 'markersize', msize);hold on
    xlabel('a*','FontSize',15)
    ylabel('b*','FontSize',15)
end
axis equal
axis([min([lab_est(:, 2); lab_meas(:, 2)]) max([lab_est(:, 2);...
    lab_meas(:, 2)]) min([lab_est(:, 3); lab_meas(:, 3)]) ...
    max([lab_meas(:, 3);lab_est(:, 3)])])

subplot 132;
for i = 2:size(lab_est, 1)
    plot(lab_est(i, 2), lab_est(i, 1), 'o', 'color', RGBStest(i, :), ...
        'markerfacecolor', RGBStest(i, :), 'markersize', msize);hold on
    
    plot(lab_meas(i, 2), lab_meas(i, 1), 'kx', 'markersize', msize);hold on
    xlabel('a*','FontSize',15)
    ylabel('L*','FontSize',15)
end
axis equal
axis([min([lab_est(:, 2); lab_meas(:, 2)]) max([lab_est(:, 2);...
    lab_meas(:, 2)]) min([lab_est(:, 1); lab_meas(:, 1)]) ...
    max([lab_meas(:, 1);lab_est(:, 1)])])

subplot 133;
for i = 2:size(lab_est, 1)
    plot(lab_est(i, 3), lab_est(i, 1), 'o', 'color', RGBStest(i, :), ...
        'markerfacecolor', RGBStest(i, :), 'markersize', msize);hold on
    
    plot(lab_meas(i, 3), lab_meas(i, 1), 'kx', 'markersize', msize);hold on
    xlabel('b*','FontSize',15)
    ylabel('L*','FontSize',15)
end
axis equal
axis([min([lab_est(:, 3); lab_meas(:, 3)]) max([lab_est(:, 3);...
    lab_meas(:, 3)]) min([lab_est(:, 1); lab_meas(:, 1)]) ...
    max([lab_meas(:, 1);lab_est(:, 1)])])


%% Display errors and estimated parameters
disp 'deltaE00 -> mean, median, std, min and max'
disp(num2str([mean(dE) median(dE) std(dE) min(dE) max(dE)]))

disp 'deltaE00 no calibration -> mean, median, std, min and max'
disp(num2str([mean(dE_nocalib) median(dE_nocalib) ...
    std(dE_nocalib) min(dE_nocalib) max(dE_nocalib)]))

%% Save characterization values and deltae errors
try
    save(save_filename, 'monXYZ', 'radiometric', ...
        'dE', 'lab_meas', 'lab_est', 'dE_nocalib', 'black_level');
catch
    disp('No file name given for saving results');
end















%% Inverse
% 
% xyz_validation = 
% invMatrix = inv(monXYZ);
% RGB_nl = xyz_validation * invMatrix;
% 
% x = (0:5:255)./255;
% for ch = 1:3
%     RGB(:, ch) = interp1(radiometric(:, ch), x, RGB_nl(:, ch));   
% end

%% 
% Ignore (Not a part of the model)

% Assuming RGB input is a 24x3 matrix with values in [0, 255]
RGB_input = uselater_RGB; 


RGB_linear = zeros(size(RGB_input));
for ch = 1:3
    % Interpolate using radiometric LUT
    RGB_linear(:, ch) = interp1(x, radiometric(:, ch), RGB_input(:, ch), 'linear', 'extrap');
end


XYZ_output = RGB_linear * monXYZ;

if black_level
    XYZ_output = XYZ_output + black_levelxyz;
end

disp('Converted XYZ values:');
disp(XYZ_output);



%%

aux  = [Validation_rand];
for i=1:length(aux)
    XYZmeas(i, :) = aux(i).color.XYZ;
end
XYZcorr = XYZmeas - black_level * black_levelxyz;

inv_monXYZ = inv(monXYZ);
RGBlinear = XYZcorr * inv_monXYZ;

RGBnormalized = zeros(size(RGBlinear));
for ch = 1:3
    LUTout = radiometric(:, ch);
    [LUT_sorted, idx] = sort(LUTout);
    x_sorted = x(idx);

    RGBch = RGBlinear(:, ch);
    RGBch = max(min(RGBch, max(LUT_sorted)), min(LUT_sorted));
    RGBnormalized(:, ch) = interp1(LUT_sorted, x_sorted, RGBch, 'linear');
end

RGBnormalized = min(max(RGBnormalized, 0), 1);
RGB_estimated = round(RGBnormalized * 255);

euclidean = sqrt(sum((RGB_estimated - PredefinedRGB).^2, 2));
mae = mean(abs(RGB_estimated - PredefinedRGB), 2);
rmse = sqrt(mean((RGB_estimated - PredefinedRGB).^2, 2));

fprintf('Mean Euclidean: %.2f\n', mean(euclidean));
fprintf('Max Euclidean: %.2f\n', max(euclidean));
fprintf('Mean RMSE: %.2f\n', mean(rmse));
fprintf('Mean Absolute Error: %.2f\n', mean(mae));


%%
 %   save('ValidationRGB_est_WWQ2.mat', 'XYZmeas', 'XYZcorr', 'RGB_estimated');

%save('ValidationRGB_est_WWPoly.mat', 'XYZmeas', 'XYZcorr', 'RGB_estimated');
%save('wp_paintingD.mat', )
% 
% %%
