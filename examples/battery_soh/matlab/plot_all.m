function plot_all()
% PLOT_ALL  Reproduce the battery State-of-Health figures in MATLAB.
%
% Reads the CSVs exported by run_battery_soh.py (../figure_data/*.csv) and
% renders publication-grade figures with MATLAB. Run from this folder:
%
%   >> plot_all
%
% Figures are written to ../figures_matlab/*.png (300 dpi). Edit the STYLE
% block below to match your house style; all the numbers live in the CSVs, so
% nothing here is hard-coded.

here = fileparts(mfilename('fullpath'));
dataDir = fullfile(here, '..', 'figure_data');
outDir  = fullfile(here, '..', 'figures_matlab');
if ~exist(outDir, 'dir'); mkdir(outDir); end

% ---- STYLE -------------------------------------------------------------
set(groot, 'defaultAxesFontName', 'Helvetica');
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultLineLineWidth', 1.8);
cObs  = [0.13 0.13 0.13];
cPhys = [0.12 0.47 0.71];
cML   = [0.84 0.15 0.16];
cHyb  = [0.17 0.63 0.17];
EOL   = 0.80;

% ---- 1. Hero forecast (B0005) -----------------------------------------
fleet = readtable(fullfile(dataDir, '00_fleet_overview.csv'));
hero  = fleet(strcmp(fleet.cell, 'B0005'), :);
T     = readtable(fullfile(dataDir, '01_hero_forecast_test.csv'));

f = figure('Color', 'w', 'Position', [100 100 760 480]); hold on
xb = [T.cycle; flipud(T.cycle)];
yb = [T.lower; flipud(T.upper)];
fill(xb, yb, cHyb, 'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
     'DisplayName', '90% interval');
plot(hero.cycle, hero.soh, 'o', 'MarkerSize', 3, 'Color', cObs, ...
     'MarkerFaceColor', cObs, 'DisplayName', 'Observed SoH');
plot(T.cycle, T.hybrid_mean, '-',  'Color', cHyb,  'DisplayName', 'Hybrid forecast');
plot(T.cycle, T.physics,     '--', 'Color', cPhys, 'DisplayName', 'Physics-only');
plot(T.cycle, T.ml,          '-.', 'Color', cML,   'DisplayName', 'ML-only (GP)');
yline(EOL, 'k--', 'EOL 80%', 'LineWidth', 0.8, 'FontSize', 9);
xlabel('Cycle'); ylabel('State of Health');
title('B0005: SoH forecast under a temporal split');
legend('Location', 'southwest', 'Box', 'off'); grid on
exportgraphics(f, fullfile(outDir, '01_hero_forecast.png'), 'Resolution', 300);

% ---- 2. Method comparison ---------------------------------------------
M = readtable(fullfile(dataDir, '02_method_comparison.csv'));
f = figure('Color', 'w', 'Position', [100 100 680 440]);
b = bar(categorical(M.method, M.method), M.rmse, 0.6);
b.FaceColor = 'flat';
for i = 1:height(M); b.CData(i,:) = cHyb; end
ylabel('Fleet-mean test RMSE (SoH)');
title('Forecast accuracy by method (lower is better)'); grid on
exportgraphics(f, fullfile(outDir, '02_method_comparison.png'), 'Resolution', 300);

% ---- 3. Calibration ----------------------------------------------------
C = readtable(fullfile(dataDir, '03_calibration.csv'));
f = figure('Color', 'w', 'Position', [100 100 520 520]); hold on
plot([0 1], [0 1], 'k--', 'DisplayName', 'Perfect calibration');
plot(C.nominal, C.empirical, 'o-', 'Color', cHyb, 'DisplayName', 'Hybrid (fleet)');
xlabel('Nominal coverage'); ylabel('Empirical coverage');
title('Calibration across the fleet'); axis equal; xlim([0 1]); ylim([0 1]);
legend('Location', 'northwest', 'Box', 'off'); grid on
exportgraphics(f, fullfile(outDir, '03_calibration.png'), 'Resolution', 300);

% ---- 4. PIT histogram --------------------------------------------------
P = readtable(fullfile(dataDir, '04_pit_histogram.csv'));
f = figure('Color', 'w', 'Position', [100 100 600 400]); hold on
histogram(P.pit, 12, 'Normalization', 'pdf', 'FaceColor', cHyb, 'FaceAlpha', 0.7);
yline(1, 'k--', 'Uniform (ideal)');
xlabel('PIT value'); ylabel('Density');
title('Probability Integral Transform (fleet)'); grid on
exportgraphics(f, fullfile(outDir, '04_pit_histogram.png'), 'Resolution', 300);

% ---- 5. Uncertainty vs horizon ----------------------------------------
H = readtable(fullfile(dataDir, '05_horizon.csv'));
f = figure('Color', 'w', 'Position', [100 100 680 440]); hold on
plot(H.h, H.width,  '-', 'Color', cHyb, 'DisplayName', '90% interval width');
plot(H.h, H.abserr, '-', 'Color', cObs, 'DisplayName', 'Mean abs. error');
xlabel('Forecast horizon (cycles ahead)'); ylabel('SoH');
title('Uncertainty grows with the forecast horizon');
legend('Box', 'off'); grid on
exportgraphics(f, fullfile(outDir, '05_horizon.png'), 'Resolution', 300);

% ---- 6. RUL parity -----------------------------------------------------
if isfile(fullfile(dataDir, '07_rul_parity.csv'))
    R = readtable(fullfile(dataDir, '07_rul_parity.csv'));
    f = figure('Color', 'w', 'Position', [100 100 560 560]); hold on
    lo = min([R.true_eol; R.pred_med]) - 5; hi = max([R.true_eol; R.pred_med]) + 5;
    plot([lo hi], [lo hi], 'k--');
    neg = R.pred_med - R.p05; pos = R.p95 - R.pred_med;
    errorbar(R.true_eol, R.pred_med, neg, pos, 'o', 'Color', cHyb, ...
             'MarkerFaceColor', cHyb, 'CapSize', 4, 'LineStyle', 'none');
    text(R.true_eol, R.pred_med, R.cell, 'FontSize', 7, ...
         'VerticalAlignment', 'bottom');
    xlabel('True EOL cycle'); ylabel('Predicted EOL cycle (median + 90%)');
    title('Remaining-useful-life: predicted vs true'); axis equal; grid on
    exportgraphics(f, fullfile(outDir, '07_rul_parity.png'), 'Resolution', 300);
end

% ---- 7. Skill vs persistence ------------------------------------------
S = readtable(fullfile(dataDir, '09_skill.csv'));
[~, idx] = sort(S.skill); S = S(idx, :);
f = figure('Color', 'w', 'Position', [100 100 680 480]);
cols = repmat(cHyb, height(S), 1); cols(S.skill < 0, :) = repmat(cML, sum(S.skill < 0), 1);
b = barh(categorical(S.cell, S.cell), S.skill, 0.6); b.FaceColor = 'flat'; b.CData = cols;
xline(0, 'k-'); xlabel('Skill vs persistence'); grid on
title('Forecast skill over the naive baseline, per cell');
exportgraphics(f, fullfile(outDir, '09_skill.png'), 'Resolution', 300);

fprintf('Wrote MATLAB figures to %s\n', outDir);
end
