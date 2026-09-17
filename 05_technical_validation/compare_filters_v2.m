function compare_filters_v2()
% COMPARE_FILTERS_V2  Requested by the supervisor (2026-09-16): compare an FFT
% (frequency-domain) filter against the median+mean smoothing currently used by
% the version-2 harsh-event rule, and against a wavelet ("wnt") filter, on the
% real released lon_acc_g / lat_acc_g signals. Draws the signal before and
% after each filter. Read-only: does not touch label_harsh_events_v2.py or any
% released file; purely a diagnostic/comparison figure set for the supervisor.
%
% Usage:  matlab -batch "cd('05_technical_validation'); compare_filters_v2"
%
% Output: 05_technical_validation/validation_output/filter_comparison/
%   fig1_D11_S2_lon_window.png      fig2_D11_S2_lat_window.png
%   fig3_D11_S4_braking_window.png  fig4_zoom_turning_event.png
%   fig5_frequency_spectrum.png     fig6a_long_strip_lon.png  fig6b_long_strip_lat.png
%   filter_comparison_metrics.csv
% Figures 1-4 and 6a/6b each show the raw signal and every filter's output in
% its own stacked panel (one under the other), not overlapped on one axes.
%
% Data source: release v3 (version-2 labels), the same file the manuscript
% and label pipeline use.
%
% Three filters compared, all applied to the WHOLE session signal first (so
% none of the three has an edge artefact at the display-window boundary),
% then sliced to the windows below for plotting:
%   1. Currently-adopted smoothing (label_rule_variants.py): 5-sample centred
%      median, then 13-sample centred mean, both at 25 Hz (0.20 s / 0.52 s).
%   2. FFT (frequency-domain) low-pass filter: rfft -> zero all bins above the
%      cutoff (and their mirror) -> irfft. Zero phase (no time shift), unlike a
%      causal IIR filter. Cutoff = 2 Hz chosen to match the -3dB-ish rolloff of
%      the 13-sample mean already in the rule (first null at 1/0.52 = 1.92 Hz);
%      1 Hz and 3 Hz are also produced for sensitivity.
%   3. Wavelet ("wnt") denoising: wdenoise() (Wavelet Toolbox, db4, level 4,
%      soft universal threshold) if licensed; otherwise a manual multi-level
%      Haar DWT + soft-threshold + inverse DWT fallback (implemented below, no
%      toolbox dependency) so the comparison always runs.

close all;
FS = 25;                 % Hz, common fused timeline
DT = 1/FS;

RELEASE = fullfile('C:','Users','halha','OneDrive - Durham University','Documents', ...
    'DriveSense_Packages','1_LATEST__labels_v2__release_v3__paper_v6','data', ...
    'Published_Dataset_Final_v3_VIDEO_ALIGNED','Preprocessed_Dataset');
OUTDIR = fullfile(fileparts(mfilename('fullpath')), 'validation_output', 'filter_comparison');
if ~exist(OUTDIR, 'dir'); mkdir(OUTDIR); end
% Clear any existing PNGs first: a previous run's file, if left open by an
% image viewer / thumbnail indexer / antivirus scan, is what caused
% "PNG library failed: Could not open file" from saveas() on this machine
% when print() tried to overwrite it in place.
old = dir(fullfile(OUTDIR, '*.png'));
for i = 1:numel(old)
    try, delete(fullfile(old(i).folder, old(i).name)); catch, end
end

haveWavelet = license('test', 'Wavelet_Toolbox') && ~isempty(which('wdenoise'));
fprintf('Wavelet Toolbox available: %d\n', haveWavelet);

% ---- sessions used for the demonstration windows (chosen from harsh_events_v2.csv:
% D11 has the most events of any driver; D11_S2 has an acceleration + a turning
% event 7.5 s apart; D11_S4 has the largest braking event, peak 0.90 g)
S2 = load_session(RELEASE, 11, 2);
S4 = load_session(RELEASE, 11, 4);

% ---- apply all three filters to the FULL session signal (avoids window-edge
% artefacts in the FFT and wavelet transforms)
S2.lon_meanmed = meanmed_filter(S2.lon);   S2.lat_meanmed = meanmed_filter(S2.lat);
S2.lon_fft2 = fft_lowpass(S2.lon, FS, 2.0); S2.lat_fft2 = fft_lowpass(S2.lat, FS, 2.0);
S2.lon_fft1 = fft_lowpass(S2.lon, FS, 1.0); S2.lat_fft1 = fft_lowpass(S2.lat, FS, 1.0);
S2.lon_fft3 = fft_lowpass(S2.lon, FS, 3.0); S2.lat_fft3 = fft_lowpass(S2.lat, FS, 3.0);
S2.lon_wav = wavelet_denoise(S2.lon, haveWavelet); S2.lat_wav = wavelet_denoise(S2.lat, haveWavelet);

S4.lon_meanmed = meanmed_filter(S4.lon);
S4.lon_fft2 = fft_lowpass(S4.lon, FS, 2.0);
S4.lon_wav = wavelet_denoise(S4.lon, haveWavelet);

% ============================================================ Figure 1
% D11_S2, 180-210 s: lon_acc_g, contains the Harsh Acceleration event
% 190.24-190.80 s (peak smoothed 0.241 g). Each filter drawn in its own
% panel, stacked one under the other, rather than overlapped.
w = window_idx(S2.t, 180, 210);
plot_stacked(S2.t(w), S2.lon(w), S2.lon_meanmed(w), S2.lon_fft2(w), S2.lon_wav(w), ...
    [190.24 190.80], 0.20, 'lon\_acc\_g  (g)', ...
    'D11\_S2, 180-210 s: longitudinal acceleration (Harsh Acceleration event shaded)', ...
    fullfile(OUTDIR, 'fig1_D11_S2_lon_window.png'));

% ============================================================ Figure 2
% Same window, lat_acc_g: contains the Harsh Turning event 197.76-198.24 s
% (peak smoothed 0.729 g)
plot_stacked(S2.t(w), S2.lat(w), S2.lat_meanmed(w), S2.lat_fft2(w), S2.lat_wav(w), ...
    [197.76 198.24], 0.45, 'lat\_acc\_g  (g)', ...
    'D11\_S2, 180-210 s: lateral acceleration (Harsh Turning event shaded)', ...
    fullfile(OUTDIR, 'fig2_D11_S2_lat_window.png'));

% ============================================================ Figure 3
% D11_S4, 210-235 s: lon_acc_g, Harsh Braking event 223.64-224.16 s
% (peak smoothed 0.896 g -- the largest braking event in the release)
w4 = window_idx(S4.t, 210, 235);
plot_stacked(S4.t(w4), S4.lon(w4), S4.lon_meanmed(w4), S4.lon_fft2(w4), S4.lon_wav(w4), ...
    [223.64 224.16], -0.35, 'lon\_acc\_g  (g)', ...
    'D11\_S4, 210-235 s: longitudinal acceleration (Harsh Braking event shaded)', ...
    fullfile(OUTDIR, 'fig3_D11_S4_braking_window.png'));

% ============================================================ Figure 4
% Zoomed single event (the turning event, +/-2 s), each filter in its own panel
wz = window_idx(S2.t, 195.76, 200.24);
plot_stacked(S2.t(wz), S2.lat(wz), S2.lat_meanmed(wz), S2.lat_fft2(wz), S2.lat_wav(wz), ...
    [197.76 198.24], 0.45, 'lat\_acc\_g  (g)', ...
    'D11\_S2 turning event, zoomed', ...
    fullfile(OUTDIR, 'fig4_zoom_turning_event.png'));

% ============================================================ Figure 5
% Frequency-domain view: where does the FFT cutoff sit relative to the actual
% spectral content of the raw signal, and how does the mean(13) filter's own
% frequency response compare? A raw single-session periodogram of ~150,000
% points is illegible (pure noise floor); Welch's method (pwelch, averaged,
% overlapped segments) gives the smooth, physically-interpretable PSD needed
% to justify the cutoff choice.
figure('Visible', 'off', 'Position', [100 100 900 550]);
seg = 4096; ov = round(seg/2);
[Pxx_lon, fax] = pwelch(S2.lon - mean(S2.lon), hann(seg), ov, seg, FS);
[Pxx_lat, ~]   = pwelch(S2.lat - mean(S2.lat), hann(seg), ov, seg, FS);
subplot(2,1,1);
semilogy(fax, Pxx_lon, 'k-', 'LineWidth', 1.2); hold on;
xlim([0 6]);
xlabel('Frequency (Hz)'); ylabel('PSD (g^2/Hz), log scale');
title('D11\_S2 lon\_acc\_g -- Welch PSD, whole session (4096-sample Hann segments, 50% overlap)');
grid on; box on;
subplot(2,1,2);
semilogy(fax, Pxx_lat, 'k-', 'LineWidth', 1.2); hold on;
xlim([0 6]);
xlabel('Frequency (Hz)'); ylabel('PSD (g^2/Hz), log scale');
title('D11\_S2 lat\_acc\_g -- Welch PSD, whole session');
grid on; box on;
save_png(gcf, fullfile(OUTDIR, 'fig5_frequency_spectrum.png'));
close(gcf);

% ============================================================ Figure 6
% Longer ordinary-driving strip (no labelled event) to see flat-region noise
% suppression, not just event preservation. One channel per file, one filter
% per panel (no event span/threshold here: this stretch has no labelled event).
w6 = window_idx(S2.t, 300, 600);
plot_stacked(S2.t(w6), S2.lon(w6), S2.lon_meanmed(w6), S2.lon_fft2(w6), S2.lon_wav(w6), ...
    [], NaN, 'lon\_acc\_g  (g)', 'D11\_S2, 300-600 s (ordinary driving, no labelled event)', ...
    fullfile(OUTDIR, 'fig6a_long_strip_lon.png'));
plot_stacked(S2.t(w6), S2.lat(w6), S2.lat_meanmed(w6), S2.lat_fft2(w6), S2.lat_wav(w6), ...
    [], NaN, 'lat\_acc\_g  (g)', 'D11\_S2, 300-600 s (ordinary driving, no labelled event)', ...
    fullfile(OUTDIR, 'fig6b_long_strip_lat.png'));

% ============================================================ Quantitative metrics
baseline = window_idx(S2.t, 300, 330);   % 30 s of ordinary driving, no event
rows = {};
rows = add_metric_rows(rows, 'D11_S2 lon_acc_g', S2.lon, S2.lon_meanmed, S2.lon_fft1, S2.lon_fft2, S2.lon_fft3, S2.lon_wav, baseline, window_idx(S2.t, 190.24, 190.80));
rows = add_metric_rows(rows, 'D11_S2 lat_acc_g', S2.lat, S2.lat_meanmed, S2.lat_fft1, S2.lat_fft2, S2.lat_fft3, S2.lat_wav, baseline, window_idx(S2.t, 197.76, 198.24));
T = cell2table(rows, 'VariableNames', {'signal','filter','noise_std_baseline','pct_noise_reduction', ...
    'raw_peak_g','filtered_peak_g','pct_peak_retained','corr_with_adopted_meanmed'});
writetable(T, fullfile(OUTDIR, 'filter_comparison_metrics.csv'));
disp(T);

fprintf('\nSaved 6 figures + filter_comparison_metrics.csv -> %s\n', OUTDIR);
end

% ================================================================ helpers

function S = load_session(root, driver, session)
tag = sprintf('D%d_S%d', driver, session);
p = fullfile(root, sprintf('D%d', driver), sprintf('Session_%d', session), [tag '_fused.csv']);
T = readtable(p);
S.t = T.elapsed_s;
S.lon = T.lon_acc_g;
S.lat = T.lat_acc_g;
S.tag = tag;
end

function idx = window_idx(t, t0, t1)
idx = find(t >= t0 & t <= t1);
end

function y = meanmed_filter(x)
% Exactly the currently-adopted rule (label_rule_variants.py): 5-sample
% centred median (k=-2..+2), then 13-sample centred mean (k=-6..+6).
x = x(:);
n = numel(x);
m = x;
for i = 1:n
    a = max(1, i-2); b = min(n, i+2);
    m(i) = median(x(a:b));
end
y = m;
for i = 1:n
    a = max(1, i-6); b = min(n, i+6);
    y(i) = mean(m(a:b));
end
end

function y = fft_lowpass(x, fs, cutoff_hz)
% Zero-phase FFT low-pass: transform the whole signal, zero every frequency
% bin (and its mirror) above cutoff_hz, inverse transform. Zero phase shift,
% unlike a causal IIR/FIR filter -- matches the fact that the adopted
% median/mean filter is also zero-phase (centred windows).
x = x(:);
n = numel(x);
X = fft(x);
f = (0:n-1) * (fs / n);
f(f > fs/2) = f(f > fs/2) - fs;      % fold to [-fs/2, fs/2] for a correct mirror test
keep = abs(f) <= cutoff_hz;
X(~keep) = 0;
y = real(ifft(X));
end

function y = wavelet_denoise(x, haveWavelet)
x = x(:);
if haveWavelet
    y = wdenoise(x, 4, 'Wavelet', 'db4', 'DenoisingMethod', 'Universal', ...
        'ThresholdRule', 'Soft', 'NoiseEstimate', 'LevelIndependent');
    return;
end
% ---- manual fallback: 4-level Haar DWT, soft-threshold the detail
% coefficients at each level (universal threshold from the finest level's
% MAD), inverse DWT. No toolbox dependency.
level = 4;
n0 = numel(x);
pad = mod(-n0, 2^level);
xp = [x; repmat(x(end), pad, 1)];
[A, Ds] = haar_dwt(xp, level);
sigma = mad_est(Ds{1}) / 0.6745;
thr = sigma * sqrt(2 * log(numel(xp)));
for L = 1:level
    Ds{L} = soft_threshold(Ds{L}, thr);
end
xr = haar_idwt(A, Ds);
y = xr(1:n0);
end

function m = mad_est(v)
m = median(abs(v - median(v)));
end

function y = soft_threshold(v, t)
y = sign(v) .* max(abs(v) - t, 0);
end

function [A, Ds] = haar_dwt(x, level)
Ds = cell(level, 1);
A = x;
for L = 1:level
    n = numel(A);
    a = (A(1:2:n-1) + A(2:2:n)) / sqrt(2);
    d = (A(1:2:n-1) - A(2:2:n)) / sqrt(2);
    Ds{L} = d;
    A = a;
end
end

function x = haar_idwt(A, Ds)
level = numel(Ds);
for L = level:-1:1
    d = Ds{L};
    n = numel(A);
    x = zeros(2*n, 1);
    x(1:2:end) = (A + d) / sqrt(2);
    x(2:2:end) = (A - d) / sqrt(2);
    A = x;
end
x = A;
end

function plot_stacked(t, raw, meanmed, fftf, wav, event_span, thr, ylab, ttl, outfile)
% Each signal drawn in its own panel, one under the other -- no overlapping
% traces. Row 1: raw only. Rows 2-4: raw (thin grey, for reference) plus one
% filter's output (in colour). event_span (may be [] for "no event to mark")
% is shaded in every row; thr (may be NaN for "no threshold to draw") is
% drawn as a dotted line in every row.
panels = {raw, meanmed, fftf, wav};
names  = {'Raw signal (before any filter)', ...
          'After: median(5) + mean(13)  [adopted]', ...
          'After: FFT low-pass, 2 Hz', ...
          'After: wavelet denoise (db4)'};
colors = {[0.45 0.45 0.45], [0 0.30 0.80], [0.80 0.10 0.10], [0 0.55 0]};

figure('Visible', 'off', 'Position', [100 100 1000 900]);
yl_all = [min(raw) max(raw)];
pad = 0.08 * (yl_all(2) - yl_all(1) + eps);
yl_all = yl_all + [-pad, pad];

for k = 1:4
    subplot(4, 1, k);
    hold on;
    if ~isempty(event_span)
        patch([event_span(1) event_span(2) event_span(2) event_span(1)], ...
              [yl_all(1) yl_all(1) yl_all(2) yl_all(2)], [1 0.9 0.8], ...
              'EdgeColor', 'none', 'FaceAlpha', 0.6, 'HandleVisibility', 'off');
    end
    if k > 1
        plot(t, raw, 'Color', [0.82 0.82 0.82], 'LineWidth', 0.8, 'DisplayName', 'raw (reference)');
    end
    plot(t, panels{k}, 'Color', colors{k}, 'LineWidth', 1.5, 'DisplayName', names{k});
    if ~isnan(thr)
        yline(thr, 'k:', sprintf('threshold %.2f g', thr), 'HandleVisibility', 'off');
    end
    ylim(yl_all); xlim([t(1) t(end)]);
    ylabel(ylab);
    title(names{k}, 'FontWeight', 'normal');
    if k < 4; set(gca, 'XTickLabel', []); else; xlabel('elapsed\_s'); end
    grid on; box on;
end
sgtitle(ttl, 'Interpreter', 'none');
save_png(gcf, outfile);
close(gcf);
end

function save_png(fig, outfile)
% Robust PNG export: writes to a temp file with print() (bitmap driver,
% 150 dpi) and moves it into place, retrying through the transient
% "PNG library failed: Could not open file" error that a fresh Windows
% file-lock (thumbnail indexer / antivirus scan / previous viewer handle)
% can cause on the very first write to a given filename.
if exist(outfile, 'file')
    try, delete(outfile); catch, end
end
[d, n, e] = fileparts(outfile);
tmp = fullfile(d, [n '.tmp' e]);
lastErr = [];
for attempt = 1:6
    try
        if exist(tmp, 'file'); delete(tmp); end
        print(fig, tmp, '-dpng', '-r150');
        movefile(tmp, outfile, 'f');
        return;
    catch ME
        lastErr = ME;
        pause(1.5);
    end
end
rethrow(lastErr);
end

function rows = add_metric_rows(rows, sigName, raw, meanmed, fft1, fft2, fft3, wav, baseIdx, evtIdx)
raw_peak = max(abs(raw(evtIdx)));
raw_noise = std(raw(baseIdx));
filters = {'median(5)+mean(13) [adopted]', meanmed; 'FFT low-pass 1 Hz', fft1; ...
           'FFT low-pass 2 Hz', fft2; 'FFT low-pass 3 Hz', fft3; 'wavelet denoise', wav};
for i = 1:size(filters,1)
    name = filters{i,1}; y = filters{i,2};
    ns = std(y(baseIdx));
    pk = max(abs(y(evtIdx)));
    r = corr(y, meanmed);
    rows(end+1, :) = {sigName, name, raw_noise_report(ns), pct(1 - ns/raw_noise), raw_peak, pk, pct(pk/raw_peak), r}; %#ok<AGROW>
end
end

function v = raw_noise_report(x)
v = x;
end

function p = pct(x)
p = round(100 * x, 1);
end
