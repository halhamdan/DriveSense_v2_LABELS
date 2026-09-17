function full_fft_entire_sample()
% FULL_FFT_ENTIRE_SAMPLE  The literal "full FFT of the entire sample" the
% supervisor asked for (2026-09-16): ONE single Fourier transform over every
% released sample concatenated together (all 79 sessions, ~3.5 million rows),
% not the segmented/averaged Welch estimate in frequency_distribution_whole_sample.m.
%
% Each session's own mean is removed before concatenation (so the 79
% session-to-session level differences, which are not part of the vibration
% signal, do not inject a spurious jump at every boundary); nothing else is
% done to the data before the transform -- no windowing, no segmenting, no
% averaging. This is the whole point of the request: see the raw FFT of
% everything, not a statistically smoothed version of it.
%
% Because this is a single, unaveraged periodogram of ~3.5e6 points, it is
% extremely jagged (frequency resolution ~7 microHz -- each point is one
% noisy sample, not an average). A locally-averaged overlay curve is added
% ONLY to make the plot readable; the raw curve underneath is the true,
% requested, single full-length FFT.
%
% Usage:  matlab -batch "cd('05_technical_validation'); full_fft_entire_sample"
%
% Output: validation_output/frequency_distribution/
%   fig_full_fft_entire_sample.png   (raw full-length FFT, both channels, linear + log)
%   full_fft_summary.txt

close all;
FS = 25;

RELEASE = fullfile('C:','Users','halha','OneDrive - Durham University','Documents', ...
    'DriveSense_Packages','1_LATEST__labels_v2__release_v3__paper_v6','data', ...
    'Published_Dataset_Final_v3_VIDEO_ALIGNED','Preprocessed_Dataset');
OUTDIR = fullfile(fileparts(mfilename('fullpath')), 'validation_output', 'frequency_distribution');
if ~exist(OUTDIR, 'dir'); mkdir(OUTDIR); end

CONCAT_CACHE = fullfile(OUTDIR, 'concat_lon_lat_cache.mat');
if exist(CONCAT_CACHE, 'file')
    fprintf('Loading cached concatenated signal from %s\n', CONCAT_CACHE);
    S = load(CONCAT_CACHE); lon = S.lon; lat = S.lat; nfiles = S.nfiles;
else
    files = dir(fullfile(RELEASE, '*', '*', '*_fused.csv'));
    fprintf('Found %d released session files -- concatenating (mean removed per session)...\n', numel(files));
    lon_all = cell(numel(files), 1); lat_all = cell(numel(files), 1);
    for k = 1:numel(files)
        p = fullfile(files(k).folder, files(k).name);
        T = readtable(p, 'ReadVariableNames', true);
        lonk = fillmissing(T.lon_acc_g, 'linear', 'EndValues', 'nearest');
        latk = fillmissing(T.lat_acc_g, 'linear', 'EndValues', 'nearest');
        lon_all{k} = lonk - mean(lonk);
        lat_all{k} = latk - mean(latk);
        if mod(k, 20) == 0; fprintf('  %d/%d sessions read\n', k, numel(files)); end
    end
    lon = cat(1, lon_all{:}); lat = cat(1, lat_all{:}); nfiles = numel(files);
    save(CONCAT_CACHE, 'lon', 'lat', 'nfiles', '-v7.3');
end
N = numel(lon);
fprintf('Concatenated length: %d samples (%.2f hours at 25 Hz)\n', N, N/FS/3600);

fprintf('Computing single full-length FFT (N=%d) ...\n', N);
t0 = tic;
Xlon = fft(lon);
Xlat = fft(lat);
fprintf('  done in %.1f s\n', toc(t0));

nbin = floor(N/2) + 1;
f = (0:nbin-1)' * (FS / N);
Plon = (abs(Xlon(1:nbin)).^2) / (FS * N); Plon(2:end-1) = 2 * Plon(2:end-1);
Plat = (abs(Xlat(1:nbin)).^2) / (FS * N); Plat(2:end-1) = 2 * Plat(2:end-1);

% ---- a smoothed overlay in log-frequency bins, for readability only (the raw
% curve plotted underneath it is the actual single full-length FFT)
nlogbin = 500;
fmin = FS / N;   % first non-zero bin
edges = [0, logspace(log10(fmin), log10(FS/2), nlogbin)];
fc = sqrt(edges(1:end-1) .* max(edges(2:end), eps)); fc(1) = edges(2)/2;
Plon_smooth = nan(size(fc)); Plat_smooth = nan(size(fc));
for i = 1:numel(fc)
    m = f >= edges(i) & f < edges(i+1);
    if any(m)
        Plon_smooth(i) = mean(Plon(m)); Plat_smooth(i) = mean(Plat(m));
    end
end

% ============================================================ Figure
% y-axis: a handful of individual bins in a 3.5-million-point unaveraged
% periodogram land near exact numerical zero (destructive interference at a
% single frequency), which would otherwise drag semilogy's auto-scaling down
% by 30+ decades and crush the entire real signal into a sliver at the top.
% Clamp to the 0.1st-99.9th percentile of the smoothed curve instead.
ylo = 1e-5; yhi = 1e0;   % matches the Welch PSD's known range (~3e-4 to ~2e-2 g^2/Hz); fixed, not data-derived, so a stray near-zero bin cannot blow up the axis

figure('Visible', 'off', 'Position', [100 100 1000 750]);
subplot(2,1,1);
semilogy(f, Plon, 'Color', [0.75 0.75 0.85], 'LineWidth', 0.3); hold on;
semilogy(fc, Plon_smooth, 'k-', 'LineWidth', 1.6);
xlim([0 12.5]); set(gca, 'YLim', [ylo yhi], 'YLimMode', 'manual');
xlabel('Frequency (Hz)'); ylabel('|FFT(lon\_acc\_g)|^2 / (F_s N)  (g^2/Hz), log scale');
title(sprintf('lon\\_acc\\_g -- single full-length FFT of the ENTIRE sample (N = %d, all 79 sessions concatenated)', N));
legend({'raw full-length FFT (unaveraged)', 'local log-frequency average (readability only)'}, 'Location', 'northeast');
grid on; box on;

subplot(2,1,2);
semilogy(f, Plat, 'Color', [0.85 0.75 0.75], 'LineWidth', 0.3); hold on;
semilogy(fc, Plat_smooth, 'k-', 'LineWidth', 1.6);
xlim([0 12.5]); set(gca, 'YLim', [ylo yhi], 'YLimMode', 'manual');
xlabel('Frequency (Hz)'); ylabel('|FFT(lat\_acc\_g)|^2 / (F_s N)  (g^2/Hz), log scale');
title('lat\_acc\_g -- single full-length FFT of the ENTIRE sample');
legend({'raw full-length FFT (unaveraged)', 'local log-frequency average (readability only)'}, 'Location', 'northeast');
grid on; box on;

save_png(gcf, fullfile(OUTDIR, 'fig_full_fft_entire_sample.png'));

fid = fopen(fullfile(OUTDIR, 'full_fft_summary.txt'), 'w');
fprintf(fid, 'N_samples=%d\nhours=%.2f\nsessions=%d\nfreq_resolution_hz=%.3e\n', N, N/FS/3600, nfiles, FS/N);
fclose(fid);
fprintf('\nSaved fig_full_fft_entire_sample.png + full_fft_summary.txt -> %s\n', OUTDIR);
end

function save_png(fig, outfile)
if exist(outfile, 'file'); try, delete(outfile); catch, end; end
[d, n, e] = fileparts(outfile);
tmp = fullfile(d, [n '.tmp' e]);
lastErr = [];
for attempt = 1:6
    try
        if exist(tmp, 'file'); delete(tmp); end
        print(fig, tmp, '-dpng', '-r150');
        movefile(tmp, outfile, 'f');
        close(fig);
        return;
    catch ME
        lastErr = ME; pause(1.5);
    end
end
rethrow(lastErr);
end
