function frequency_distribution_whole_sample()
% FREQUENCY_DISTRIBUTION_WHOLE_SAMPLE  Requested by the supervisor (2026-09-16):
% the frequency distribution of the WHOLE sample (all 79 released sessions, not
% one demo session as in compare_filters_v2.m's fig5), computed by FFT.
%
% Method: Welch's method (periodogram averaging) applied correctly across many
% separate recordings of different lengths -- the dataset cannot simply be
% concatenated into one long vector, because the artificial jump at every
% session boundary (79 discontinuities) would inject broadband energy into the
% spectrum that is not part of the real signal. Instead: split EVERY session
% into 4096-sample (163.8 s) Hann-windowed segments with 50% overlap, never
% letting a segment cross a session boundary, compute each segment's own
% periodogram, then average all of them together (pooled over every session --
% longer sessions contribute more segments, in proportion to their length, so
% this is a genuine whole-dataset average, not an average-of-79-equal-votes).
%
% Usage:  matlab -batch "cd('05_technical_validation'); frequency_distribution_whole_sample"
%
% Output: validation_output/frequency_distribution/
%   fig_whole_sample_psd.png          (lon_acc_g and lat_acc_g, log PSD, cutoff marked)
%   fig_cumulative_energy.png         (fraction of total power below frequency f)
%   band_energy_table.csv             (power fraction in each frequency band)
%   n_segments_used.txt

close all;
FS = 25;                 % Hz
SEG = 4096;               % samples per segment (163.84 s)
OV  = SEG / 2;            % 50% overlap
WIN = hann(SEG, 'periodic');
U   = sum(WIN.^2);        % window power, for PSD normalisation

RELEASE = fullfile('C:','Users','halha','OneDrive - Durham University','Documents', ...
    'DriveSense_Packages','1_LATEST__labels_v2__release_v3__paper_v6','data', ...
    'Published_Dataset_Final_v3_VIDEO_ALIGNED','Preprocessed_Dataset');
OUTDIR = fullfile(fileparts(mfilename('fullpath')), 'validation_output', 'frequency_distribution');
if ~exist(OUTDIR, 'dir'); mkdir(OUTDIR); end
% Only this script's own output files -- OUTDIR is shared with
% full_fft_entire_sample.m (fig_full_fft_entire_sample.png), which a
% wildcard '*.png' delete here would silently wipe out.
old = [dir(fullfile(OUTDIR, 'fig_whole_sample_psd.png')); dir(fullfile(OUTDIR, 'fig_cumulative_energy.png'))];
for i = 1:numel(old); try, delete(fullfile(old(i).folder, old(i).name)); catch, end; end

files = dir(fullfile(RELEASE, '*', '*', '*_fused.csv'));
fprintf('Found %d released session files\n', numel(files));

nbin = SEG/2 + 1;
f = (0:nbin-1)' * (FS / SEG);
sumP_lon = zeros(nbin, 1); sumP_lat = zeros(nbin, 1);
nseg_lon = 0; nseg_lat = 0;
n_sessions_used = 0; n_sessions_skipped = 0;
total_rows = 0;

for k = 1:numel(files)
    p = fullfile(files(k).folder, files(k).name);
    T = readtable(p, 'ReadVariableNames', true);
    if ~all(ismember({'lon_acc_g','lat_acc_g'}, T.Properties.VariableNames))
        n_sessions_skipped = n_sessions_skipped + 1;
        continue;
    end
    lon = fillmissing(T.lon_acc_g, 'linear', 'EndValues', 'nearest');
    lat = fillmissing(T.lat_acc_g, 'linear', 'EndValues', 'nearest');
    total_rows = total_rows + height(T);
    if numel(lon) < SEG
        n_sessions_skipped = n_sessions_skipped + 1;
        continue;   % session shorter than one segment (none expected, but be safe)
    end
    n_sessions_used = n_sessions_used + 1;
    lon = lon - mean(lon); lat = lat - mean(lat);   % remove per-session DC offset only
    starts = 1:OV:(numel(lon) - SEG + 1);
    for s = starts
        seg = lon(s:s+SEG-1) .* WIN;
        P = (abs(fft(seg)).^2) / (FS * U);
        P = P(1:nbin); P(2:end-1) = 2 * P(2:end-1);   % one-sided PSD
        sumP_lon = sumP_lon + P; nseg_lon = nseg_lon + 1;

        seg = lat(s:s+SEG-1) .* WIN;
        P = (abs(fft(seg)).^2) / (FS * U);
        P = P(1:nbin); P(2:end-1) = 2 * P(2:end-1);
        sumP_lat = sumP_lat + P; nseg_lat = nseg_lat + 1;
    end
    if mod(k, 10) == 0
        fprintf('  %d/%d sessions processed (%d segments so far)\n', k, numel(files), nseg_lon);
    end
end

Pxx_lon = sumP_lon / nseg_lon;
Pxx_lat = sumP_lat / nseg_lat;

fprintf('\nSessions used: %d, skipped: %d, total rows: %d\n', n_sessions_used, n_sessions_skipped, total_rows);
fprintf('Segments pooled: lon=%d, lat=%d (4096 samples = 163.84 s each, 50%% overlap)\n', nseg_lon, nseg_lat);
fid = fopen(fullfile(OUTDIR, 'n_segments_used.txt'), 'w');
fprintf(fid, 'sessions_used=%d\nsessions_skipped=%d\ntotal_rows=%d\nsegments_lon=%d\nsegments_lat=%d\nseg_len_s=%.2f\noverlap_pct=50\n', ...
    n_sessions_used, n_sessions_skipped, total_rows, nseg_lon, nseg_lat, SEG/FS);
fclose(fid);

% ============================================================ Figure: whole-sample PSD
figure('Visible', 'off', 'Position', [100 100 950 650]);
subplot(2,1,1);
semilogy(f, Pxx_lon, 'k-', 'LineWidth', 1.3); hold on;
xlim([0 12.5]);
xlabel('Frequency (Hz)'); ylabel('PSD (g^2/Hz), log scale');
title(sprintf('lon\\_acc\\_g -- whole-sample Welch PSD (%d sessions, %d segments of 163.8 s, 50%% overlap)', n_sessions_used, nseg_lon));
grid on; box on;
subplot(2,1,2);
semilogy(f, Pxx_lat, 'k-', 'LineWidth', 1.3); hold on;
xlim([0 12.5]);
xlabel('Frequency (Hz)'); ylabel('PSD (g^2/Hz), log scale');
title(sprintf('lat\\_acc\\_g -- whole-sample Welch PSD (%d sessions, %d segments)', n_sessions_used, nseg_lat));
grid on; box on;
save_png(gcf, fullfile(OUTDIR, 'fig_whole_sample_psd.png'));

% ============================================================ Figure: cumulative energy
cum_lon = cumsum(Pxx_lon) / sum(Pxx_lon);
cum_lat = cumsum(Pxx_lat) / sum(Pxx_lat);
figure('Visible', 'off', 'Position', [100 100 900 500]);
plot(f, 100*cum_lon, 'b-', 'LineWidth', 1.6); hold on;
plot(f, 100*cum_lat, 'r-', 'LineWidth', 1.6);
yline(90, 'k:'); yline(95, 'k:');
xlim([0 6]); ylim([0 100]);
xlabel('Frequency (Hz)'); ylabel('Cumulative share of total signal power (%)');
legend({'lon\_acc\_g', 'lat\_acc\_g'}, 'Location', 'southeast');
title('Whole-sample cumulative power spectrum (all 79 sessions)');
grid on; box on;
save_png(gcf, fullfile(OUTDIR, 'fig_cumulative_energy.png'));

% ============================================================ Band-energy table
edges = [0 0.5 1.0 1.5 1.92 2.0 3.0 5.0 12.5];
rows = {};
for i = 1:numel(edges)-1
    m = f >= edges(i) & f < edges(i+1);
    rows(end+1, :) = {sprintf('%.2f - %.2f Hz', edges(i), edges(i+1)), ...
        100*sum(Pxx_lon(m))/sum(Pxx_lon), 100*sum(Pxx_lat(m))/sum(Pxx_lat)}; %#ok<AGROW>
end
T2 = cell2table(rows, 'VariableNames', {'band', 'pct_power_lon_acc_g', 'pct_power_lat_acc_g'});
writetable(T2, fullfile(OUTDIR, 'band_energy_table.csv'));
disp(T2);
fprintf('\npct power below 2.0 Hz cutoff: lon = %.1f%%, lat = %.1f%%\n', ...
    100*sum(Pxx_lon(f < 2.0))/sum(Pxx_lon), 100*sum(Pxx_lat(f < 2.0))/sum(Pxx_lat));
fprintf('pct power below 1.92 Hz (mean(13) first null): lon = %.1f%%, lat = %.1f%%\n', ...
    100*sum(Pxx_lon(f < 1.92))/sum(Pxx_lon), 100*sum(Pxx_lat(f < 1.92))/sum(Pxx_lat));

fprintf('\nSaved 2 figures + band_energy_table.csv + n_segments_used.txt -> %s\n', OUTDIR);
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
