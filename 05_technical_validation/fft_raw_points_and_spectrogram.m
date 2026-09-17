function fft_raw_points_and_spectrogram()
% FFT_RAW_POINTS_AND_SPECTROGRAM  Answers "can't you just draw the FFT points
% as they are?" and "is there another way to show the frequency distribution?"
% (2026-09-16).
%
% Part 1: the literal, unprocessed FFT of a short segment, plotted as
% individual points (no PSD squaring/normalisation, no averaging, no
% smoothing curve) -- genuinely visible as separate dots, unlike the
% whole-sample transform (3.5 million points cannot be shown as distinguishable
% dots in any plot of reasonable size; at that length "the points as they
% are" and "a filled band" are the same picture, because there really are
% hundreds of thousands of points inside any 1 Hz span).
%
% Part 2: a spectrogram -- a genuinely different way to look at frequency
% content. Every figure so far collapses the whole 39-hour recording (or one
% session) into ONE static distribution, which cannot show WHEN the
% higher-frequency content occurs. A spectrogram is a grid of short-time FFTs
% (time on x, frequency on y, colour = magnitude): it shows how the frequency
% content changes minute to minute across a real session, e.g. whether the
% mid-frequency energy that showed up in the whole-sample spectrum concentrates
% around harsh events or is spread through ordinary driving.
%
% Usage: matlab -batch "cd('05_technical_validation'); fft_raw_points_and_spectrogram"
%
% Output: validation_output/frequency_distribution/
%   fig_raw_points_single_segment.png
%   fig_spectrogram_D11_S2.png

close all;
FS = 25;
OUTDIR = fullfile(fileparts(mfilename('fullpath')), 'validation_output', 'frequency_distribution');
if ~exist(OUTDIR, 'dir'); mkdir(OUTDIR); end

RELEASE = fullfile('C:','Users','halha','OneDrive - Durham University','Documents', ...
    'DriveSense_Packages','1_LATEST__labels_v2__release_v3__paper_v6','data', ...
    'Published_Dataset_Final_v3_VIDEO_ALIGNED','Preprocessed_Dataset');
p = fullfile(RELEASE, 'D11', 'Session_2', 'D11_S2_fused.csv');
T = readtable(p, 'ReadVariableNames', true);
lon = fillmissing(T.lon_acc_g, 'linear', 'EndValues', 'nearest');
lat = fillmissing(T.lat_acc_g, 'linear', 'EndValues', 'nearest');
t = T.elapsed_s;

% ============================================================ Part 1: raw points
% One clean 4096-sample segment (163.8 s), starting at 300 s (ordinary
% driving, same stretch used in the earlier figures) -- short enough that
% every one of its 2049 one-sided FFT bins is an individually visible point.
SEG = 4096;
i0 = find(t >= 300, 1);
idx = i0:(i0 + SEG - 1);
x_lon = lon(idx) - mean(lon(idx));
x_lat = lat(idx) - mean(lat(idx));

X_lon = fft(x_lon); X_lat = fft(x_lat);
nbin = SEG/2 + 1;
f = (0:nbin-1)' * (FS / SEG);
mag_lon = abs(X_lon(1:nbin));      % raw magnitude, no squaring, no /(Fs*N), no averaging
mag_lat = abs(X_lat(1:nbin));

figure('Visible', 'off', 'Position', [100 100 1000 750]);
subplot(2,2,1);
plot(f, mag_lon, '.', 'MarkerSize', 6, 'Color', [0.1 0.3 0.8]);
xlabel('Frequency (Hz)'); ylabel('|FFT(lon\_acc\_g)|  (linear, raw)');
title('lon\_acc\_g -- every FFT point of one 163.8 s segment (linear)');
xlim([0 12.5]); grid on; box on;
subplot(2,2,2);
semilogy(f, mag_lon, '.', 'MarkerSize', 6, 'Color', [0.1 0.3 0.8]);
xlabel('Frequency (Hz)'); ylabel('|FFT(lon\_acc\_g)|  (log scale, raw)');
title('Same points, log y-axis (so the small ones are visible too)');
xlim([0 12.5]); grid on; box on;
subplot(2,2,3);
plot(f, mag_lat, '.', 'MarkerSize', 6, 'Color', [0.8 0.2 0.2]);
xlabel('Frequency (Hz)'); ylabel('|FFT(lat\_acc\_g)|  (linear, raw)');
title('lat\_acc\_g -- same segment (linear)');
xlim([0 12.5]); grid on; box on;
subplot(2,2,4);
semilogy(f, mag_lat, '.', 'MarkerSize', 6, 'Color', [0.8 0.2 0.2]);
xlabel('Frequency (Hz)'); ylabel('|FFT(lat\_acc\_g)|  (log scale, raw)');
title('Same points, log y-axis');
xlim([0 12.5]); grid on; box on;
sgtitle('The FFT points exactly as they come out -- no squaring, no normalising, no averaging, no smoothing', 'Interpreter', 'none');
save_png(gcf, fullfile(OUTDIR, 'fig_raw_points_single_segment.png'));

% ============================================================ Part 2: spectrogram
% Short-time FFT: splits the WHOLE session into overlapping short windows and
% stacks their spectra side by side, so frequency content over TIME is
% visible -- something no single "distribution" plot (however it's drawn)
% can show.
win = 256; ov = round(win * 0.9); nfft = 512;
figure('Visible', 'off', 'Position', [100 100 1050 700]);
subplot(2,1,1);
spectrogram(lon - mean(lon), hann(win), ov, nfft, FS, 'yaxis');
ylim([0 6]);
title('D11\_S2 lon\_acc\_g -- spectrogram (time-frequency view, whole session)');
subplot(2,1,2);
spectrogram(lat - mean(lat), hann(win), ov, nfft, FS, 'yaxis');
ylim([0 6]);
title('D11\_S2 lat\_acc\_g -- spectrogram');
save_png(gcf, fullfile(OUTDIR, 'fig_spectrogram_D11_S2.png'));

fprintf('Saved fig_raw_points_single_segment.png + fig_spectrogram_D11_S2.png -> %s\n', OUTDIR);
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
