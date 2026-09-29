function result = run_7883227_matlab_cpu_cuda_matched(varargin)
%RUN_7883227_MATLAB_CPU_CUDA_MATCHED CPU MATLAB reference for the 7883227 CUDA path.
%
% The input binary layout is the CUDA export layout:
%   [seq][channel][sample][frame] complex64, with frame as the fastest axis.
%
% This reference intentionally mirrors the CUDA configuration in main.cpp:
% rectangular receive aperture, linear interpolation, phase rotation based on
% zero-based sample delay, post-beamforming SVD clutter removal, and power sum.

p = inputParser;
p.addParameter('BinDir', fullfile(pwd, 'bins_7883227'), @(x) ischar(x) || isstring(x));
p.addParameter('OutputPrefix', fullfile(pwd, 'outputs', 'power_doppler_7883227_rf001_to_075_matlab_cpu'), @(x) ischar(x) || isstring(x));
p.addParameter('MaxFiles', 75, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('CutoffStart', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('CutoffEnd', 800, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('InterpChunkPixels', 2048, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('DisplayMinDb', -40, @(x) isnumeric(x) && isscalar(x));
p.addParameter('DisplayMaxDb', 0, @(x) isnumeric(x) && isscalar(x));
p.addParameter('UseParallelFiles', false, @(x) islogical(x) || isnumeric(x));
p.addParameter('NumWorkers', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('WorkerNumThreads', 1, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('NumericPrecision', 'double', @(x) ischar(x) || isstring(x));
p.addParameter('BeamformerBackend', 'matlab', @(x) ischar(x) || isstring(x));
p.addParameter('BeamformDumpPath', '', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opt = p.Results;
numeric_precision = lower(string(opt.NumericPrecision));
if numeric_precision ~= "double" && numeric_precision ~= "single"
    error('NumericPrecision must be ''double'' or ''single''.');
end
beamformer_backend = lower(string(opt.BeamformerBackend));
if beamformer_backend ~= "matlab" && beamformer_backend ~= "mex"
    error('BeamformerBackend must be ''matlab'' or ''mex''.');
end

cfg = make_cuda_7883227_config();
cutoff_start = round(opt.CutoffStart);
cutoff_end = round(opt.CutoffEnd);
if cutoff_start ~= opt.CutoffStart || cutoff_end ~= opt.CutoffEnd || ...
        cutoff_start < 1 || cutoff_end < cutoff_start || cutoff_end > cfg.n_frames
    error('SVD cutoff must use integer ranks satisfying 1 <= CutoffStart <= CutoffEnd <= %d.', cfg.n_frames);
end

bin_dir = char(opt.BinDir);
listing = dir(fullfile(bin_dir, 'RF_*_7883227_bs100bw_iq_float2.bin'));
if isempty(listing)
    error('No 7883227 IQ bin files found under %s', bin_dir);
end
[~, order] = sort({listing.name});
listing = listing(order);
listing = listing(1:min(numel(listing), round(opt.MaxFiles)));

output_prefix = char(opt.OutputPrefix);
output_dir = fileparts(output_prefix);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('[Info] MATLAB CPU CUDA-matched 7883227 reference\n');
fprintf('[Info] bin_dir=%s\n', bin_dir);
fprintf('[Info] files=%d, grid=%dx%d, frames=%d, cutoff=[%d,%d]\n', ...
    numel(listing), cfg.nx, cfg.nz, cfg.n_frames, opt.CutoffStart, opt.CutoffEnd);
fprintf('[Info] numeric_precision=%s\n', numeric_precision);
fprintf('[Info] beamformer_backend=%s\n', beamformer_backend);

% Match the CUDA host path: each block is reduced in the selected precision,
% then block power arrays are accumulated in double precision.
power_sum = zeros(cfg.n_pixels, 1, 'double');
timing = struct('read_s', 0, 'beamform_s', 0, 'svd_s', 0, 'power_s', 0);
wall_tic = tic;

[rx_delay, rx_weight, tx_delay] = precompute_delays(cfg);
if beamformer_backend == "mex"
    [phase_cos, phase_sin] = precompute_phase_tables(cfg, rx_delay, tx_delay);
else
    phase_cos = [];
    phase_sin = [];
end
bin_paths = cell(numel(listing), 1);
for file_idx = 1:numel(listing)
    bin_paths{file_idx} = fullfile(listing(file_idx).folder, listing(file_idx).name);
end
interp_chunk_pixels = round(opt.InterpChunkPixels);
worker_num_threads = round(opt.WorkerNumThreads);

if logical(opt.UseParallelFiles)
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= round(opt.NumWorkers)
        if ~isempty(pool)
            delete(pool);
        end
        parpool('local', round(opt.NumWorkers));
    end

    partial_power = cell(numel(listing), 1);
    read_s = zeros(numel(listing), 1);
    beamform_s = zeros(numel(listing), 1);
    svd_s = zeros(numel(listing), 1);
    power_s = zeros(numel(listing), 1);
    progress_queue = parallel.pool.DataQueue;
    afterEach(progress_queue, @(file_idx) fprintf('[Info] completed (%d/%d) %s\n', ...
        file_idx, numel(listing), listing(file_idx).name));

    parfor file_idx = 1:numel(listing)
        beamform_dump_path = first_file_dump_path(opt.BeamformDumpPath, file_idx);
        [partial_power{file_idx}, per_file_timing] = process_one_file_cpu( ...
            bin_paths{file_idx}, cfg, rx_delay, rx_weight, tx_delay, phase_cos, phase_sin, interp_chunk_pixels, cutoff_start, cutoff_end, worker_num_threads, numeric_precision, beamformer_backend, beamform_dump_path);
        read_s(file_idx) = per_file_timing.read_s;
        beamform_s(file_idx) = per_file_timing.beamform_s;
        svd_s(file_idx) = per_file_timing.svd_s;
        power_s(file_idx) = per_file_timing.power_s;
        send(progress_queue, file_idx);
    end

    for file_idx = 1:numel(listing)
        power_sum = power_sum + double(partial_power{file_idx});
    end
    timing.read_s = sum(read_s);
    timing.beamform_s = sum(beamform_s);
    timing.svd_s = sum(svd_s);
    timing.power_s = sum(power_s);
else
    for file_idx = 1:numel(listing)
        fprintf('[Info] (%d/%d) %s\n', file_idx, numel(listing), listing(file_idx).name);
        beamform_dump_path = first_file_dump_path(opt.BeamformDumpPath, file_idx);
        [file_power, per_file_timing] = process_one_file_cpu( ...
            bin_paths{file_idx}, cfg, rx_delay, rx_weight, tx_delay, phase_cos, phase_sin, interp_chunk_pixels, cutoff_start, cutoff_end, 0, numeric_precision, beamformer_backend, beamform_dump_path);
        power_sum = power_sum + double(file_power);
        timing.read_s = timing.read_s + per_file_timing.read_s;
        timing.beamform_s = timing.beamform_s + per_file_timing.beamform_s;
        timing.svd_s = timing.svd_s + per_file_timing.svd_s;
        timing.power_s = timing.power_s + per_file_timing.power_s;
    end
end

timing.wall_s = toc(wall_tic);
power_db = build_power_db_from_sum(power_sum);

power_sum_path = [output_prefix '_power_sum.bin'];
power_db_path = [output_prefix '_power_db.bin'];
png_path = sprintf('%s_db%dto%d_red.png', output_prefix, abs(round(opt.DisplayMinDb)), round(opt.DisplayMaxDb));
summary_path = [output_prefix '.txt'];

write_float_bin(power_sum_path, single(power_sum));
write_float_bin(power_db_path, single(power_db));
save_doppler_png(png_path, power_db, cfg, [opt.DisplayMinDb, opt.DisplayMaxDb], ...
    sprintf('MATLAB CPU SVD Power Doppler from %d RF blocks', numel(listing)));
write_summary(summary_path, bin_dir, listing, cfg, opt, timing, power_sum_path, power_db_path, png_path, class(power_sum));

fprintf('[Info] Saved power sum to %s\n', power_sum_path);
fprintf('[Info] Saved power dB to %s\n', power_db_path);
fprintf('[Info] Saved PNG to %s\n', png_path);
fprintf('[Info] Saved summary to %s\n', summary_path);
fprintf('[Info] block_accumulation_precision=%s\n', class(power_sum));
fprintf('[Info] Done. wall=%.3f s, read=%.3f s, beamform=%.3f s, svd=%.3f s, power=%.3f s\n', ...
    timing.wall_s, timing.read_s, timing.beamform_s, timing.svd_s, timing.power_s);

result = struct();
result.power_sum_path = power_sum_path;
result.power_db_path = power_db_path;
result.png_path = png_path;
result.summary_path = summary_path;
result.timing = timing;
end

function [file_power, timing] = process_one_file_cpu(bin_path, cfg, rx_delay, rx_weight, tx_delay, phase_cos, phase_sin, chunk_pixels, cutoff_start, cutoff_end, worker_num_threads, numeric_precision, beamformer_backend, beamform_dump_path)
if worker_num_threads > 0
    maxNumCompThreads(worker_num_threads);
end
timing = struct('read_s', 0, 'beamform_s', 0, 'svd_s', 0, 'power_s', 0);

read_tic = tic;
if beamformer_backend == "matlab"
    iq_map = map_iq_bin(bin_path, cfg);
else
    iq_map = [];
end
timing.read_s = toc(read_tic);

beamform_tic = tic;
if beamformer_backend == "matlab"
    iq_cache = beamform_iq_cache_cpu(iq_map, cfg, rx_delay, rx_weight, tx_delay, chunk_pixels);
else
    iq_cache = beamform_iq_cache_cpu_mex( ...
        bin_path, rx_delay, rx_weight, tx_delay, cfg.phase_scale, cfg.n_frames, cfg.samples, phase_cos, phase_sin);
end
timing.beamform_s = toc(beamform_tic);

if strlength(beamform_dump_path) > 0
    write_complex_float2_bin(char(beamform_dump_path), iq_cache);
    fprintf('[Info] Saved beamformed IQ cache to %s\n', beamform_dump_path);
end

svd_tic = tic;
iq_filtered = svd_band_filter_cpu(iq_cache, cutoff_start, cutoff_end, numeric_precision);
timing.svd_s = toc(svd_tic);

power_tic = tic;
if numeric_precision == "single"
    file_power = power_from_iq_cache_single(iq_filtered);
else
    file_power = double(sum(abs(iq_filtered).^2, 2));
end
timing.power_s = toc(power_tic);

clear iq_map iq_cache iq_filtered
end

function dump_path = first_file_dump_path(configured_path, file_idx)
if file_idx == 1
    dump_path = string(configured_path);
else
    dump_path = "";
end
end

function cfg = make_cuda_7883227_config()
cfg.seq = 5;
cfg.channels = 128;
cfg.samples = 256;
cfg.n_frames = 800;
cfg.nx = 256;
cfg.nz = 180;
cfg.n_pixels = cfg.nx * cfg.nz;
cfg.f0 = single(15.625e6);
cfg.fs = single(15.625e6);
cfg.sound_speed = single(1540.0);
cfg.pitch = single(0.1e-3);
cfg.f_number = single(1.9);
cfg.minimum_aperture = single(1e-3);

k_wavelength = single(0.09856e-3);
k_origin_x_wl = single(-64.427759740259731);
k_delta_x_wl = single(0.50730519480519476);
k_origin_z_wl = single(18.0);
k_delta_z_wl = single(0.5);
k_start_depth_wl = single(10.0);
k_tw_peak_cycles = single(2.84375);

cfg.x_axis = single((k_origin_x_wl + single(0:cfg.nx - 1) * k_delta_x_wl) * k_wavelength);
cfg.z_axis = single((k_origin_z_wl + single(0:cfg.nz - 1) * k_delta_z_wl) * k_wavelength);
cfg.angles = single(linspace(-6, 6, cfg.seq) * pi / 180);

t0 = single(2.0) * k_start_depth_wl * k_wavelength / cfg.sound_speed - k_tw_peak_cycles / cfg.f0;
cfg.tx_delay_offset_samples = t0 * cfg.fs;
cfg.fs_div_sound_speed = cfg.fs / cfg.sound_speed;
cfg.phase_scale = single(2.0 * pi) * cfg.f0 / cfg.fs;
end

function [rx_delay, rx_weight, tx_delay] = precompute_delays(cfg)
x_vec = repmat(cfg.x_axis(:), cfg.nz, 1);
z_vec = repelem(cfg.z_axis(:), cfg.nx);

elem_x = single((-(cfg.channels - 1) / 2 : (cfg.channels - 1) / 2) * cfg.pitch);
rx_delay = zeros(cfg.n_pixels, cfg.channels, 'single');
rx_weight = false(cfg.n_pixels, cfg.channels);

apod_depth = max(z_vec, cfg.f_number * cfg.minimum_aperture);
aperture_radius = apod_depth / (single(2.0) * cfg.f_number);
for ch = 1:cfg.channels
    rx_dx = x_vec - elem_x(ch);
    rx_weight(:, ch) = abs(rx_dx) <= aperture_radius;
    rx_delay(:, ch) = hypot(rx_dx, z_vec) * cfg.fs_div_sound_speed;
end

tx_delay = zeros(cfg.n_pixels, cfg.seq, 'single');
for s = 1:cfg.seq
    tx_delay(:, s) = ...
        (z_vec * cos(cfg.angles(s)) + x_vec * sin(cfg.angles(s))) * cfg.fs_div_sound_speed ...
        - cfg.tx_delay_offset_samples;
end
end

function [phase_cos, phase_sin] = precompute_phase_tables(cfg, rx_delay, tx_delay)
phase_cos = zeros(cfg.n_pixels, cfg.channels, cfg.seq, 'single');
phase_sin = zeros(cfg.n_pixels, cfg.channels, cfg.seq, 'single');
for s = 1:cfg.seq
    for ch = 1:cfg.channels
        phase_arg = cfg.phase_scale * (tx_delay(:, s) + rx_delay(:, ch));
        phase_cos(:, ch, s) = cos(phase_arg);
        phase_sin(:, ch, s) = sin(phase_arg);
    end
end
end

function iq_map = map_iq_bin(bin_path, cfg)
expected_bytes = 2 * cfg.seq * cfg.channels * cfg.samples * cfg.n_frames * 4;
info = dir(bin_path);
if isempty(info)
    error('Missing input bin: %s', bin_path);
end
if info.bytes ~= expected_bytes
    error('Unexpected file size for %s: expected %d bytes, got %d.', bin_path, expected_bytes, info.bytes);
end
iq_map = memmapfile(bin_path, ...
    'Format', {'single', [2, cfg.n_frames, cfg.samples, cfg.channels, cfg.seq], 'raw'}, ...
    'Writable', false);
end

function iq_cache = beamform_iq_cache_cpu(iq_map, cfg, rx_delay, rx_weight, tx_delay, chunk_pixels)
iq_cache = complex(zeros(cfg.n_pixels, cfg.n_frames, 'single'));
frame_offsets = double((1:cfg.n_frames).');

for s = 1:cfg.seq
    tx_s = tx_delay(:, s);
    for ch = 1:cfg.channels
        raw_trace = iq_map.Data.raw(:, :, :, ch, s);
        trace = complex(squeeze(raw_trace(1, :, :)), squeeze(raw_trace(2, :, :)));
        delay = tx_s + rx_delay(:, ch);
        valid = rx_weight(:, ch) & delay >= 0 & delay < single(cfg.samples - 1);
        valid_idx = find(valid);

        for start_idx = 1:chunk_pixels:numel(valid_idx)
            stop_idx = min(start_idx + chunk_pixels - 1, numel(valid_idx));
            pix = valid_idx(start_idx:stop_idx);
            d = delay(pix);
            i0 = floor(d);
            frac = d - i0;

            lin0 = frame_offsets + double(i0(:).') * cfg.n_frames;
            s0 = trace(lin0);
            s1 = trace(lin0 + cfg.n_frames);
            delayed = s0 + (s1 - s0) .* frac(:).';

            phase_arg = cfg.phase_scale * d;
            phase = complex(cos(phase_arg), sin(phase_arg));
            iq_cache(pix, :) = iq_cache(pix, :) + delayed.' .* phase(:);
        end
    end
end
end

function iq_filtered = svd_band_filter_cpu(iq_cache, cutoff_start, cutoff_end, numeric_precision)
n_frames = size(iq_cache, 2);
clutter_count = max(round(cutoff_start) - 1, 0);
if clutter_count <= 0 && cutoff_end == n_frames
    iq_filtered = iq_cache;
    return
end

if numeric_precision == "single"
    gram = iq_cache' * iq_cache;
    gram = tril(gram) + tril(gram, -1)';
else
    gram = double(iq_cache') * double(iq_cache);
    gram = (gram + gram') * 0.5;
end
[v, d] = eig(gram, 'vector');
[~, order] = sort(real(d), 'descend');
if cutoff_end == n_frames
    v_clutter = single(v(:, order(1:clutter_count)));
    iq_filtered = iq_cache - (iq_cache * v_clutter) * v_clutter';
else
    v_keep = single(v(:, order(cutoff_start:cutoff_end)));
    iq_filtered = (iq_cache * v_keep) * v_keep';
end
end

function file_power = power_from_iq_cache_single(iq_cache)
file_power = zeros(size(iq_cache, 1), 1, 'single');
for frame_idx = 1:size(iq_cache, 2)
    iq = iq_cache(:, frame_idx);
    file_power = file_power + real(iq) .* real(iq) + imag(iq) .* imag(iq);
end
end

function power_db = build_power_db_from_sum(power_sum)
peak = max(sqrt(max(power_sum, 0)));
power_db = -200 * ones(size(power_sum), 'single');
if peak <= 0
    return
end
magnitude = sqrt(max(power_sum, 0)) / peak;
power_db = single(20 * log10(max(single(magnitude), single(1.0e-20))));
end

function write_float_bin(path_name, data)
fid = fopen(path_name, 'wb');
if fid < 0
    error('Cannot open output file: %s', path_name);
end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, data(:), 'single');
clear cleanup
end

function write_complex_float2_bin(path_name, data)
fid = fopen(path_name, 'wb');
if fid < 0
    error('Cannot open complex output file: %s', path_name);
end
cleanup = onCleanup(@() fclose(fid));
frames_per_chunk = 16;
for first_frame = 1:frames_per_chunk:size(data, 2)
    last_frame = min(first_frame + frames_per_chunk - 1, size(data, 2));
    chunk = data(:, first_frame:last_frame);
    float2 = zeros(2, numel(chunk), 'single');
    float2(1, :) = real(chunk(:));
    float2(2, :) = imag(chunk(:));
    fwrite(fid, float2, 'single');
end
clear cleanup
end

function save_doppler_png(path_name, power_db, cfg, display_range_db, title_text)
img = reshape(power_db, [cfg.nx, cfg.nz]).';
cmap = doppler_redyellow_colormap(256);

fig = figure('Visible', 'off', 'Color', 'w');
imagesc(double(cfg.x_axis) * 1e3, double(cfg.z_axis) * 1e3, img, display_range_db);
axis image;
set(gca, 'YDir', 'reverse');
colormap(gca, cmap);
c = colorbar;
c.Label.String = 'dB';
xlabel('Lateral (mm)');
ylabel('Depth (mm)');
title(title_text, 'Interpreter', 'none');
exportgraphics(fig, path_name, 'Resolution', 180);
close(fig);
end

function cmap = doppler_redyellow_colormap(n)
stops = [
    0.00, 0.000, 0.000, 0.000
    0.18, 0.188, 0.000, 0.000
    0.40, 0.545, 0.000, 0.000
    0.62, 1.000, 0.165, 0.000
    0.82, 1.000, 0.749, 0.000
    0.94, 1.000, 0.941, 0.290
    1.00, 1.000, 0.996, 0.949
];
x = linspace(0, 1, n).';
cmap = zeros(n, 3);
for k = 1:3
    cmap(:, k) = interp1(stops(:, 1), stops(:, k + 1), x, 'linear');
end
end

function write_summary(path_name, bin_dir, listing, cfg, opt, timing, power_sum_path, power_db_path, png_path, block_accumulation_precision)
fid = fopen(path_name, 'w');
if fid < 0
    error('Cannot create summary file: %s', path_name);
end
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, 'mode=matlab_cpu_cuda_matched\n');
fprintf(fid, 'numeric_precision=%s\n', lower(string(opt.NumericPrecision)));
fprintf(fid, 'beamform_precision=single\n');
fprintf(fid, 'beamformer_backend=%s\n', lower(string(opt.BeamformerBackend)));
if strlength(string(opt.BeamformDumpPath)) > 0
    fprintf(fid, 'beamform_dump_path=%s\n', string(opt.BeamformDumpPath));
end
fprintf(fid, 'gram_eig_precision=%s\n', lower(string(opt.NumericPrecision)));
fprintf(fid, 'projection_precision=single\n');
fprintf(fid, 'block_power_precision=%s\n', lower(string(opt.NumericPrecision)));
fprintf(fid, 'block_accumulation_precision=%s\n', block_accumulation_precision);
fprintf(fid, 'output_power_sum_precision=single\n');
fprintf(fid, 'svd_method=covariance_exact\n');
fprintf(fid, 'bin_dir=%s\n', bin_dir);
fprintf(fid, 'processed_files=%d\n', numel(listing));
fprintf(fid, 'seq=%d\n', cfg.seq);
fprintf(fid, 'channels=%d\n', cfg.channels);
fprintf(fid, 'samples=%d\n', cfg.samples);
fprintf(fid, 'frames=%d\n', cfg.n_frames);
fprintf(fid, 'grid=%dx%d\n', cfg.nx, cfg.nz);
fprintf(fid, 'cutoff_start=%d\n', round(opt.CutoffStart));
fprintf(fid, 'cutoff_end=%d\n', round(opt.CutoffEnd));
fprintf(fid, 'display_range_db=%.6g,%.6g\n', opt.DisplayMinDb, opt.DisplayMaxDb);
fprintf(fid, 'tx_delay_offset_samples=%.9g\n', cfg.tx_delay_offset_samples);
fprintf(fid, 'read_total_s=%.6f\n', timing.read_s);
fprintf(fid, 'beamform_total_s=%.6f\n', timing.beamform_s);
fprintf(fid, 'svd_total_s=%.6f\n', timing.svd_s);
fprintf(fid, 'power_total_s=%.6f\n', timing.power_s);
fprintf(fid, 'wall_total_s=%.6f\n', timing.wall_s);
fprintf(fid, 'power_sum_bin=%s\n', power_sum_path);
fprintf(fid, 'power_db_bin=%s\n', power_db_path);
fprintf(fid, 'png=%s\n', png_path);
for i = 1:numel(listing)
    fprintf(fid, '%s\n', listing(i).name);
end

clear cleanup
end
