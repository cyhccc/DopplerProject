function report = validate_beamform_iq_cache_cpu_mex(varargin)
%VALIDATE_BEAMFORM_IQ_CACHE_CPU_MEX Validate MEX IQ against MATLAB arithmetic.
% This is an actual-data numerical check, not a unit test. By default it
% recomputes a deterministic set of complete pixel time series in MATLAB.

p = inputParser;
p.addParameter('BinPath', '', @(x) ischar(x) || isstring(x));
p.addParameter('ReferenceIqPath', '', @(x) ischar(x) || isstring(x));
p.addParameter('SamplePixelCount', 64, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('MaxAbsTolerance', 1e-1, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('RelativeL2Tolerance', 2e-6, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('NormalizedMaxTolerance', 5e-7, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.parse(varargin{:});
opt = p.Results;

project_dir = fileparts(mfilename('fullpath'));
if isempty(opt.BinPath)
    listing = dir(fullfile(project_dir, 'bins_7883227', 'RF_*_7883227_bs100bw_iq_float2.bin'));
    if isempty(listing)
        listing = dir(fullfile('E:\Code\Cuda\LinearArrayCPWC_OPT_test\bins_7883227', ...
            'RF_*_7883227_bs100bw_iq_float2.bin'));
    end
    assert(~isempty(listing), 'No actual 7883227 IQ input was found. Pass BinPath explicitly.');
    [~, order] = sort({listing.name});
    listing = listing(order);
    bin_path = fullfile(listing(1).folder, listing(1).name);
else
    bin_path = char(opt.BinPath);
end

cfg = make_config();
[rx_delay, rx_weight, tx_delay] = precompute_delays(cfg);
[phase_cos, phase_sin] = precompute_phase(cfg, rx_delay, tx_delay);
assert(exist('beamform_iq_cache_cpu_mex', 'file') == 3, ...
    'MEX is missing. Run build_beamform_iq_cache_cpu_mex first.');

fprintf('[Info] Running MEX on actual input: %s\n', bin_path);
mex_tic = tic;
iq_mex = beamform_iq_cache_cpu_mex(bin_path, rx_delay, rx_weight, tx_delay, ...
    cfg.phase_scale, cfg.n_frames, cfg.samples, phase_cos, phase_sin);
mex_seconds = toc(mex_tic);

sample_count = min(round(opt.SamplePixelCount), cfg.n_pixels);
sample_pixels = unique(round(linspace(1, cfg.n_pixels, sample_count)));
matlab_tic = tic;
iq_matlab_sample = beamform_sample_pixels_matlab(bin_path, sample_pixels, cfg, ...
    rx_delay, rx_weight, tx_delay);
matlab_sample_seconds = toc(matlab_tic);

sample_diff = iq_mex(sample_pixels, :) - iq_matlab_sample;
sample_max_abs = double(max(abs(sample_diff), [], 'all'));
sample_relative_l2 = double(norm(sample_diff(:))) / max(double(norm(iq_matlab_sample(:))), eps);
sample_signal_max = double(max(abs(iq_matlab_sample), [], 'all'));
sample_normalized_max = sample_max_abs / max(sample_signal_max, eps);

reference_max_abs = NaN;
reference_relative_l2 = NaN;
reference_signal_max = NaN;
reference_normalized_max = NaN;
if strlength(string(opt.ReferenceIqPath)) > 0
    iq_reference = read_interleaved_complex_single(char(opt.ReferenceIqPath), cfg.n_pixels, cfg.n_frames);
    reference_diff = iq_mex - iq_reference;
    reference_max_abs = double(max(abs(reference_diff), [], 'all'));
    reference_relative_l2 = double(norm(reference_diff(:))) / max(double(norm(iq_reference(:))), eps);
    reference_signal_max = double(max(abs(iq_reference), [], 'all'));
    reference_normalized_max = reference_max_abs / max(reference_signal_max, eps);
    clear iq_reference reference_diff
end

fprintf('[Result] mex_seconds=%.3f, matlab_sample_seconds=%.3f\n', mex_seconds, matlab_sample_seconds);
fprintf('[Result] sampled_pixels=%d, signal_max=%.9g, max_abs=%.9g, normalized_max=%.9g, relative_l2=%.9g\n', ...
    numel(sample_pixels), sample_signal_max, sample_max_abs, sample_normalized_max, sample_relative_l2);
if ~isnan(reference_max_abs)
    fprintf('[Result] full_reference signal_max=%.9g, max_abs=%.9g, normalized_max=%.9g, relative_l2=%.9g\n', ...
        reference_signal_max, reference_max_abs, reference_normalized_max, reference_relative_l2);
end

assert(sample_max_abs <= opt.MaxAbsTolerance, ...
    'Sampled MATLAB IQ max absolute error %.9g exceeds tolerance %.9g.', sample_max_abs, opt.MaxAbsTolerance);
assert(sample_relative_l2 <= opt.RelativeL2Tolerance, ...
    'Sampled MATLAB IQ relative L2 error %.9g exceeds tolerance %.9g.', sample_relative_l2, opt.RelativeL2Tolerance);
assert(sample_normalized_max <= opt.NormalizedMaxTolerance, ...
    'Sampled MATLAB IQ normalized max error %.9g exceeds tolerance %.9g.', sample_normalized_max, opt.NormalizedMaxTolerance);
if ~isnan(reference_max_abs)
    assert(reference_max_abs <= opt.MaxAbsTolerance, ...
        'Full MATLAB IQ max absolute error %.9g exceeds tolerance %.9g.', reference_max_abs, opt.MaxAbsTolerance);
    assert(reference_relative_l2 <= opt.RelativeL2Tolerance, ...
        'Full MATLAB IQ relative L2 error %.9g exceeds tolerance %.9g.', reference_relative_l2, opt.RelativeL2Tolerance);
    assert(reference_normalized_max <= opt.NormalizedMaxTolerance, ...
        'Full MATLAB IQ normalized max error %.9g exceeds tolerance %.9g.', reference_normalized_max, opt.NormalizedMaxTolerance);
end

report = struct('bin_path', bin_path, 'mex_seconds', mex_seconds, ...
    'matlab_sample_seconds', matlab_sample_seconds, 'sample_pixels', sample_pixels, ...
    'sample_signal_max', sample_signal_max, 'sample_max_abs', sample_max_abs, ...
    'sample_normalized_max', sample_normalized_max, 'sample_relative_l2', sample_relative_l2, ...
    'reference_signal_max', reference_signal_max, 'reference_max_abs', reference_max_abs, ...
    'reference_normalized_max', reference_normalized_max, 'reference_relative_l2', reference_relative_l2);
end

function [phase_cos, phase_sin] = precompute_phase(cfg, rx_delay, tx_delay)
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

function iq = beamform_sample_pixels_matlab(bin_path, pixels, cfg, rx_delay, rx_weight, tx_delay)
iq_map = memmapfile(bin_path, ...
    'Format', {'single', [2, cfg.n_frames, cfg.samples, cfg.channels, cfg.seq], 'raw'}, ...
    'Writable', false);
iq = complex(zeros(numel(pixels), cfg.n_frames, 'single'));
frame_offsets = double((1:cfg.n_frames).');
for s = 1:cfg.seq
    for ch = 1:cfg.channels
        delay = tx_delay(pixels, s) + rx_delay(pixels, ch);
        valid = rx_weight(pixels, ch) & delay >= 0 & delay < single(cfg.samples - 1);
        if ~any(valid), continue; end
        selected = find(valid);
        d = delay(selected);
        i0 = floor(d);
        frac = d - i0;
        raw_trace = iq_map.Data.raw(:, :, :, ch, s);
        trace = complex(squeeze(raw_trace(1, :, :)), squeeze(raw_trace(2, :, :)));
        lin0 = frame_offsets + double(i0(:).') * cfg.n_frames;
        s0 = trace(lin0);
        s1 = trace(lin0 + cfg.n_frames);
        delayed = s0 + (s1 - s0) .* frac(:).';
        phase_arg = cfg.phase_scale * d;
        phase = complex(cos(phase_arg), sin(phase_arg));
        iq(selected, :) = iq(selected, :) + delayed.' .* phase(:);
    end
end
end

function iq = read_interleaved_complex_single(path, n_pixels, n_frames)
fid = fopen(path, 'rb');
assert(fid >= 0, 'Cannot open ReferenceIqPath: %s', path);
cleanup = onCleanup(@() fclose(fid));
raw = fread(fid, [2, n_pixels * n_frames], '*single');
assert(numel(raw) == 2 * n_pixels * n_frames, 'Reference IQ file has an unexpected size.');
iq = reshape(complex(raw(1, :), raw(2, :)), n_pixels, n_frames);
end

function cfg = make_config()
cfg.seq = 5; cfg.channels = 128; cfg.samples = 256; cfg.n_frames = 800;
cfg.nx = 256; cfg.nz = 180; cfg.n_pixels = cfg.nx * cfg.nz;
cfg.f0 = single(15.625e6); cfg.fs = single(15.625e6); cfg.sound_speed = single(1540.0);
cfg.pitch = single(0.1e-3); cfg.f_number = single(1.9); cfg.minimum_aperture = single(1e-3);
k_wavelength = single(0.09856e-3);
cfg.x_axis = single((single(-64.427759740259731) + single(0:cfg.nx - 1) * single(0.50730519480519476)) * k_wavelength);
cfg.z_axis = single((single(18.0) + single(0:cfg.nz - 1) * single(0.5)) * k_wavelength);
cfg.angles = single(linspace(-6, 6, cfg.seq) * pi / 180);
t0 = single(2.0) * single(10.0) * k_wavelength / cfg.sound_speed - single(2.84375) / cfg.f0;
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
    tx_delay(:, s) = (z_vec * cos(cfg.angles(s)) + x_vec * sin(cfg.angles(s))) * ...
        cfg.fs_div_sound_speed - cfg.tx_delay_offset_samples;
end
end
