function export_fieldii_rf_frame_major(data_root, pw_values)
%EXPORT_FIELDII_RF_FRAME_MAJOR Export Field II RF as [frame][seq][channel][sample].
%
% The saved frame_*.mat files store rf_data as [sample][channel][seq].
% This exporter aligns RF by per-angle tstart before Hilbert, then writes each
% frame in acquisition-style order: [frame][seq][channel][sample] float32.

if nargin < 1 || isempty(data_root)
    data_root = 'E:\Code\mat\y_branch_fieldii_gpu_pd';
end
if nargin < 2 || isempty(pw_values)
    pw_values = 3:16;
end

channels = 128;
samples = 788;
frames = 128;
fs = single(30.0e6);

for pw = pw_values
    pw_text = sprintf('%02d', pw);
    result_dir = fullfile(data_root, sprintf('results_vertical_pw%s_128f', pw_text));
    sim_dir = fullfile(result_dir, 'simulated_channel_data');
    output_dir = fullfile(result_dir, 'cuda_rf_frame_major_bin');
    if ~exist(sim_dir, 'dir')
        error('Missing simulated channel data directory: %s', sim_dir);
    end
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    first_frame = fullfile(sim_dir, 'frame_001.mat');
    first = load(first_frame, 'tstart');
    reference_samples = min(single(first.tstart(:)) .* fs);

    output_bin = fullfile(output_dir, ...
        sprintf('vertical_pw%s_128f_rf_frame_seq_channel_sample_float.bin', pw_text));
    fid = fopen(output_bin, 'w');
	if fid < 0
		error('Cannot create output RF bin: %s', output_bin);
	end

	for frame_idx = 1:frames
        frame_path = fullfile(sim_dir, sprintf('frame_%03d.mat', frame_idx));
        frame_data = load(frame_path, 'rf_data', 'tstart');
        rf = single(frame_data.rf_data);
        if ~isequal(size(rf), [samples, channels, pw])
            error('Unexpected rf_data size in %s: [%s]', ...
                frame_path, num2str(size(rf)));
        end
        aligned = align_rf_by_tstart_single( ...
            rf, single(frame_data.tstart(:)) .* fs, reference_samples);

        written = fwrite(fid, aligned(:), 'single');
        expected = samples * channels * pw;
        if written ~= expected
			error('Short write for frame %d in %s.', frame_idx, output_bin);
		end
	end
	fclose(fid);

    output_txt = replace(output_bin, '.bin', '.txt');
    write_metadata(output_txt, output_bin, pw, channels, samples, frames, fs, reference_samples);

    expected_bytes = uint64(frames) * uint64(pw) * uint64(channels) * uint64(samples) * 4;
    info = dir(output_bin);
    if uint64(info.bytes) ~= expected_bytes
        error('Unexpected file size for %s. Expected %d, got %d.', ...
            output_bin, expected_bytes, info.bytes);
    end
    fprintf('Exported PW%s: %s (%d bytes), reference_samples=%.15g\n', ...
        pw_text, output_bin, info.bytes, double(reference_samples));
end
end

function aligned = align_rf_by_tstart_single(rf, tstart_samples, reference_samples)
[n_samples, n_channels, n_seq] = size(rf);
aligned = zeros(size(rf), 'single');
sample_axis = single((0:n_samples - 1).');

for seq_idx = 1:n_seq
    delta = single(tstart_samples(seq_idx) - reference_samples);
    source_positions = sample_axis - delta;
    for ch_idx = 1:n_channels
        aligned(:, ch_idx, seq_idx) = interp_linear_zero_single( ...
            rf(:, ch_idx, seq_idx), source_positions);
    end
end
end

function out = interp_linear_zero_single(trace, positions)
n_samples = size(trace, 1);
idx0 = floor(positions);
frac = positions - idx0;
valid = idx0 >= 0 & idx0 < single(n_samples - 1);

out = zeros(size(positions), 'single');
if ~any(valid)
    return;
end

idx0_valid = int32(idx0(valid)) + 1;
idx1_valid = idx0_valid + 1;
s0 = trace(idx0_valid);
s1 = trace(idx1_valid);
frac_valid = single(frac(valid));
out(valid) = s0 + frac_valid .* (s1 - s0);
end

function write_metadata(output_txt, output_bin, pw, channels, samples, frames, fs, reference_samples)
fid = fopen(output_txt, 'w');
	if fid < 0
		error('Cannot create metadata file: %s', output_txt);
	end
	fprintf(fid, 'saved=%s\n', output_bin);
fprintf(fid, 'seq=%d\n', pw);
fprintf(fid, 'channels=%d\n', channels);
fprintf(fid, 'samples=%d\n', samples);
fprintf(fid, 'frames=%d\n', frames);
fprintf(fid, 'fs=%.9g\n', double(fs));
fprintf(fid, 'tstart_ref_samples=%.15g\n', double(reference_samples));
fprintf(fid, 'tstart_alignment=rf_before_hilbert\n');
	fprintf(fid, 'output_format=rf_float32\n');
	fprintf(fid, 'layout=[frame][seq][channel][sample] float32\n');
	fclose(fid);
end
