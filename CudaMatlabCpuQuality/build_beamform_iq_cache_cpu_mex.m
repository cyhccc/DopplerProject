function output_path = build_beamform_iq_cache_cpu_mex()
%BUILD_BEAMFORM_IQ_CACHE_CPU_MEX Build the optional pure-CPU OpenMP beamformer.

project_dir = fileparts(mfilename('fullpath'));
source_path = fullfile(project_dir, 'beamform_iq_cache_cpu_mex.cpp');

old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir));
cd(project_dir);

mex('-R2018a', ...
    'COMPFLAGS=$COMPFLAGS /O2 /openmp /fp:precise /std:c++17', ...
    'LINKFLAGS=$LINKFLAGS /LTCG', ...
    source_path, ...
    '-output', 'beamform_iq_cache_cpu_mex');

output_path = fullfile(project_dir, ['beamform_iq_cache_cpu_mex.' mexext]);
fprintf('[Info] Built %s\n', output_path);
end
