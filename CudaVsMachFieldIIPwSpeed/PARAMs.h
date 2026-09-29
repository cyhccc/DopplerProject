#ifndef PARAMS_H
#define PARAMS_H
#include<cmath>

class PARAMs
{
public:
	PARAMs();
	~PARAMs();
	void UpdateParamter();
	int seq = 1;
	int datalen = 6144;
	int channels = 128;
	float f0 = 3e6;
	float fs = 30e6;
	float sound_speed = 1540;
	float pitch = 0.3e-3;
	int img_nx = 256;
	int img_nz = 512;

	float x_start = -20e-3;
	float x_end = 20e-3;
	float z_start = 0;
	float z_end = 80e-3;
	float depth = 80e-3;
	float F_number = 1.7;
	float minimum_aperture = 1e-3;
	int filterLength = 129;
	int show_range = 60;
	float cycle = 2;
	float init_angle = -atan(1.0f / 2.0f / F_number);
	float final_angle = atan(1.0f / 2.0f / F_number);
	float angle_step = 0.0f;
	float tx_delay_offset_samples = 0.0f;
	float pixel_dx = 0.0f;
	float pixel_dz = 0.0f;
	bool use_auto_angle_span = true;

	float fs_div_sound_speed;
};
#endif 

