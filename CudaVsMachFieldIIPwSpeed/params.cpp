#include "PARAMs.h"


PARAMs::PARAMs()
{
	UpdateParamter();
}

void PARAMs::UpdateParamter()
{
	if (use_auto_angle_span)
	{
		init_angle = -atan(1.0f / 2.0f / F_number);
		final_angle = atan(1.0f / 2.0f / F_number);
	}
	angle_step = (seq > 1) ? (final_angle - init_angle) / static_cast<float>(seq - 1) : 0.0f;
	pixel_dx = (img_nx > 1) ? (x_end - x_start) / static_cast<float>(img_nx - 1) : 0.0f;
	pixel_dz = (img_nz > 1) ? (z_end - z_start) / static_cast<float>(img_nz - 1) : 0.0f;
	fs_div_sound_speed = fs / sound_speed;
}
PARAMs::~PARAMs()
{

}
