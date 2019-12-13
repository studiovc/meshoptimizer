// This file is part of gltfpack; see gltfpack.h for version/license details
#include "gltfpack.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

#include <algorithm>

#include "../src/meshoptimizer.h"

static bool getAttributeBounds(const std::vector<Mesh>& meshes, cgltf_attribute_type type, Attr& min, Attr& max)
{
	min.f[0] = min.f[1] = min.f[2] = min.f[3] = +FLT_MAX;
	max.f[0] = max.f[1] = max.f[2] = max.f[3] = -FLT_MAX;

	Attr pad = {};

	bool valid = false;

	for (size_t i = 0; i < meshes.size(); ++i)
	{
		const Mesh& mesh = meshes[i];

		for (size_t j = 0; j < mesh.streams.size(); ++j)
		{
			const Stream& s = mesh.streams[j];

			if (s.type == type)
			{
				if (s.target == 0)
				{
					for (size_t k = 0; k < s.data.size(); ++k)
					{
						const Attr& a = s.data[k];

						min.f[0] = std::min(min.f[0], a.f[0]);
						min.f[1] = std::min(min.f[1], a.f[1]);
						min.f[2] = std::min(min.f[2], a.f[2]);
						min.f[3] = std::min(min.f[3], a.f[3]);

						max.f[0] = std::max(max.f[0], a.f[0]);
						max.f[1] = std::max(max.f[1], a.f[1]);
						max.f[2] = std::max(max.f[2], a.f[2]);
						max.f[3] = std::max(max.f[3], a.f[3]);

						valid = true;
					}
				}
				else
				{
					for (size_t k = 0; k < s.data.size(); ++k)
					{
						const Attr& a = s.data[k];

						pad.f[0] = std::max(pad.f[0], fabsf(a.f[0]));
						pad.f[1] = std::max(pad.f[1], fabsf(a.f[1]));
						pad.f[2] = std::max(pad.f[2], fabsf(a.f[2]));
						pad.f[3] = std::max(pad.f[3], fabsf(a.f[3]));
					}
				}
			}
		}
	}

	if (valid)
	{
		for (int k = 0; k < 4; ++k)
		{
			min.f[k] -= pad.f[k];
			max.f[k] += pad.f[k];
		}
	}

	return valid;
}

QuantizationParams prepareQuantization(const std::vector<Mesh>& meshes, const Settings& settings)
{
	QuantizationParams result = {};

	result.pos_bits = settings.pos_bits;

	Attr pos_min, pos_max;
	if (getAttributeBounds(meshes, cgltf_attribute_type_position, pos_min, pos_max))
	{
		result.pos_offset[0] = pos_min.f[0];
		result.pos_offset[1] = pos_min.f[1];
		result.pos_offset[2] = pos_min.f[2];
		result.pos_scale = std::max(pos_max.f[0] - pos_min.f[0], std::max(pos_max.f[1] - pos_min.f[1], pos_max.f[2] - pos_min.f[2]));
	}

	result.uv_bits = settings.tex_bits;

	Attr uv_min, uv_max;
	if (getAttributeBounds(meshes, cgltf_attribute_type_texcoord, uv_min, uv_max))
	{
		result.uv_offset[0] = uv_min.f[0];
		result.uv_offset[1] = uv_min.f[1];
		result.uv_scale[0] = uv_max.f[0] - uv_min.f[0];
		result.uv_scale[1] = uv_max.f[1] - uv_min.f[1];
	}

	return result;
}

static void rescaleNormal(float& nx, float& ny, float& nz)
{
	// scale the normal to make sure the largest component is +-1.0
	// this reduces the entropy of the normal by ~1.5 bits without losing precision
	// it's better to use octahedral encoding but that requires special shader support
	float nm = std::max(fabsf(nx), std::max(fabsf(ny), fabsf(nz)));
	float ns = nm == 0.f ? 0.f : 1 / nm;

	nx *= ns;
	ny *= ns;
	nz *= ns;
}

static void renormalizeWeights(uint8_t (&w)[4])
{
	int sum = w[0] + w[1] + w[2] + w[3];

	if (sum == 255)
		return;

	// we assume that the total error is limited to 0.5/component = 2
	// this means that it's acceptable to adjust the max. component to compensate for the error
	int max = 0;

	for (int k = 1; k < 4; ++k)
		if (w[k] > w[max])
			max = k;

	w[max] += uint8_t(255 - sum);
}

StreamFormat writeVertexStream(std::string& bin, const Stream& stream, const QuantizationParams& params, const Settings& settings, bool has_targets)
{
	if (stream.type == cgltf_attribute_type_position)
	{
		if (stream.target == 0)
		{
			float pos_rscale = params.pos_scale == 0.f ? 0.f : 1.f / params.pos_scale;

			for (size_t i = 0; i < stream.data.size(); ++i)
			{
				const Attr& a = stream.data[i];

				uint16_t v[4] = {
				    uint16_t(meshopt_quantizeUnorm((a.f[0] - params.pos_offset[0]) * pos_rscale, params.pos_bits)),
				    uint16_t(meshopt_quantizeUnorm((a.f[1] - params.pos_offset[1]) * pos_rscale, params.pos_bits)),
				    uint16_t(meshopt_quantizeUnorm((a.f[2] - params.pos_offset[2]) * pos_rscale, params.pos_bits)),
				    0};
				bin.append(reinterpret_cast<const char*>(v), sizeof(v));
			}

			StreamFormat format = {cgltf_type_vec3, cgltf_component_type_r_16u, false, 8};
			return format;
		}
		else
		{
			float pos_rscale = params.pos_scale == 0.f ? 0.f : 1.f / params.pos_scale;

			int maxv = 0;

			for (size_t i = 0; i < stream.data.size(); ++i)
			{
				const Attr& a = stream.data[i];

				maxv = std::max(maxv, meshopt_quantizeUnorm(fabsf(a.f[0]) * pos_rscale, params.pos_bits));
				maxv = std::max(maxv, meshopt_quantizeUnorm(fabsf(a.f[1]) * pos_rscale, params.pos_bits));
				maxv = std::max(maxv, meshopt_quantizeUnorm(fabsf(a.f[2]) * pos_rscale, params.pos_bits));
			}

			if (maxv <= 127)
			{
				for (size_t i = 0; i < stream.data.size(); ++i)
				{
					const Attr& a = stream.data[i];

					int8_t v[4] = {
					    int8_t((a.f[0] >= 0.f ? 1 : -1) * meshopt_quantizeUnorm(fabsf(a.f[0]) * pos_rscale, params.pos_bits)),
					    int8_t((a.f[1] >= 0.f ? 1 : -1) * meshopt_quantizeUnorm(fabsf(a.f[1]) * pos_rscale, params.pos_bits)),
					    int8_t((a.f[2] >= 0.f ? 1 : -1) * meshopt_quantizeUnorm(fabsf(a.f[2]) * pos_rscale, params.pos_bits)),
					    0};
					bin.append(reinterpret_cast<const char*>(v), sizeof(v));
				}

				StreamFormat format = {cgltf_type_vec3, cgltf_component_type_r_8, false, 4};
				return format;
			}
			else
			{
				for (size_t i = 0; i < stream.data.size(); ++i)
				{
					const Attr& a = stream.data[i];

					int16_t v[4] = {
					    int16_t((a.f[0] >= 0.f ? 1 : -1) * meshopt_quantizeUnorm(fabsf(a.f[0]) * pos_rscale, params.pos_bits)),
					    int16_t((a.f[1] >= 0.f ? 1 : -1) * meshopt_quantizeUnorm(fabsf(a.f[1]) * pos_rscale, params.pos_bits)),
					    int16_t((a.f[2] >= 0.f ? 1 : -1) * meshopt_quantizeUnorm(fabsf(a.f[2]) * pos_rscale, params.pos_bits)),
					    0};
					bin.append(reinterpret_cast<const char*>(v), sizeof(v));
				}

				StreamFormat format = {cgltf_type_vec3, cgltf_component_type_r_16, false, 8};
				return format;
			}
		}
	}
	else if (stream.type == cgltf_attribute_type_texcoord)
	{
		float uv_rscale[2] = {
		    params.uv_scale[0] == 0.f ? 0.f : 1.f / params.uv_scale[0],
		    params.uv_scale[1] == 0.f ? 0.f : 1.f / params.uv_scale[1],
		};

		for (size_t i = 0; i < stream.data.size(); ++i)
		{
			const Attr& a = stream.data[i];

			uint16_t v[2] = {
			    uint16_t(meshopt_quantizeUnorm((a.f[0] - params.uv_offset[0]) * uv_rscale[0], params.uv_bits)),
			    uint16_t(meshopt_quantizeUnorm((a.f[1] - params.uv_offset[1]) * uv_rscale[1], params.uv_bits)),
			};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_vec2, cgltf_component_type_r_16u, false, 4};
		return format;
	}
	else if (stream.type == cgltf_attribute_type_normal)
	{
		bool unnormalized = settings.nrm_unnormalized && !has_targets;
		int bits = unnormalized ? settings.nrm_bits : (settings.nrm_bits > 8 ? 16 : 8);

		for (size_t i = 0; i < stream.data.size(); ++i)
		{
			const Attr& a = stream.data[i];

			float nx = a.f[0], ny = a.f[1], nz = a.f[2];

			if (unnormalized)
				rescaleNormal(nx, ny, nz);

			if (bits > 8)
			{
				int16_t v[4] = {
				    int16_t(meshopt_quantizeSnorm(nx, bits)),
				    int16_t(meshopt_quantizeSnorm(ny, bits)),
				    int16_t(meshopt_quantizeSnorm(nz, bits)),
				    0};
				bin.append(reinterpret_cast<const char*>(v), sizeof(v));
			}
			else
			{
				int8_t v[4] = {
				    int8_t(meshopt_quantizeSnorm(nx, bits)),
				    int8_t(meshopt_quantizeSnorm(ny, bits)),
				    int8_t(meshopt_quantizeSnorm(nz, bits)),
				    0};
				bin.append(reinterpret_cast<const char*>(v), sizeof(v));
			}
		}

		if (bits > 8)
		{
			StreamFormat format = {cgltf_type_vec3, cgltf_component_type_r_16, true, 8};
			return format;
		}
		else
		{
			StreamFormat format = {cgltf_type_vec3, cgltf_component_type_r_8, true, 4};
			return format;
		}
	}
	else if (stream.type == cgltf_attribute_type_tangent)
	{
		bool unnormalized = settings.nrm_unnormalized && !has_targets;
		int bits = unnormalized ? settings.nrm_bits : (settings.nrm_bits > 8 ? 16 : 8);

		for (size_t i = 0; i < stream.data.size(); ++i)
		{
			const Attr& a = stream.data[i];

			float nx = a.f[0], ny = a.f[1], nz = a.f[2], nw = a.f[3];

			if (unnormalized)
				rescaleNormal(nx, ny, nz);

			if (bits > 8)
			{
				int16_t v[4] = {
				    int16_t(meshopt_quantizeSnorm(nx, bits)),
				    int16_t(meshopt_quantizeSnorm(ny, bits)),
				    int16_t(meshopt_quantizeSnorm(nz, bits)),
				    int16_t(meshopt_quantizeSnorm(nw, 8))};
				bin.append(reinterpret_cast<const char*>(v), sizeof(v));
			}
			else
			{
				int8_t v[4] = {
				    int8_t(meshopt_quantizeSnorm(nx, bits)),
				    int8_t(meshopt_quantizeSnorm(ny, bits)),
				    int8_t(meshopt_quantizeSnorm(nz, bits)),
				    int8_t(meshopt_quantizeSnorm(nw, 8))};
				bin.append(reinterpret_cast<const char*>(v), sizeof(v));
			}
		}

		cgltf_type type = (stream.target == 0) ? cgltf_type_vec4 : cgltf_type_vec3;

		if (bits > 8)
		{
			StreamFormat format = {type, cgltf_component_type_r_16, true, 8};
			return format;
		}
		else
		{
			StreamFormat format = {type, cgltf_component_type_r_8, true, 4};
			return format;
		}
	}
	else if (stream.type == cgltf_attribute_type_color)
	{
		for (size_t i = 0; i < stream.data.size(); ++i)
		{
			const Attr& a = stream.data[i];

			uint8_t v[4] = {
			    uint8_t(meshopt_quantizeUnorm(a.f[0], 8)),
			    uint8_t(meshopt_quantizeUnorm(a.f[1], 8)),
			    uint8_t(meshopt_quantizeUnorm(a.f[2], 8)),
			    uint8_t(meshopt_quantizeUnorm(a.f[3], 8))};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_vec4, cgltf_component_type_r_8u, true, 4};
		return format;
	}
	else if (stream.type == cgltf_attribute_type_weights)
	{
		for (size_t i = 0; i < stream.data.size(); ++i)
		{
			const Attr& a = stream.data[i];

			float ws = a.f[0] + a.f[1] + a.f[2] + a.f[3];
			float wsi = (ws == 0.f) ? 0.f : 1.f / ws;

			uint8_t v[4] = {
			    uint8_t(meshopt_quantizeUnorm(a.f[0] * wsi, 8)),
			    uint8_t(meshopt_quantizeUnorm(a.f[1] * wsi, 8)),
			    uint8_t(meshopt_quantizeUnorm(a.f[2] * wsi, 8)),
			    uint8_t(meshopt_quantizeUnorm(a.f[3] * wsi, 8))};

			if (wsi != 0.f)
				renormalizeWeights(v);

			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_vec4, cgltf_component_type_r_8u, true, 4};
		return format;
	}
	else if (stream.type == cgltf_attribute_type_joints)
	{
		unsigned int maxj = 0;

		for (size_t i = 0; i < stream.data.size(); ++i)
			maxj = std::max(maxj, unsigned(stream.data[i].f[0]));

		assert(maxj <= 65535);

		if (maxj <= 255)
		{
			for (size_t i = 0; i < stream.data.size(); ++i)
			{
				const Attr& a = stream.data[i];

				uint8_t v[4] = {
				    uint8_t(a.f[0]),
				    uint8_t(a.f[1]),
				    uint8_t(a.f[2]),
				    uint8_t(a.f[3])};
				bin.append(reinterpret_cast<const char*>(v), sizeof(v));
			}

			StreamFormat format = {cgltf_type_vec4, cgltf_component_type_r_8u, false, 4};
			return format;
		}
		else
		{
			for (size_t i = 0; i < stream.data.size(); ++i)
			{
				const Attr& a = stream.data[i];

				uint16_t v[4] = {
				    uint16_t(a.f[0]),
				    uint16_t(a.f[1]),
				    uint16_t(a.f[2]),
				    uint16_t(a.f[3])};
				bin.append(reinterpret_cast<const char*>(v), sizeof(v));
			}

			StreamFormat format = {cgltf_type_vec4, cgltf_component_type_r_16u, false, 8};
			return format;
		}
	}
	else
	{
		for (size_t i = 0; i < stream.data.size(); ++i)
		{
			const Attr& a = stream.data[i];

			float v[4] = {a.f[0], a.f[1], a.f[2], a.f[3]};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_vec4, cgltf_component_type_r_32f, false, 16};
		return format;
	}
}

void getPositionBounds(int min[3], int max[3], const Stream& stream, const QuantizationParams& params)
{
	assert(stream.type == cgltf_attribute_type_position);
	assert(stream.data.size() > 0);

	min[0] = min[1] = min[2] = INT_MAX;
	max[0] = max[1] = max[2] = INT_MIN;

	float pos_rscale = params.pos_scale == 0.f ? 0.f : 1.f / params.pos_scale;

	if (stream.target == 0)
	{
		for (size_t i = 0; i < stream.data.size(); ++i)
		{
			const Attr& a = stream.data[i];

			for (int k = 0; k < 3; ++k)
			{
				int v = meshopt_quantizeUnorm((a.f[k] - params.pos_offset[k]) * pos_rscale, params.pos_bits);

				min[k] = std::min(min[k], v);
				max[k] = std::max(max[k], v);
			}
		}
	}
	else
	{
		for (size_t i = 0; i < stream.data.size(); ++i)
		{
			const Attr& a = stream.data[i];

			for (int k = 0; k < 3; ++k)
			{
				int v = (a.f[k] >= 0.f ? 1 : -1) * meshopt_quantizeUnorm(fabsf(a.f[k]) * pos_rscale, params.pos_bits);

				min[k] = std::min(min[k], v);
				max[k] = std::max(max[k], v);
			}
		}
	}
}

StreamFormat writeIndexStream(std::string& bin, const std::vector<unsigned int>& stream)
{
	unsigned int maxi = 0;
	for (size_t i = 0; i < stream.size(); ++i)
		maxi = std::max(maxi, stream[i]);

	// save 16-bit indices if we can; note that we can't use restart index (65535)
	if (maxi < 65535)
	{
		for (size_t i = 0; i < stream.size(); ++i)
		{
			uint16_t v[1] = {uint16_t(stream[i])};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_scalar, cgltf_component_type_r_16u, false, 2};
		return format;
	}
	else
	{
		for (size_t i = 0; i < stream.size(); ++i)
		{
			uint32_t v[1] = {stream[i]};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_scalar, cgltf_component_type_r_32u, false, 4};
		return format;
	}
}

StreamFormat writeTimeStream(std::string& bin, const std::vector<float>& data)
{
	for (size_t i = 0; i < data.size(); ++i)
	{
		float v[1] = {data[i]};
		bin.append(reinterpret_cast<const char*>(v), sizeof(v));
	}

	StreamFormat format = {cgltf_type_scalar, cgltf_component_type_r_32f, false, 4};
	return format;
}

StreamFormat writeKeyframeStream(std::string& bin, cgltf_animation_path_type type, const std::vector<Attr>& data)
{
	if (type == cgltf_animation_path_type_rotation)
	{
		for (size_t i = 0; i < data.size(); ++i)
		{
			const Attr& a = data[i];

			int16_t v[4] = {
			    int16_t(meshopt_quantizeSnorm(a.f[0], 16)),
			    int16_t(meshopt_quantizeSnorm(a.f[1], 16)),
			    int16_t(meshopt_quantizeSnorm(a.f[2], 16)),
			    int16_t(meshopt_quantizeSnorm(a.f[3], 16)),
			};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_vec4, cgltf_component_type_r_16, true, 8};
		return format;
	}
	else if (type == cgltf_animation_path_type_weights)
	{
		for (size_t i = 0; i < data.size(); ++i)
		{
			const Attr& a = data[i];

			uint8_t v[1] = {uint8_t(meshopt_quantizeUnorm(a.f[0], 8))};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_scalar, cgltf_component_type_r_8u, true, 1};
		return format;
	}
	else if (type == cgltf_animation_path_type_translation || type == cgltf_animation_path_type_scale)
	{
		int bits = 15;

		for (size_t i = 0; i < data.size(); ++i)
		{
			const Attr& a = data[i];

			float v[3] = {
			    meshopt_quantizeFloat(a.f[0], bits),
			    meshopt_quantizeFloat(a.f[1], bits),
			    meshopt_quantizeFloat(a.f[2], bits)};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_vec3, cgltf_component_type_r_32f, false, 12};
		return format;
	}
	else
	{
		for (size_t i = 0; i < data.size(); ++i)
		{
			const Attr& a = data[i];

			float v[4] = {a.f[0], a.f[1], a.f[2], a.f[3]};
			bin.append(reinterpret_cast<const char*>(v), sizeof(v));
		}

		StreamFormat format = {cgltf_type_vec4, cgltf_component_type_r_32f, false, 16};
		return format;
	}
}

void compressVertexStream(std::string& bin, const std::string& data, size_t count, size_t stride)
{
	assert(data.size() == count * stride);

	std::vector<unsigned char> compressed(meshopt_encodeVertexBufferBound(count, stride));
	size_t size = meshopt_encodeVertexBuffer(&compressed[0], compressed.size(), data.c_str(), count, stride);

	bin.append(reinterpret_cast<const char*>(&compressed[0]), size);
}

void compressIndexStream(std::string& bin, const std::string& data, size_t count, size_t stride)
{
	assert(stride == 2 || stride == 4);
	assert(data.size() == count * stride);

	std::vector<unsigned char> compressed(meshopt_encodeIndexBufferBound(count, count * 3));
	size_t size = 0;

	if (stride == 2)
		size = meshopt_encodeIndexBuffer(&compressed[0], compressed.size(), reinterpret_cast<const uint16_t*>(data.c_str()), count);
	else
		size = meshopt_encodeIndexBuffer(&compressed[0], compressed.size(), reinterpret_cast<const uint32_t*>(data.c_str()), count);

	bin.append(reinterpret_cast<const char*>(&compressed[0]), size);
}
