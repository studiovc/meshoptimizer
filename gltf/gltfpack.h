// gltfpack is part of meshoptimizer library; see meshoptimizer.h for version/license details
//
// gltfpack is a command-line tool that takes a glTF file as an input and can produce two types of files:
// - regular glb/gltf files that use data that has been optimized for GPU consumption using various cache optimizers
// and quantization
// - packed glb/gltf files that additionally use meshoptimizer codecs to reduce the size of vertex/index data; these
// files can be further compressed with deflate/etc.
//
// To load regular glb files, it should be sufficient to use a standard glTF loader (although note that these files
// use quantized position/texture coordinates that require support for KHR_mesh_quantization; THREE.js and BabylonJS
// support these files out of the box).
// To load packed glb files, meshoptimizer vertex decoder needs to be integrated into the loader; demo/GLTFLoader.js
// contains a work-in-progress loader - please note that the extension specification isn't ready yet so the format
// will change!
#pragma once

#include <string>
#include <vector>

#include "../tools/cgltf.h"

struct Attr
{
	float f[4];
};

struct Stream
{
	cgltf_attribute_type type;
	int index;
	int target; // 0 = base mesh, 1+ = morph target

	std::vector<Attr> data;
};

struct Mesh
{
	cgltf_node* node;

	cgltf_material* material;
	cgltf_skin* skin;

	cgltf_primitive_type type;

	std::vector<Stream> streams;
	std::vector<unsigned int> indices;

	size_t targets;
	std::vector<float> target_weights;
	std::vector<const char*> target_names;
};

struct Track
{
	cgltf_node* node;
	cgltf_animation_path_type path;

	bool dummy;

	size_t components; // 1 unless path is cgltf_animation_path_type_weights

	cgltf_interpolation_type interpolation;

	std::vector<float> time; // empty for resampled or constant animations
	std::vector<Attr> data;
};

struct Animation
{
	const char* name;

	float start;
	int frames;

	std::vector<Track> tracks;
};

struct Settings
{
	int pos_bits;
	int tex_bits;
	int nrm_bits;
	bool nrm_unnormalized;

	int anim_freq;
	bool anim_const;

	bool keep_named;

	float simplify_threshold;
	bool simplify_aggressive;

	bool texture_embed;
	bool texture_basis;
	bool texture_ktx2;

	int texture_quality;

	bool compress;
	bool fallback;

	int verbose;
};

struct QuantizationParams
{
	float pos_offset[3];
	float pos_scale;
	int pos_bits;

	float uv_offset[2];
	float uv_scale[2];
	int uv_bits;
};

struct StreamFormat
{
	cgltf_type type;
	cgltf_component_type component_type;
	bool normalized;
	size_t stride;
};

struct NodeInfo
{
	bool keep;
	bool animated;

	unsigned int animated_paths;

	int remap;
	std::vector<size_t> meshes;
};

struct MaterialInfo
{
	bool keep;

	int remap;
};

struct ImageInfo
{
	bool normal_map;
	bool srgb;
};

struct ExtensionInfo
{
	const char* name;

	bool used;
	bool required;
};

struct BufferView
{
	enum Kind
	{
		Kind_Vertex,
		Kind_Index,
		Kind_Skin,
		Kind_Time,
		Kind_Keyframe,
		Kind_Image,
		Kind_Count
	};

	Kind kind;
	int variant;
	size_t stride;
	bool compressed;

	std::string data;

	size_t bytes;
};

bool compareMeshTargets(const Mesh& lhs, const Mesh& rhs);

void mergeMeshes(std::vector<Mesh>& meshes, const Settings& settings);
void filterEmptyMeshes(std::vector<Mesh>& meshes);

void processMesh(Mesh& mesh, const Settings& settings);
void processAnimation(Animation& animation, const Settings& settings);
