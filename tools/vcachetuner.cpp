#include "../src/meshoptimizer.h"
#include "fast_obj.h"
#include "../demo/miniz.h"

#include <algorithm>
#include <functional>
#include <vector>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

const int kCacheSizeMax = 16;
const int kValenceMax = 8;

namespace meshopt
{
	struct VertexScoreTable
	{
		float cache[1 + kCacheSizeMax];
		float live[1 + kValenceMax];
	};
} // namespace meshopt

void meshopt_optimizeVertexCacheTable(unsigned int* destination, const unsigned int* indices, size_t index_count, size_t vertex_count, const meshopt::VertexScoreTable* table);

struct Profile
{
	float weight;
	int cache, warp, triangle; // vcache tuning parameters
};

Profile profiles[] =
{
	{1.f, 0, 0, 0},  // Compression
	// {1.f, 14, 64, 128}, // AMD GCN
	// {1.f, 32, 32, 32},  // NVidia Pascal
	// {1.f, 16, 32, 32}, // NVidia Kepler, Maxwell
	// {1.f, 128, 0, 0}, // Intel
};

const int Profile_Count = sizeof(profiles) / sizeof(profiles[0]);

struct pcg32_random_t
{
	uint64_t state;
	uint64_t inc;
};

#define PCG32_INITIALIZER { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL }

uint32_t pcg32_random_r(pcg32_random_t* rng)
{
	uint64_t oldstate = rng->state;
	// Advance internal state
	rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
	// Calculate output function (XSH RR), uses old state for max ILP
	uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
	uint32_t rot = oldstate >> 59u;
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

pcg32_random_t rngstate = PCG32_INITIALIZER;

float rand01()
{
	return pcg32_random_r(&rngstate) / float(1ull << 32);
}

uint32_t rand32()
{
	return pcg32_random_r(&rngstate);
}

struct State
{
	float cache[kCacheSizeMax];
	float live[kValenceMax];
	float fitness;
};

struct Mesh
{
	const char* name;

	size_t vertex_count;
	std::vector<unsigned int> indices;

	float metric_base[Profile_Count];
};

Mesh gridmesh(unsigned int N)
{
	Mesh result;

	result.name = "grid";

	result.vertex_count = (N + 1) * (N + 1);
	result.indices.reserve(N * N * 6);

	for (unsigned int y = 0; y < N; ++y)
		for (unsigned int x = 0; x < N; ++x)
		{
			result.indices.push_back((y + 0) * (N + 1) + (x + 0));
			result.indices.push_back((y + 0) * (N + 1) + (x + 1));
			result.indices.push_back((y + 1) * (N + 1) + (x + 0));

			result.indices.push_back((y + 1) * (N + 1) + (x + 0));
			result.indices.push_back((y + 0) * (N + 1) + (x + 1));
			result.indices.push_back((y + 1) * (N + 1) + (x + 1));
		}

	return result;
}

Mesh objmesh(const char* path)
{
	fastObjMesh* obj = fast_obj_read(path);
	if (!obj)
	{
		printf("Error loading %s: file not found\n", path);
		return Mesh();
	}

	size_t total_indices = 0;

	for (unsigned int i = 0; i < obj->face_count; ++i)
		total_indices += 3 * (obj->face_vertices[i] - 2);

	struct Vertex
	{
		float px, py, pz;
		float nx, ny, nz;
		float tx, ty;
	};

	std::vector<Vertex> vertices(total_indices);

	size_t vertex_offset = 0;
	size_t index_offset = 0;

	for (unsigned int i = 0; i < obj->face_count; ++i)
	{
		for (unsigned int j = 0; j < obj->face_vertices[i]; ++j)
		{
			fastObjIndex gi = obj->indices[index_offset + j];

			Vertex v =
			    {
			        obj->positions[gi.p * 3 + 0],
			        obj->positions[gi.p * 3 + 1],
			        obj->positions[gi.p * 3 + 2],
			        obj->normals[gi.n * 3 + 0],
			        obj->normals[gi.n * 3 + 1],
			        obj->normals[gi.n * 3 + 2],
			        obj->texcoords[gi.t * 2 + 0],
			        obj->texcoords[gi.t * 2 + 1],
			    };

			// triangulate polygon on the fly; offset-3 is always the first polygon vertex
			if (j >= 3)
			{
				vertices[vertex_offset + 0] = vertices[vertex_offset - 3];
				vertices[vertex_offset + 1] = vertices[vertex_offset - 1];
				vertex_offset += 2;
			}

			vertices[vertex_offset] = v;
			vertex_offset++;
		}

		index_offset += obj->face_vertices[i];
	}

	fast_obj_destroy(obj);

	Mesh result;

	result.name = path;

	std::vector<unsigned int> remap(total_indices);

	size_t total_vertices = meshopt_generateVertexRemap(&remap[0], NULL, total_indices, &vertices[0], total_indices, sizeof(Vertex));

	result.indices.resize(total_indices);
	meshopt_remapIndexBuffer(&result.indices[0], NULL, total_indices, &remap[0]);

	result.vertex_count = total_vertices;

	return result;
}

template <typename T>
size_t compress(const std::vector<T>& data)
{
	std::vector<unsigned char> cbuf(tdefl_compress_bound(data.size() * sizeof(T)));
	unsigned int flags = tdefl_create_comp_flags_from_zip_params(MZ_DEFAULT_LEVEL, 15, MZ_DEFAULT_STRATEGY);
	return tdefl_compress_mem_to_mem(&cbuf[0], cbuf.size(), &data[0], data.size() * sizeof(T), flags);
}

void compute_metric(const State* state, const Mesh& mesh, float result[Profile_Count])
{
	std::vector<unsigned int> indices(mesh.indices.size());

	if (state)
	{
		meshopt::VertexScoreTable table = {};
		memcpy(table.cache + 1, state->cache, kCacheSizeMax * sizeof(float));
		memcpy(table.live + 1, state->live, kValenceMax * sizeof(float));
		meshopt_optimizeVertexCacheTable(&indices[0], &mesh.indices[0], mesh.indices.size(), mesh.vertex_count, &table);
	}
	else
	{
		meshopt_optimizeVertexCache(&indices[0], &mesh.indices[0], mesh.indices.size(), mesh.vertex_count);
	}

	meshopt_optimizeVertexFetch(NULL, &indices[0], indices.size(), NULL, mesh.vertex_count, 0);

	for (int profile = 0; profile < Profile_Count; ++profile)
	{
		if (profiles[profile].cache)
		{
			meshopt_VertexCacheStatistics stats = meshopt_analyzeVertexCache(&indices[0], indices.size(), mesh.vertex_count, profiles[profile].cache, profiles[profile].warp, profiles[profile].triangle);
			result[profile] = stats.atvr;
		}
		else
		{
			std::vector<unsigned char> ibuf(meshopt_encodeIndexBufferBound(indices.size(), mesh.vertex_count));
			ibuf.resize(meshopt_encodeIndexBuffer(&ibuf[0], ibuf.size(), &indices[0], indices.size(), 1));

			// take into account both pre-deflate and post-deflate size but focus a bit more on post-deflate
			size_t csize = ibuf.size() / 2 + compress(ibuf);

			result[profile] = double(csize) / double(indices.size() / 3);
		}
	}
}

float fitness_score(const State& state, const std::vector<Mesh>& meshes)
{
	float result = 0;
	float count = 0;

	for (auto& mesh : meshes)
	{
		float metric[Profile_Count];
		compute_metric(&state, mesh, metric);

		for (int profile = 0; profile < Profile_Count; ++profile)
		{
			result += mesh.metric_base[profile] / metric[profile] * profiles[profile].weight;
			count += profiles[profile].weight;
		}
	}

	return result / count;
}

size_t rndindex(const std::vector<float>& prob)
{
	float r = rand01();

	for (size_t i = 0; i < prob.size(); ++i)
	{
		r -= prob[i];

		if (r <= 0)
			return i;
	}

	return prob.size() - 1;
}

State mutate(const State& state)
{
	State result = state;

	if (rand01() < 0.7f)
	{
		size_t idxcache = std::min(int(rand01() * kCacheSizeMax + 0.5f), int(kCacheSizeMax - 1));

		result.cache[idxcache] = rand01();
	}

	if (rand01() < 0.7f)
	{
		size_t idxlive = std::min(int(rand01() * kValenceMax + 0.5f), int(kValenceMax - 1));

		result.live[idxlive] = rand01();
	}

	if (rand01() < 0.2f)
	{
		uint32_t mask = rand32();

		for (size_t i = 0; i < kCacheSizeMax; ++i)
			if (mask & (1 << i))
				result.cache[i] *= 0.9f + 0.2f * rand01();
	}

	if (rand01() < 0.2f)
	{
		uint32_t mask = rand32();

		for (size_t i = 0; i < kValenceMax; ++i)
			if (mask & (1 << i))
				result.live[i] *= 0.9f + 0.2f * rand01();
	}

	if (rand01() < 0.05f)
	{
		uint32_t mask = rand32();

		for (size_t i = 0; i < kCacheSizeMax; ++i)
			if (mask & (1 << i))
				result.cache[i] = rand01();
	}

	if (rand01() < 0.05f)
	{
		uint32_t mask = rand32();

		for (size_t i = 0; i < kValenceMax; ++i)
			if (mask & (1 << i))
				result.live[i] = rand01();
	}

	return result;
}

bool accept(float fitnew, float fitold, float temp)
{
	if (fitnew >= fitold)
		return true;

	if (temp == 0)
		return false;

	float prob = exp2((fitnew - fitold) / temp);

	return rand01() < prob;
}

std::vector<State> gen0(size_t count, const std::vector<Mesh>& meshes)
{
	std::vector<State> result;

	for (size_t i = 0; i < count; ++i)
	{
		State state = {};

		for (int j = 0; j < kCacheSizeMax; ++j)
			state.cache[j] = rand01();

		for (int j = 0; j < kValenceMax; ++j)
			state.live[j] = rand01();

		state.fitness = fitness_score(state, meshes);

		result.push_back(state);
	}

	return result;
}

// https://en.wikipedia.org/wiki/Genetic_algorithm
std::pair<State, float> genN_GA(std::vector<State>& seed, const std::vector<Mesh>& meshes, float crossover, float mutate)
{
	std::vector<State> result;
	result.reserve(seed.size());

	std::vector<float> seedprob(seed.size());

#pragma omp parallel for
	for (size_t i = 0; i < seed.size(); ++i)
	{
		seedprob[i] = fitness_score(seed[i], meshes);
	}

	State best = {};
	float bestfit = 0;
	float probsum = 0;

	for (size_t i = 0; i < seed.size(); ++i)
	{
		float score = seedprob[i];
		probsum += score;

		if (score > bestfit)
		{
			best = seed[i];
			bestfit = score;
		}
	}

	for (auto& prob : seedprob)
	{
		prob /= probsum;
	}

	std::vector<unsigned int> seedidx;
	seedidx.reserve(seed.size());
	for (size_t i = 0; i < seed.size(); ++i)
		seedidx.push_back(i);

	std::sort(seedidx.begin(), seedidx.end(), [&](size_t l, size_t r) { return seedprob[l] < seedprob[r]; });

	while (result.size() < seed.size() / 4)
	{
		size_t idx = seedidx.back();
		seedidx.pop_back();

		result.push_back(seed[idx]);
	}

	while (result.size() < seed.size())
	{
		State s0 = seed[rndindex(seedprob)];
		State s1 = seed[rndindex(seedprob)];

		State state = s0;

		// crossover
		if (rand01() < crossover)
		{
			size_t idxcache = std::min(int(rand01() * kCacheSizeMax + 0.5f), 15);

			memcpy(state.cache + idxcache, s1.cache + idxcache, (kCacheSizeMax - idxcache) * sizeof(float));
		}

		if (rand01() < crossover)
		{
			size_t idxlive = std::min(int(rand01() * kValenceMax + 0.5f), 7);

			memcpy(state.live + idxlive, s1.live + idxlive, (kValenceMax - idxlive) * sizeof(float));
		}

		// mutate
		if (rand01() < mutate)
		{
			size_t idxcache = std::min(int(rand01() * kCacheSizeMax + 0.5f), 15);

			state.cache[idxcache] = rand01();
		}

		if (rand01() < mutate)
		{
			size_t idxlive = std::min(int(rand01() * kValenceMax + 0.5f), 7);

			state.live[idxlive] = rand01();
		}

		result.push_back(state);
	}

	seed.swap(result);

	return std::make_pair(best, bestfit);
}

// https://en.wikipedia.org/wiki/Simulated_annealing
std::pair<State, float> genN_SA(std::vector<State>& seed, const std::vector<Mesh>& meshes, size_t steps)
{
	std::vector<State> result;
	result.reserve(seed.size() * (1 + steps));

	// perform several parallel steps of mutation for each temperature
	for (size_t i = 0; i < seed.size(); ++i)
	{
		result.push_back(seed[i]);

		for (size_t s = 0; s < steps; ++s)
			result.push_back(mutate(seed[i]));
	}

	// compute fitness for all temperatures & mutations in parallel
	std::vector<float> resultfit(result.size());

#pragma omp parallel for
	for (size_t i = 0; i < result.size(); ++i)
	{
		resultfit[i] = fitness_score(result[i], meshes);
	}

	// perform annealing for each temperature
	std::vector<float> seedfit(seed.size());

	for (size_t i = 0; i < seed.size(); ++i)
	{
		size_t offset = i * (1 + steps);

		seedfit[i] = resultfit[offset];

		float temp = (float(i) / float(seed.size() - 1)) / 0.1f;

		for (size_t s = 0; s < steps; ++s)
		{
			if (accept(resultfit[offset + s + 1], seedfit[i], temp))
			{
				seedfit[i] = resultfit[offset + s + 1];
				seed[i] = result[offset + s + 1];
			}
		}
	}

	// perform promotion from each temperature to the next one
	for (size_t i = seed.size() - 1; i > 0; --i)
	{
		if (seedfit[i] > seedfit[i - 1])
		{
			seedfit[i - 1] = seedfit[i];
			seed[i - 1] = seed[i];
			break;
		}
	}

	return std::make_pair(seed[0], seedfit[0]);
}

// https://en.wikipedia.org/wiki/Differential_evolution
// Good Parameters for Differential Evolution. Magnus Erik Hvass Pedersen, 2010
std::pair<State, float> genN_DE(std::vector<State>& seed, const std::vector<Mesh>& meshes, float crossover = 0.8803f, float weight = 0.4717f)
{
	std::vector<State> result(seed.size());

	for (size_t i = 0; i < seed.size(); ++i)
	{
		for (;;)
		{
			int a = rand32() % seed.size();
			int b = rand32() % seed.size();
			int c = rand32() % seed.size();

			if (a == b || a == c || b == c || a == int(i) || b == int(i) || c == int(i))
				continue;

			int rc = rand32() % kCacheSizeMax;
			int rl = rand32() % kValenceMax;

			for (int j = 0; j < kCacheSizeMax; ++j)
			{
				float r = rand01();

				if (r < crossover || j == rc)
					result[i].cache[j] = std::max(0.f, std::min(1.f, seed[a].cache[j] + weight * (seed[b].cache[j] - seed[c].cache[j])));
				else
					result[i].cache[j] = seed[i].cache[j];
			}

			for (int j = 0; j < kValenceMax; ++j)
			{
				float r = rand01();

				if (r < crossover || j == rl)
					result[i].live[j] = std::max(0.f, std::min(1.f, seed[a].live[j] + weight * (seed[b].live[j] - seed[c].live[j])));
				else
					result[i].live[j] = seed[i].live[j];
			}

			break;
		}
	}

	#pragma omp parallel for
	for (size_t i = 0; i < seed.size(); ++i)
	{
		result[i].fitness = fitness_score(result[i], meshes);
	}

	State best = {};
	float bestfit = 0;

	for (size_t i = 0; i < seed.size(); ++i)
	{
		if (result[i].fitness > seed[i].fitness)
			seed[i] = result[i];

		if (seed[i].fitness > bestfit)
		{
			best = seed[i];
			bestfit = seed[i].fitness;
		}
	}

	return std::make_pair(best, bestfit);
}

bool load_state(const char* path, std::vector<State>& result)
{
	FILE* file = fopen(path, "rb");
	if (!file)
		return false;

	State state;

	result.clear();

	while (fread(&state, sizeof(State), 1, file) == 1)
		result.push_back(state);

	fclose(file);

	return true;
}

bool save_state(const char* path, const std::vector<State>& result)
{
	FILE* file = fopen(path, "wb");
	if (!file)
		return false;

	for (auto& state : result)
	{
		if (fwrite(&state, sizeof(State), 1, file) != 1)
		{
			fclose(file);
			return false;
		}
	}

	return fclose(file) == 0;
}

void dump_state(const State& state)
{
	printf("cache:");
	for (int i = 0; i < kCacheSizeMax; ++i)
	{
		printf(" %.3f", state.cache[i]);
	}
	printf("\n");

	printf("live:");
	for (int i = 0; i < kValenceMax; ++i)
	{
		printf(" %.3f", state.live[i]);
	}
	printf("\n");
}

void dump_stats(const State& state, const std::vector<Mesh>& meshes)
{
	float improvement[Profile_Count] = {};

	for (size_t i = 0; i < meshes.size(); ++i)
	{
		float metric[Profile_Count];
		compute_metric(&state, meshes[i], metric);

		printf(" %s", meshes[i].name);
		for (int profile = 0; profile < Profile_Count; ++profile)
			printf(" %f", metric[profile]);

		for (int profile = 0; profile < Profile_Count; ++profile)
			improvement[profile] += meshes[i].metric_base[profile] / metric[profile];
	}

	printf("; improvement");
	for (int profile = 0; profile < Profile_Count; ++profile)
		printf(" %f", improvement[profile] / float(meshes.size()));

	printf("\n");
}

enum Algorithm
{
	GA,
	SA,
	DE
};

int main(int argc, char** argv)
{
	Algorithm algorithm = DE;
	int seeds = 0;

	if (strcmp(argv[1], "GA") == 0)
	{
		algorithm = GA;
		seeds = 1000;
	}
	else if (strcmp(argv[1], "SA") == 0)
	{
		algorithm = SA;
		seeds = 32;
	}
	else if (strcmp(argv[1], "DE") == 0)
	{
		algorithm = DE;
		seeds = 95;
	}
	else
	{
		printf("Error: need to specify GA/SA/DE as first argument\n");
		return 1;
	}

	std::vector<Mesh> meshes;
	meshes.push_back(gridmesh(50));

	for (int i = 2; i < argc; ++i)
		meshes.push_back(objmesh(argv[i]));

	size_t total_triangles = 0;

	for (auto& mesh : meshes)
	{
		compute_metric(nullptr, mesh, mesh.metric_base);

		total_triangles += mesh.indices.size() / 3;
	}

	std::vector<State> pop;
	size_t gen = 0;

	if (load_state("mutator.state", pop))
	{
		printf("Loaded %d state vectors\n", int(pop.size()));
	}
	else
	{
		pop = gen0(seeds, meshes);
	}

	printf("%d meshes, %.1fM triangles\n", int(meshes.size()), double(total_triangles) / 1e6);

	for (;;)
	{
		std::pair<State, float> best;

		switch (algorithm)
		{
		case GA:
			best = genN_GA(pop, meshes, 0.7f, 0.3f);
			break;

		case SA:
			best = genN_SA(pop, meshes, 31);
			break;

		case DE:
			best = genN_DE(pop, meshes);
			break;
		}

		gen++;

		if (gen % 10 == 0)
		{
			printf("%s: %d: fitness %f;", argv[1], int(gen), best.second);
			dump_stats(best.first, meshes);
		}
		else
		{
			printf("%s: %d: fitness %f\n", argv[1], int(gen), best.second);
		}

		dump_state(best.first);

		if (save_state("mutator.state-temp", pop) && rename("mutator.state-temp", "mutator.state") == 0)
		{
		}
		else
		{
			printf("ERROR: Can't save state\n");
		}
	}
}
