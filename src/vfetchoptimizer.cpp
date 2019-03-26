// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <string.h>

size_t meshopt_optimizeVertexFetchRemap(unsigned int* destination, const unsigned int* indices, size_t index_count, size_t vertex_count)
{
	assert(index_count % 3 == 0);

	memset(destination, -1, vertex_count * sizeof(unsigned int));

	unsigned int next_vertex = 0;

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int index = indices[i];
		assert(index < vertex_count);

		if (destination[index] == ~0u)
		{
			destination[index] = next_vertex++;
		}
	}

	assert(next_vertex <= vertex_count);

	return next_vertex;
}

size_t meshopt_optimizeVertexFetch(void* destination, unsigned int* indices, size_t index_count, const void* vertices, size_t vertex_count, size_t vertex_size)
{
	assert(index_count % 3 == 0);
	assert(vertex_size > 0 && vertex_size <= 256);

	meshopt_Allocator allocator;

	// support in-place optimization
	if (destination == vertices)
	{
		unsigned char* vertices_copy = allocator.allocate<unsigned char>(vertex_count * vertex_size);
		memcpy(vertices_copy, vertices, vertex_count * vertex_size);
		vertices = vertices_copy;
	}

	// build vertex remap table
	unsigned int* vertex_remap = allocator.allocate<unsigned int>(vertex_count);
	memset(vertex_remap, -1, vertex_count * sizeof(unsigned int));

	unsigned int next_vertex = 0;

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int index = indices[i];
		assert(index < vertex_count);

		unsigned int& remap = vertex_remap[index];

		// vertex was not added to destination VB
		if (remap == ~0u)
		{
			memcpy(static_cast<unsigned char*>(destination) + next_vertex * vertex_size, static_cast<const unsigned char*>(vertices) + index * vertex_size, vertex_size);

			remap = next_vertex++;
		}

		// modify indices in place
		indices[i] = remap;
	}

	assert(next_vertex <= vertex_count);

	return next_vertex;
}

size_t meshopt_optimizeVertexFetchInplace(unsigned int* indices, size_t index_count, void* vertices, size_t vertex_count, size_t vertex_size)
{
	assert(index_count % 3 == 0);
	assert(vertex_size > 0 && vertex_size <= 256);

	meshopt_Allocator allocator;

	// remaps: where did the source vertex go?
	// remapd: which source vertex is in the array?
	unsigned int* remaps = allocator.allocate<unsigned int>(vertex_count);
	unsigned int* remapd = allocator.allocate<unsigned int>(vertex_count);

	for (size_t i = 0; i < vertex_count; ++i)
	{
		remaps[i] = unsigned(i);
		remapd[i] = unsigned(i);
	}

	char vertex_temp[256];

	unsigned int next_vertex = 0;

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int index = indices[i];
		assert(index < vertex_count);

		// index isn't remapped yet
		if (remaps[index] >= next_vertex)
		{
			unsigned char* vn = static_cast<unsigned char*>(vertices) + next_vertex * vertex_size;
			unsigned char* vi = static_cast<unsigned char*>(vertices) + remaps[index] * vertex_size;

			// swap next & current vertex
			memcpy(vertex_temp, vn, vertex_size);
			memcpy(vn, vi, vertex_size);
			memcpy(vi, vertex_temp, vertex_size);

			// adjust remap tables to follow the swap
			unsigned int p0 = next_vertex;
			unsigned int l0 = remapd[next_vertex];

			unsigned int p1 = remaps[index];
			unsigned int l1 = index;

			remapd[p0] = index;
			remapd[p1] = l0;

			remaps[l0] = p1;
			remaps[l1] = p0;

			next_vertex++;
		}

		// modify indices in place
		indices[i] = remaps[index];
	}

	assert(next_vertex <= vertex_count);

	return next_vertex;
}

size_t meshopt_optimizeVertexFetchInplaceMulti(unsigned int* indices, size_t index_count, size_t vertex_count, const meshopt_Stream* streams, size_t stream_count)
{
	assert(index_count % 3 == 0);
	assert(stream_count > 0 && stream_count <= 16);

	meshopt_Allocator allocator;

	// remaps: where did the source vertex go?
	// remapd: which source vertex is in the array?
	unsigned int* remaps = allocator.allocate<unsigned int>(vertex_count);
	unsigned int* remapd = allocator.allocate<unsigned int>(vertex_count);

	for (size_t i = 0; i < vertex_count; ++i)
	{
		remaps[i] = unsigned(i);
		remapd[i] = unsigned(i);
	}

	char vertex_temp[256];

	unsigned int next_vertex = 0;

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int index = indices[i];
		assert(index < vertex_count);

		// index isn't remapped yet
		if (remaps[index] >= next_vertex)
		{
			// swap next & current vertex
			for (size_t j = 0; j < stream_count; ++j)
			{
				const meshopt_Stream& s = streams[j];

				unsigned char* data = static_cast<unsigned char*>(const_cast<void*>(s.data));
				unsigned char* vn = data + next_vertex * s.stride;
				unsigned char* vi = data + remaps[index] * s.stride;

				memcpy(vertex_temp, vn, s.size);
				memcpy(vn, vi, s.size);
				memcpy(vi, vertex_temp, s.size);
			}

			// adjust remap tables to follow the swap
			unsigned int p0 = next_vertex;
			unsigned int l0 = remapd[next_vertex];

			unsigned int p1 = remaps[index];
			unsigned int l1 = index;

			remapd[p0] = index;
			remapd[p1] = l0;

			remaps[l0] = p1;
			remaps[l1] = p0;

			next_vertex++;
		}

		// modify indices in place
		indices[i] = remaps[index];
	}

	assert(next_vertex <= vertex_count);

	return next_vertex;
}