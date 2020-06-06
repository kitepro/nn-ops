#include <immintrin.h>
#include <time.h>
#include <omp.h>
#include <iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

#define THREADS 16

DLLEXPORT float* avx2_malloc(unsigned long long size) {
	return (float*)_aligned_malloc(size, 32);
}

DLLEXPORT void memcopy(float* x, float* y, int size) {

	#pragma omp parallel for
	for (int t = 0; t < THREADS; t++) {
		int l = t * ((float)size / THREADS);
		int h = (t + 1) * ((float)size / THREADS);
		memcpy(y + l, x + l, sizeof(float) * (h - l));
	}

}

DLLEXPORT float* pointer_offset(float* x, int offset) {
	return x + offset;
}

DLLEXPORT void avx2_add(float* x, float* y, int xsize, int ysize) {

	#pragma omp parallel for
	for (int t = 0; t < THREADS; t++) {

		int il = t * ((float)(xsize / ysize) / THREADS);
		int ih = (t + 1) * ((float)(xsize / ysize) / THREADS);

		int ym = (ysize / 8) * 8;

		for (int i = il; i < ih; i++) {
			float* px = x + i * ysize;
			__asm {

				MOV rsi, y
				MOV rdi, px

				MOV ecx, ym
				yloops:
				CMP ecx, 0
				JE yloope

				VMOVUPS ymm0, [rsi]
				VADDPS ymm1, ymm0, [rdi]
				VMOVUPS [rdi], ymm1

				ADD rsi, 32
				ADD rdi, 32

				SUB ecx, 8
				JMP yloops
				yloope:
			}
			int p = i * ysize;
			for (int j = ym; j < ysize; j++) {
				x[p + j] += y[j];
			}
		}
	}

}

DLLEXPORT void avx2_sum(float* x, float* y, int stride, int xsize) {

	int c = xsize / stride;

	#pragma omp parallel for
	for (int t = 0; t < stride; t++) {

		float s = 0;
		float* px = x + t;
		for (int i = 0; i < c; i++) {
			s += *px;
			px += stride;
		}
		y[t] = s;

	}

}
