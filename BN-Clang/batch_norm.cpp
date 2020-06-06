#include <immintrin.h>
#include <time.h>
#include <omp.h>
#include <iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

#define THREADS 16

DLLEXPORT float* avx2_malloc(unsigned long long size) {
	return (float*)_aligned_malloc(size, 32);
}

DLLEXPORT void batch_norm(float* x, float* cache, float* out, int size, int batchsize, float eps) {

	int sizeperbatch = size / batchsize;
	int spb8 = (sizeperbatch / 8) * 8;

	//#pragma omp parallel for
	for (int b = 0; b < batchsize; b++) {
		int p = b * sizeperbatch;
		float* px = x + p;
		float* po = out + p;

		float mu = 0;
		for (int i = 0; i < sizeperbatch; i++) {
			mu += px[i];
		}
		mu /= sizeperbatch;

		
		float* mt = (float*)_aligned_malloc(sizeof(float) * 8, 32);
		__asm {

			VXORPS ymm0, ymm0, ymm0
			VBROADCASTSS ymm1, mu

			MOV rsi, px

			MOV ecx, spb8
			loops:
			CMP ecx, 0
			JE loope

			VMOVUPS ymm2, [rsi]
			ADD rsi, 32

			VSUBPS ymm2, ymm2, ymm1
			VFMADD231PS ymm0, ymm2, ymm2

			SUB ecx, 8
			JMP loops
			loope:

			MOV rsi, mt
			VMOVUPS [rsi], ymm0
		}
		float sd = mt[0] + mt[1] + mt[2] + mt[3] + mt[4] + mt[5] + mt[6] + mt[7];
		for (int i = spb8; i < sizeperbatch; i++) {
			sd += (px[i] - mu) * (px[i] - mu);
		}
		sd /= sizeperbatch;
		sd = sqrt(sd + eps);
		cache[b * 2 + 0] = sd;
		cache[b * 2 + 1] = mu;

		__asm {

			VBROADCASTSS ymm0, mu
			VBROADCASTSS ymm1, sd

			MOV rsi, px
			MOV rdi, po

			MOV ecx, spb8
			loops:
			CMP ecx, 0
			JE loope

			VMOVUPS ymm2, [rsi]
			ADD rsi, 32
			VSUBPS ymm2, ymm2, ymm0
			VDIVPS ymm2, ymm2, ymm1
			VMOVUPS [rdi], ymm2
			ADD rdi, 32

			SUB ecx, 8
			JMP loops
			loope:

		}
		for (int i = spb8; i < sizeperbatch; i++) {
			po[i] = (px[i] - mu) / sd;
		}
		_aligned_free(mt);

	}

}

DLLEXPORT void batch_norm_back(float* x, float* out, float* cache, float* grad, int size, int batchsize) {

	int sizeperbatch = size / batchsize;
	int spb8 = 0;// (sizeperbatch / 8) * 8;

	#pragma omp parallel for
	for (int b = 0; b < batchsize; b++) {
		int p = b * sizeperbatch;
		float* px = x + p;
		float* pg = grad + p;
		float* po = out + p;

		float sder = cache[b * 2 + 0];
		float mu = cache[b * 2 + 1];

		float t1 = (2 * pow(sder, 3));
		float t2 = 2 / sizeperbatch;
		float t3 = 2 / sizeperbatch / sizeperbatch;

		for (int i = 0; i < sizeperbatch; i++) {
			float dsd = -pg[i] * (px[i] - mu) / t1;

			po[i] = pg[i] / sder + dsd * (px[i] - mu) * t2 + (-pg[i] / sder - dsd * (px[i] - mu) * t3);
		}

	}

}