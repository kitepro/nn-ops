#include <immintrin.h>
#include <time.h>
#include <omp.h>
#include <iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

#define THREADS 16

DLLEXPORT float* avx2_malloc(unsigned long long size) {
	return (float*)_aligned_malloc(size, 32);
}

DLLEXPORT void avx2_mrs(float* x, float* g, float* m, float* r, int size, float lr, float beta1, float beta2, float eps, int timestep) {
		
	#pragma omp parallel for
	for (int t = 0; t < THREADS; t++) {

		int i = t * ((float)size / THREADS);
		int ih = (t + 1) * ((float)size / THREADS);
		int im = i + ((ih - i) / 8) * 8;

		float* px = x + i;
		float* pg = g + i;
		float* pm = m + i;
		float* pr = r + i;
		int ic = 0;// im - i;
		float nbeta1 = 1 - beta1;
		float nbeta2 = 1 - beta2;
		float nbeta1t = 1 / (1 - pow(beta1, timestep));
		float nbeta2t = 1 / (1 - pow(beta2, timestep));
		float nlr = lr * sqrt(1 - pow(beta2, timestep)) / (1 - pow(beta1, timestep));
		__asm {

			MOV rsi, px
			MOV rdi, pg
			MOV r12, pm
			MOV r13, pr

			VBROADCASTSS ymm0, beta1
			VBROADCASTSS ymm1, beta2
			VBROADCASTSS ymm2, eps
			VBROADCASTSS ymm3, lr
			VBROADCASTSS ymm4, nbeta1
			VBROADCASTSS ymm5, nbeta2
			VBROADCASTSS ymm13, nbeta1t
			VBROADCASTSS ymm14, nbeta2t

			MOV ecx, ic
			iloops:
			CMP ecx, 0
			JE iloope

			VMOVUPS ymm6, [rdi]
			VMULPS ymm15, ymm6, ymm6

			// Update moment
			VMOVUPS ymm7, [r12]
			VMULPS ymm8, ymm7, ymm0
			VFMADD231PS ymm8, ymm6, ymm4
			VMOVUPS [r12], ymm8

			// Update rms
			VMOVUPS ymm9, [r13]
			VMULPS ymm10, ymm9, ymm1
			VFMADD231PS ymm10, ymm15, ymm5
			VMOVUPS [r13], ymm10

			// Bias correction
			VMULPS ymm8, ymm8, ymm13
			VMULPS ymm10, ymm10, ymm14

			// Update
			VMOVUPS ymm7, [rsi]
			VSQRTPS ymm10, ymm10
			VADDPS ymm10, ymm10, ymm2
			VDIVPS ymm8, ymm8, ymm10
			VFNMADD231PS ymm7, ymm8, ymm3
			VMOVUPS [rsi], ymm7

			ADD rsi, 32
			ADD rdi, 32
			ADD r12, 32
			ADD r13, 32

			SUB ecx, 8
			JMP iloops
			iloope:

		}
		for (; i < ih; i++) {
			m[i] = beta1 * m[i] + nbeta1 * g[i];
			r[i] = beta2 * r[i] + nbeta2 * g[i] * g[i];
			//x[i] -= lr * (m[i] * nbeta1t) / (sqrt(r[i] * nbeta2t) + eps);
			x[i] -= nlr * m[i] / (sqrt(r[i]) + eps);
		}

	}

}



/*
DLLEXPORT void avx2_mrs(float* x, float* g, float* m, float* r, int size, float alr, float slr, float sgdm, float beta1, float beta2, float eps, int timestep) {

	#pragma omp parallel for
	for (int t = 0; t < THREADS; t++) {

		int i = t * ((float)size / THREADS);
		int ih = (t + 1) * ((float)size / THREADS);
		int im = i + ((ih - i) / 8) * 8;
		float sgdmul = pow(sgdm, timestep);

		float* px = x + i;
		float* pg = g + i;
		float* pm = m + i;
		float* pr = r + i;
		int ic = im - i;
		float nbeta1 = 1 - beta1;
		float nbeta2 = 1 - beta2;
		float nbeta1t = 1 / (1 - pow(beta1, timestep));
		float nbeta2t = 1 / (1 - pow(beta2, timestep));
		for (; i < ih; i++) {
			m[i] = beta1 * m[i] + nbeta1 * g[i];
			r[i] = beta2 * r[i] + nbeta2 * g[i] * g[i];
			x[i] -= alr * (m[i] * nbeta1t) / (sqrt(r[i] * nbeta2t) + eps) * sgdmul + slr * g[i] * (1 - sgdmul);
		}

	}

}
*/