#include <immintrin.h>
#include <time.h>
#include <omp.h>
#include <iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

#define THREADS 16

DLLEXPORT float* avx2_malloc(unsigned long long size) {
	return (float*)_aligned_malloc(size, 32);
}

float HELPER_VALUE_0 = 0;
float HELPER_VALUE_1 = 1;
float EXP_HELPER_VALUE_2F = 0.5;
float EXP_HELPER_VALUE_3F = 0.16666666666666;
float EXP_HELPER_VALUE_4F = 0.0416666666;
float EXP_HELPER_VALUE_5F = 0.0083333333333;
float EXP_HELPER_VALUE_6F = 0.00138888888;
float EXP_HELPER_VALUE_7F = 0.0001984126;
float EXP_HELPER_VALUE_8F = 0.0000248015;
float EXP_HELPER_VALUE_9F = 0.0000027557319;

void avx2_sigmoid_x8(float* x, float* y, int C) {

	__asm {

		MOV rsi, x
		MOV rdi, y

		MOV ecx, C
		cloops:
		CMP ecx, 0
		JE cloope

		VMOVUPS ymm0, [rsi]
		VBROADCASTSS ymm1, HELPER_VALUE_0
		VSUBPS ymm0, ymm1, ymm0
		VMOVAPS ymm1, ymm0

		// e^-x in ymm2
		VBROADCASTSS ymm2, HELPER_VALUE_1
		VADDPS ymm2, ymm2, ymm1
		VMULPS ymm1, ymm1, ymm0							// X ** 2
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_2F
		VFMADD231PS ymm2, ymm1, ymm3

		VMULPS ymm1, ymm1, ymm0							// X ** 3
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_3F
		VFMADD231PS ymm2, ymm1, ymm3

		VMULPS ymm1, ymm1, ymm0							// X ** 4
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_4F
		VFMADD231PS ymm2, ymm1, ymm3

		VMULPS ymm1, ymm1, ymm0							// X ** 5
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_5F
		VFMADD231PS ymm2, ymm1, ymm3

		VMULPS ymm1, ymm1, ymm0							// X ** 6
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_6F
		VFMADD231PS ymm2, ymm1, ymm3

		VMULPS ymm1, ymm1, ymm0							// X ** 7
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_7F
		VFMADD231PS ymm2, ymm1, ymm3

		VMULPS ymm1, ymm1, ymm0							// X ** 8
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_8F
		VFMADD231PS ymm2, ymm1, ymm3

		VMULPS ymm1, ymm1, ymm0							// X ** 9
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_9F
		VFMADD231PS ymm2, ymm1, ymm3

		// 1/(1+e^-x)
		VBROADCASTSS ymm0, HELPER_VALUE_1
		VADDPS ymm1, ymm0, ymm2
		VDIVPS ymm1, ymm0, ymm1

		VMOVUPS [rdi], ymm1

		ADD rsi, 32
		ADD rdi, 32

		DEC ecx
		JMP cloops
		cloope:

	}

}

void avx2_sigmoid_derv_x8(float* x, float* y, float* g, int C) {

	__asm {

		MOV rsi, x
		MOV rdi, y
		MOV r12, g

		MOV ecx, C
		cloops :
		CMP ecx, 0
		JE cloope

		VMOVUPS ymm0, [rsi]
		VBROADCASTSS ymm1, HELPER_VALUE_0
		VSUBPS ymm0, ymm1, ymm0
		VMOVAPS ymm1, ymm0

		// e^-x in ymm2
		VBROADCASTSS ymm2, HELPER_VALUE_1
		VADDPS ymm2, ymm2, ymm1
		VMULPS ymm1, ymm1, ymm0							// X ** 2
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_2F
		VFMADD231PS ymm2, ymm1, ymm3

		VMULPS ymm1, ymm1, ymm0							// X ** 3
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_3F
		VFMADD231PS ymm2, ymm1, ymm3

		VMULPS ymm1, ymm1, ymm0							// X ** 4
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_4F
		VFMADD231PS ymm2, ymm1, ymm3

		VMULPS ymm1, ymm1, ymm0							// X ** 5
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_5F
		VFMADD231PS ymm2, ymm1, ymm3

		VMULPS ymm1, ymm1, ymm0							// X ** 6
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_6F
		VFMADD231PS ymm2, ymm1, ymm3

		VMULPS ymm1, ymm1, ymm0							// X ** 7
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_7F
		VFMADD231PS ymm2, ymm1, ymm3

		VMULPS ymm1, ymm1, ymm0							// X ** 8
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_8F
		VFMADD231PS ymm2, ymm1, ymm3

		VMULPS ymm1, ymm1, ymm0							// X ** 9
		VBROADCASTSS ymm3, EXP_HELPER_VALUE_9F
		VFMADD231PS ymm2, ymm1, ymm3

		// g*e^-x/(1+e^-x)^2
		VBROADCASTSS ymm0, HELPER_VALUE_1
		VADDPS ymm1, ymm0, ymm2
		VMULPS ymm1, ymm1, ymm1
		VDIVPS ymm1, ymm2, ymm1
		VMULPS ymm1, ymm1, [r12]

		VMOVUPS[rdi], ymm1

		ADD rsi, 32
		ADD rdi, 32
		ADD r12, 32

		DEC ecx
		JMP cloops
		cloope :

	}

}

void avx2_relu_x8(float* x, float* y, int C, float leak) {

	__asm {

		MOV rsi, x
		MOV rdi, y

		VBROADCASTSS ymm1, HELPER_VALUE_0
		VBROADCASTSS ymm4, HELPER_VALUE_1
		VBROADCASTSS ymm3, leak

		MOV ecx, C
		cloops:
		CMP ecx, 0
		JE cloope

		VMOVUPS ymm0, [rsi]
		ADD rsi, 32

		VCMPGTPS ymm2, ymm0, ymm1
		VANDPS ymm5, ymm0, ymm2
		VMULPS ymm0, ymm0, ymm3
		VXORPS ymm2, ymm2, ymm4
		VANDPS ymm0, ymm2, ymm0
		VADDPS ymm0, ymm0, ymm5

		VMOVUPS [rdi], ymm0
		ADD rdi, 32

		DEC ecx
		JMP cloops
		cloope:

	}

}

void avx2_relu_derv_x8(float* x, float* y, float* g, int C, float leak) {

	__asm {

		MOV rsi, x
		MOV rdi, y
		MOV r12, g

		VBROADCASTSS ymm1, HELPER_VALUE_0
		VBROADCASTSS ymm6, HELPER_VALUE_1
		VBROADCASTSS ymm4, leak		

		MOV ecx, C
		cloops :
		CMP ecx, 0
		JE cloope

		VMOVUPS ymm0, [rsi]
		ADD rsi, 32
		VMOVUPS ymm3, [r12]
		ADD r12, 32

		VCMPGTPS ymm2, ymm0, ymm1
		VANDPS ymm0, ymm3, ymm2
		VMULPS ymm3, ymm3, ymm4
		VXORPS ymm2, ymm2, ymm6
		VANDPS ymm3, ymm2, ymm3
		VADDPS ymm0, ymm0, ymm3

		VMOVUPS[rdi], ymm0
		ADD rdi, 32

		DEC ecx
		JMP cloops
		cloope :

	}

}



DLLEXPORT void avx2_actv_f(float* x, float* y, int size, int type, float* data) {

	#pragma omp parallel for
	for (int t = 0; t < THREADS; t++) {

		int il = t * ((float)size / THREADS);
		int ih = (t + 1) * ((float)size / THREADS);

		int ic = (ih - il) / 8;
		int im = il + ic * 8;
		if (type == 0) {
			avx2_relu_x8(x + il, y + il, ic, data[0]);
			for (int i = im; i < ih; i++) {
				y[i] = (x[i] > 0) ? x[i] : data[0] * x[i];
			}			
		}
		else if (type == 1) {
			//avx2_sigmoid_x8(x + il, y + il, ic);
			for (int i = il; i < ih; i++) {
				y[i] = 1.0 / (1.0 + exp(-x[i]));
			}
		}

	}

}

DLLEXPORT void avx2_actv_b(float* x, float* y, float* g, int size, int type, float* data) {
	
	#pragma omp parallel for
	for (int t = 0; t < THREADS; t++) {

		int il = t * ((float)size / THREADS);
		int ih = (t + 1) * ((float)size / THREADS);

		int ic = (ih - il) / 8;
		int im = il + ic * 8;
		if (type == 0) {
			avx2_relu_derv_x8(x + il, y + il, g + il, ic, data[0]);
			for (int i = im; i < ih; i++) {
				y[i] = (x[i] > 0) ? g[i] : data[0] * g[i];
			}
		}
		else if (type == 1) {
			//avx2_sigmoid_derv_x8(x + il, y + il, g + il, ic);
			for (int i = il; i < ih; i++) {
				float e = exp(-x[i]);
				y[i] = e * g[i] / (1 + e) / (1 + e);
			}
		}

	}

}