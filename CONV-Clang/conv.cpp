#include <immintrin.h>
#include <time.h>
#include <omp.h>
#include <iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

#define THREADS 16

DLLEXPORT float* avx2_malloc(unsigned long long size) {
	return (float*)_aligned_malloc(size, 32);
}


DLLEXPORT void avx2_conv3d(float* img, float* kern, float* out, int H, int W, int C, int N, int Kx, int Ky, int stride, int padding, int batchsize) {

	int NH = (H - Kx) / stride + 1;
	int NW = (W - Ky) / stride + 1;

	#pragma omp parallel for
	for (int n = 0; n < N; n++) {
		memset(out + n * batchsize * NH * NW, 0, sizeof(float) * batchsize * NH * NW);
	}


	int NWh = (NW / 3) * 3;
	int Nh = (N / 4) * 4;
	int C8 = (C / 8) * 8;

	int WC = W * C;
	int HWC = H * WC;
	int NWN = NW * N;
	int NHNWN = NH * NWN;
	int NC = N * C;
	int KyNC = Ky * NC;

	int CS = C * stride;
	long long CS4 = CS * 4;
	long long C4 = C * 4;


	/*
	#pragma omp parallel for
	for (int n = 0; n < N; n++) {
		for (int b = 0; b < batchsize; b++) {
			for (int nh = 0; nh < NH; nh++) {
				for (int nw = 0; nw < NW; nw++) {
					for (int x = 0; x < Kx; x++) {
						int h = nh * stride + x;
						for (int y = 0; y < Ky; y++) {
							int w = nw * stride + y;
							for (int c = 0; c < C; c++) {
								out[b * NHNWN + nh * NWN + nw * N + n] += img[b * HWC + h * WC + w * C + c] * kern[x * KyNC + y * NC + n * C + c];
							}
						}
					}
				}
			}
		}
	}
	*/

	#pragma omp parallel for
	for (int nh = 0; nh < NH; nh++) {
		float* mt = (float*)_aligned_malloc(sizeof(float) * 8 * 12, 32);
		for (int b = 0; b < batchsize; b++) {
			for (int x = 0; x < Kx; x++) {
				int h = nh * stride + x;
				for (int y = 0; y < Ky; y++) {
					for (int nw = 0; nw < NWh; nw += 3) {
						int w = nw * stride + y;
						for (int n = 0; n < Nh; n += 4) {

							float* pImg = img + (b * HWC + h * WC + w * C);
							float* pKern = kern + (x * KyNC + y * NC + n * C);
							float* pOut = out + (b * NHNWN + nh * NWN + nw * N + n);

							__asm {

								VXORPS ymm0, ymm0, ymm0
								VXORPS ymm1, ymm1, ymm1
								VXORPS ymm2, ymm2, ymm2
								VXORPS ymm3, ymm3, ymm3
								VXORPS ymm4, ymm4, ymm4
								VXORPS ymm5, ymm5, ymm5
								VXORPS ymm6, ymm6, ymm6
								VXORPS ymm7, ymm7, ymm7
								VXORPS ymm8, ymm8, ymm8
								VXORPS ymm9, ymm9, ymm9
								VXORPS ymm10, ymm10, ymm10
								VXORPS ymm11, ymm11, ymm11

								MOV r12, CS4
								MOV r13, C4

								MOV rsi, pImg
								MOV rdi, pKern

								MOV ecx, C8
								cloops:
								CMP ecx, 0
								JE cloope

								MOV rdx, rsi
								VMOVUPS ymm12, [rsi]
								ADD rsi, r12
								VMOVUPS ymm13, [rsi]
								ADD rsi, r12
								VMOVUPS ymm14, [rsi]
								MOV rsi, rdx
								ADD rsi, 32

								MOV rdx, rdi
								VMOVUPS ymm15, [rdi]
								ADD rdi, r13
								VFMADD231PS ymm0, ymm15, ymm12
								VFMADD231PS ymm1, ymm15, ymm13
								VFMADD231PS ymm2, ymm15, ymm14
								VMOVUPS ymm15, [rdi]
								ADD rdi, r13
								VFMADD231PS ymm3, ymm15, ymm12
								VFMADD231PS ymm4, ymm15, ymm13
								VFMADD231PS ymm5, ymm15, ymm14
								VMOVUPS ymm15, [rdi]
								ADD rdi, r13
								VFMADD231PS ymm6, ymm15, ymm12
								VFMADD231PS ymm7, ymm15, ymm13
								VFMADD231PS ymm8, ymm15, ymm14
								VMOVUPS ymm15, [rdi]
								VFMADD231PS ymm9, ymm15, ymm12
								VFMADD231PS ymm10, ymm15, ymm13
								VFMADD231PS ymm11, ymm15, ymm14
								MOV rdi, rdx
								ADD rdi, 32

								SUB ecx, 8
								JMP cloops
								cloope:

								MOV rsi, mt
								VMOVUPS[rsi], ymm0
								ADD rsi, 32
								VMOVUPS[rsi], ymm1
								ADD rsi, 32
								VMOVUPS[rsi], ymm2
								ADD rsi, 32
								VMOVUPS[rsi], ymm3
								ADD rsi, 32
								VMOVUPS[rsi], ymm4
								ADD rsi, 32
								VMOVUPS[rsi], ymm5
								ADD rsi, 32
								VMOVUPS[rsi], ymm6
								ADD rsi, 32
								VMOVUPS[rsi], ymm7
								ADD rsi, 32
								VMOVUPS[rsi], ymm8
								ADD rsi, 32
								VMOVUPS[rsi], ymm9
								ADD rsi, 32
								VMOVUPS[rsi], ymm10
								ADD rsi, 32
								VMOVUPS[rsi], ymm11
							}

							float s11, s12, s13, s14, s21, s22, s23, s24, s31, s32, s33, s34;
							s11 = s12 = s13 = s14 = s21 = s22 = s23 = s24 = s31 = s32 = s33 = s34 = 0;
							for (int c = C8; c < C; c++) {
								float w1 = pImg[0 * CS + c];
								float w2 = pImg[1 * CS + c];
								float w3 = pImg[2 * CS + c];

								float k1 = pKern[0 * C + c];
								float k2 = pKern[1 * C + c];
								float k3 = pKern[2 * C + c];
								float k4 = pKern[3 * C + c];

								s11 += w1 * k1;
								s12 += w1 * k2;
								s13 += w1 * k3;
								s14 += w1 * k4;
								s21 += w2 * k1;
								s22 += w2 * k2;
								s23 += w2 * k3;
								s24 += w2 * k4;
								s31 += w3 * k1;
								s32 += w3 * k2;
								s33 += w3 * k3;
								s34 += w3 * k4;
							}
							float* t = mt;
							pOut[0 * N + 0] += s11 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[1 * N + 0] += s21 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[2 * N + 0] += s31 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[0 * N + 1] += s12 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[1 * N + 1] += s22 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[2 * N + 1] += s32 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[0 * N + 2] += s13 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[1 * N + 2] += s23 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[2 * N + 2] += s33 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[0 * N + 3] += s14 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[1 * N + 3] += s24 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[2 * N + 3] += s34 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
						}
						for (int n = Nh; n < N; n++) {

							float* pImg = img + (b * HWC + h * WC + w * C);
							float* pKern = kern + (x * KyNC + y * NC + n * C);
							float* pOut = out + (b * NHNWN + nh * NWN + nw * N + n);

							__asm {

								VXORPS ymm0, ymm0, ymm0
								VXORPS ymm1, ymm1, ymm1
								VXORPS ymm2, ymm2, ymm2

								MOV r12, CS4
								MOV r13, C4

								MOV rsi, pImg
								MOV rdi, pKern

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									MOV rdx, rsi
									VMOVUPS ymm12, [rsi]
									ADD rsi, r12
									VMOVUPS ymm13, [rsi]
									ADD rsi, r12
									VMOVUPS ymm14, [rsi]
									MOV rsi, rdx
									ADD rsi, 32

									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm0, ymm15, ymm12
									VFMADD231PS ymm1, ymm15, ymm13
									VFMADD231PS ymm2, ymm15, ymm14
									ADD rdi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

								MOV rsi, mt
									VMOVUPS[rsi], ymm0
									ADD rsi, 32
									VMOVUPS[rsi], ymm1
									ADD rsi, 32
									VMOVUPS[rsi], ymm2

							}

							float s1, s2, s3;
							s1 = s2 = s3 = 0;
							for (int c = C8; c < C; c++) {
								float k1 = pKern[c];
								s1 += pImg[0 * CS + c] * k1;
								s2 += pImg[1 * CS + c] * k1;
								s3 += pImg[2 * CS + c] * k1;
							}
							float* t = mt;
							pOut[0 * N] += s1 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[1 * N] += s2 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[2 * N] += s3 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
						}
					}
					for (int nw = NWh; nw < NW; nw++) {
						int w = nw * stride + y;
						for (int n = 0; n < Nh; n += 4) {

							float* pImg = img + (b * HWC + h * WC + w * C);
							float* pKern = kern + (x * KyNC + y * NC + n * C);
							float* pOut = out + (b * NHNWN + nh * NWN + nw * N + n);

							__asm {

								VXORPS ymm0, ymm0, ymm0
								VXORPS ymm1, ymm1, ymm1
								VXORPS ymm2, ymm2, ymm2
								VXORPS ymm3, ymm3, ymm3

								MOV r12, CS4
								MOV r13, C4

								MOV rsi, pImg
								MOV rdi, pKern

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									VMOVUPS ymm12, [rsi]
									ADD rsi, 32

									MOV rdx, rdi
									VMOVUPS ymm15, [rdi]
									ADD rdi, r13
									VFMADD231PS ymm0, ymm15, ymm12
									VMOVUPS ymm15, [rdi]
									ADD rdi, r13
									VFMADD231PS ymm1, ymm15, ymm12
									VMOVUPS ymm15, [rdi]
									ADD rdi, r13
									VFMADD231PS ymm2, ymm15, ymm12
									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm3, ymm15, ymm12
									MOV rdi, rdx
									ADD rdi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

								MOV rsi, mt
									VMOVUPS[rsi], ymm0
									ADD rsi, 32
									VMOVUPS[rsi], ymm1
									ADD rsi, 32
									VMOVUPS[rsi], ymm2
									ADD rsi, 32
									VMOVUPS[rsi], ymm3
							}

							float s11, s12, s13, s14;
							s11 = s12 = s13 = s14 = 0;
							for (int c = C8; c < C; c++) {
								float w1 = pImg[0 * CS + c];

								float k1 = pKern[0 * C + c];
								float k2 = pKern[1 * C + c];
								float k3 = pKern[2 * C + c];
								float k4 = pKern[3 * C + c];

								s11 += w1 * k1;
								s12 += w1 * k2;
								s13 += w1 * k3;
								s14 += w1 * k4;
							}
							float* t = mt;
							pOut[0] += s11 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[1] += s12 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[2] += s13 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[3] += s14 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
						}
						for (int n = Nh; n < N; n++) {

							float* pImg = img + (b * HWC + h * WC + w * C);
							float* pKern = kern + (x * KyNC + y * NC + n * C);
							float* pOut = out + (b * NHNWN + nh * NWN + nw * N + n);

							__asm {

								VXORPS ymm0, ymm0, ymm0

								MOV r12, CS4
								MOV r13, C4

								MOV rsi, pImg
								MOV rdi, pKern

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									VMOVUPS ymm12, [rsi]
									ADD rsi, 32

									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm0, ymm15, ymm12
									ADD rdi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

								MOV rsi, mt
									VMOVUPS[rsi], ymm0
							}

							float s11;
							s11 = 0;
							for (int c = C8; c < C; c++) {
								float w1 = pImg[0 * CS + c];

								float k1 = pKern[0 * C + c];

								s11 += w1 * k1;
							}
							float* t = mt;
							pOut[0] += s11 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
						}
					}
				}
			}
		}
		_aligned_free(mt);
	}
}


DLLEXPORT void avx2_conv3d_back_i(float* img, float* kern, float* out, int H, int W, int C, int N, int Kx, int Ky, int stride, int padding, int batchsize) {

	int NH = (H - Kx) / stride + 1;
	int NW = (W - Ky) / stride + 1;

	memset(img, 0, sizeof(float) * batchsize * C * H * W);

	int NWh = (NW / 3) * 3;
	int Nh = (N / 4) * 4;
	int C8 = (C / 8) * 8;

	int NWN = NW * N;
	int NHNWN = NH * NWN;
	int NC = N * C;
	int WC = W * C;
	int KyNC = Ky * NC;
	int HWC = H * WC;

	long long C4 = C * 4;
	long long CS4 = C4 * stride;
	long long N4 = N * 4;
	int CS = C * stride;

	/*
	#pragma omp parallel for
	for (int c = 0; c < C; c++) {
		for (int nh = 0; nh < NH; nh++) {
			for (int b = 0; b < batchsize; b++) {
				for (int x = 0; x < Kx; x++) {
					int h = nh * stride + x;
					for (int nw = 0; nw < NW; nw++) {
						for (int y = 0; y < Ky; y++) {
							int w = nw * stride + y;
							for (int n = 0; n < N; n++) {
								img[b * HWC + h * WC + w * C + c] += kern[x * KyNC + y * NC + n * C + c] * out[b * NHNWN + nh * NWN + nw * N + n];
							}
						}
					}
				}
			}
		}
	}
	*/

	#pragma omp parallel for
	for (int h = 0; h < H; h++) {
		int xl = h % stride;
		int t = h - (NH - 1) * stride;
		xl = (t < xl) ? xl : t;
		int xh = (h + 1 < Kx) ? h + 1 : Kx;
		for (int b = 0; b < batchsize; b++) {
			for (int x = xl; x < xh; x += stride) {
				int nh = (h - x) / stride;
				for (int y = 0; y < Ky; y++) {
					for (int nw = 0; nw < NWh; nw += 3) {
						int w = nw * stride + y;

						for (int n = 0; n < Nh; n += 4) {

							float* pImg = img + (b * HWC + h * WC + w * C);
							float* pKern = kern + (x * KyNC + y * NC + n * C);
							float* pOut = out + (b * NHNWN + nh * NWN + nw * N + n);

							__asm {

								MOV r12, N4

								MOV rsi, pOut
								VBROADCASTSS ymm0, [rsi + 0]
								VBROADCASTSS ymm1, [rsi + 4]
								VBROADCASTSS ymm2, [rsi + 8]
								VBROADCASTSS ymm3, [rsi + 12]
								ADD rsi, r12
								VBROADCASTSS ymm4, [rsi + 0]
								VBROADCASTSS ymm5, [rsi + 4]
								VBROADCASTSS ymm6, [rsi + 8]
								VBROADCASTSS ymm7, [rsi + 12]
								ADD rsi, r12
								VBROADCASTSS ymm8, [rsi + 0]
								VBROADCASTSS ymm9, [rsi + 4]
								VBROADCASTSS ymm10, [rsi + 8]
								VBROADCASTSS ymm11, [rsi + 12]

								MOV r12, CS4
								MOV r13, C4

								MOV rsi, pImg
								MOV rdi, pKern

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									MOV rdx, rsi
									VMOVUPS ymm12, [rsi]
									ADD rsi, r12
									VMOVUPS ymm13, [rsi]
									ADD rsi, r12
									VMOVUPS ymm14, [rsi]
									MOV rsi, rdx

									MOV rdx, rdi
									VMOVUPS ymm15, [rdi]
									ADD rdi, r13
									VFMADD231PS ymm12, ymm15, ymm0
									VFMADD231PS ymm13, ymm15, ymm4
									VFMADD231PS ymm14, ymm15, ymm8
									VMOVUPS ymm15, [rdi]
									ADD rdi, r13
									VFMADD231PS ymm12, ymm15, ymm1
									VFMADD231PS ymm13, ymm15, ymm5
									VFMADD231PS ymm14, ymm15, ymm9
									VMOVUPS ymm15, [rdi]
									ADD rdi, r13
									VFMADD231PS ymm12, ymm15, ymm2
									VFMADD231PS ymm13, ymm15, ymm6
									VFMADD231PS ymm14, ymm15, ymm10
									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm12, ymm15, ymm3
									VFMADD231PS ymm13, ymm15, ymm7
									VFMADD231PS ymm14, ymm15, ymm11
									MOV rdi, rdx
									ADD rdi, 32

									MOV rdx, rsi
									VMOVUPS [rsi], ymm12
									ADD rsi, r12
									VMOVUPS [rsi], ymm13
									ADD rsi, r12
									VMOVUPS [rsi], ymm14
									MOV rsi, rdx
									ADD rsi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

							}

							float o11 = pOut[0 * N + 0];
							float o12 = pOut[0 * N + 1];
							float o13 = pOut[0 * N + 2];
							float o14 = pOut[0 * N + 3];
							float o21 = pOut[1 * N + 0];
							float o22 = pOut[1 * N + 1];
							float o23 = pOut[1 * N + 2];
							float o24 = pOut[1 * N + 3];
							float o31 = pOut[2 * N + 0];
							float o32 = pOut[2 * N + 1];
							float o33 = pOut[2 * N + 2];
							float o34 = pOut[2 * N + 3];

							for (int c = C8; c < C; c++) {
								float k1 = pKern[0 * C + c];
								float k2 = pKern[1 * C + c];
								float k3 = pKern[2 * C + c];
								float k4 = pKern[3 * C + c];

								pImg[0 * CS + c] += o11 * k1 + o12 * k2 + o13 * k3 + o14 * k4;
								pImg[1 * CS + c] += o21 * k1 + o22 * k2 + o23 * k3 + o24 * k4;
								pImg[2 * CS + c] += o31 * k1 + o32 * k2 + o33 * k3 + o34 * k4;
							}
						}
						for (int n = Nh; n < N; n++) {
							float* pImg = img + (b * HWC + h * WC + w * C);
							float* pKern = kern + (x * KyNC + y * NC + n * C);
							float* pOut = out + (b * NHNWN + nh * NWN + nw * N + n);

							__asm {

								MOV r12, N4

								MOV rsi, pOut
								VBROADCASTSS ymm0, [rsi + 0]
								ADD rsi, r12
								VBROADCASTSS ymm4, [rsi + 0]
								ADD rsi, r12
								VBROADCASTSS ymm8, [rsi + 0]

								MOV r12, CS4
								MOV r13, C4

								MOV rsi, pImg
								MOV rdi, pKern

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									VMOVUPS ymm15, [rdi]
									ADD rdi, 32

									MOV rdx, rsi
									VMOVUPS ymm12, [rsi]
									VFMADD231PS ymm12, ymm15, ymm0
									VMOVUPS [rsi], ymm12
									ADD rsi, r12
									VMOVUPS ymm13, [rsi]
									VFMADD231PS ymm13, ymm15, ymm4
									VMOVUPS [rsi], ymm13
									ADD rsi, r12
									VMOVUPS ymm14, [rsi]
									VFMADD231PS ymm14, ymm15, ymm8
									VMOVUPS [rsi], ymm14
									MOV rsi, rdx
									ADD rsi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

							}

							float o11 = pOut[0 * N];
							float o21 = pOut[1 * N];
							float o31 = pOut[2 * N];							
							for (int c = C8; c < C; c++) {
								float k1 = pKern[c];

								pImg[0 * CS + c] += o11 * k1;
								pImg[1 * CS + c] += o21 * k1;
								pImg[2 * CS + c] += o31 * k1;
							}
						}
					}
					for (int nw = NWh; nw < NW; nw++) {
						int w = nw * stride + y;

						for (int n = 0; n < Nh; n += 4) {
							float* pImg = img + (b * HWC + h * WC + w * C);
							float* pKern = kern + (x * KyNC + y * NC + n * C);
							float* pOut = out + (b * NHNWN + nh * NWN + nw * N + n);

							__asm {

								MOV rsi, pOut
								VBROADCASTSS ymm0, [rsi + 0]
								VBROADCASTSS ymm1, [rsi + 4]
								VBROADCASTSS ymm2, [rsi + 8]
								VBROADCASTSS ymm3, [rsi + 12]

								MOV r12, CS4
								MOV r13, C4

								MOV rsi, pImg
								MOV rdi, pKern

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									VMOVUPS ymm12, [rsi]

									MOV rdx, rdi
									VMOVUPS ymm15, [rdi]
									ADD rdi, r13
									VFMADD231PS ymm12, ymm15, ymm0
									VMOVUPS ymm15, [rdi]
									ADD rdi, r13
									VFMADD231PS ymm12, ymm15, ymm1
									VMOVUPS ymm15, [rdi]
									ADD rdi, r13
									VFMADD231PS ymm12, ymm15, ymm2
									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm12, ymm15, ymm3
									MOV rdi, rdx
									ADD rdi, 32

									VMOVUPS[rsi], ymm12
									ADD rsi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

							}


							float o11 = pOut[0];
							float o12 = pOut[1];
							float o13 = pOut[2];
							float o14 = pOut[3];
							for (int c = C8; c < C; c++) {
								float k1 = pKern[0 * C + c];
								float k2 = pKern[1 * C + c];
								float k3 = pKern[2 * C + c];
								float k4 = pKern[3 * C + c];

								pImg[c] += o11 * k1 + o12 * k2 + o13 * k3 + o14 * k4;
							}
						}
						for (int n = Nh; n < N; n++) {
							float* pImg = img + (b * HWC + h * WC + w * C);
							float* pKern = kern + (x * KyNC + y * NC + n * C);
							float* pOut = out + (b * NHNWN + nh * NWN + nw * N + n);

							__asm {

								MOV rsi, pOut
								VBROADCASTSS ymm0, [rsi]

								MOV r12, CS4
								MOV r13, C4

								MOV rsi, pImg
								MOV rdi, pKern

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									VMOVUPS ymm12, [rsi]

									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm12, ymm15, ymm0
									ADD rdi, 32

									VMOVUPS[rsi], ymm12
									ADD rsi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

							}

							float o11 = pOut[0];
							for (int c = C8; c < C; c++) {
								float k1 = pKern[c];

								pImg[c] += o11 * k1;
							}
						}
					}
				}
			}
		}
	}
}


DLLEXPORT void avx2_conv3d_back_k(float* img, float* kern, float* out, int H, int W, int C, int N, int Kx, int Ky, int stride, int padding, int batchsize) {

	int NH = (H - Kx) / stride + 1;
	int NW = (W - Ky) / stride + 1;

	memset(kern, 0, sizeof(float) * Kx * Ky * N * C);

	int NWh = (NW / 3) * 3;
	int C8 = (C / 8) * 8;

	int WC = W * C;
	int HWC = H * WC;
	int NWN = NW * N;
	int NHNWN = NH * NWN;

	int NC = N * C;
	int KyNC = Ky * NC;

	long long C4 = C * 4;
	long long CS4 = stride * C4;
	long long N4 = N * 4;
	int CS = C * stride;

	int Nh = (N / 4) * 4;
	int Nm = ceil((float)N / 4);
	int T = Kx * Ky * Nm;
	#pragma omp parallel for
	for (int t = 0; t < T; t++) {
		int x = t / (Ky * Nm);
		int t1 = t % (Ky * Nm);
		int y = t1 / Nm;
		int n = (t1 % Nm) * 4;
		int nl = n;

		if (n + 4 <= N) {
			for (int b = 0; b < batchsize; b++) {
				for (int nh = 0; nh < NH; nh++) {
					int h = nh * stride + x;
					for (int nw = 0; nw < NWh; nw += 3) {
						int w = nw * stride + y;

						float* pImg = img + b * HWC + h * WC + w * C;
						float* pKern = kern + x * KyNC + y * NC + n * C;
						float* pOut = out + b * NHNWN + nh * NWN + nw * N + n;

						__asm {

							MOV r12, N4

							MOV rsi, pOut
							VBROADCASTSS ymm0, [rsi + 0]
							VBROADCASTSS ymm1, [rsi + 4]
							VBROADCASTSS ymm2, [rsi + 8]
							VBROADCASTSS ymm3, [rsi + 12]
							ADD rsi, r12
							VBROADCASTSS ymm4, [rsi + 0]
							VBROADCASTSS ymm5, [rsi + 4]
							VBROADCASTSS ymm6, [rsi + 8]
							VBROADCASTSS ymm7, [rsi + 12]
							ADD rsi, r12
							VBROADCASTSS ymm8, [rsi + 0]
							VBROADCASTSS ymm9, [rsi + 4]
							VBROADCASTSS ymm10, [rsi + 8]
							VBROADCASTSS ymm11, [rsi + 12]

							MOV r12, C4
							MOV r13, CS4

							MOV rsi, pImg
							MOV rdi, pKern

							MOV ecx, C8
							cloops :
							CMP ecx, 0
								JE cloope

								MOV rdx, rsi
								VMOVUPS ymm12, [rsi]
								ADD rsi, r13
								VMOVUPS ymm13, [rsi]
								ADD rsi, r13
								VMOVUPS ymm14, [rsi]
								MOV rsi, rdx
								ADD rsi, 32

								MOV rdx, rdi
								VMOVUPS ymm15, [rdi]
								VFMADD231PS ymm15, ymm0, ymm12
								VFMADD231PS ymm15, ymm4, ymm13
								VFMADD231PS ymm15, ymm8, ymm14
								VMOVUPS[rdi], ymm15
								ADD rdi, r12

								VMOVUPS ymm15, [rdi]
								VFMADD231PS ymm15, ymm1, ymm12
								VFMADD231PS ymm15, ymm5, ymm13
								VFMADD231PS ymm15, ymm9, ymm14
								VMOVUPS[rdi], ymm15
								ADD rdi, r12

								VMOVUPS ymm15, [rdi]
								VFMADD231PS ymm15, ymm2, ymm12
								VFMADD231PS ymm15, ymm6, ymm13
								VFMADD231PS ymm15, ymm10, ymm14
								VMOVUPS[rdi], ymm15
								ADD rdi, r12

								VMOVUPS ymm15, [rdi]
								VFMADD231PS ymm15, ymm3, ymm12
								VFMADD231PS ymm15, ymm7, ymm13
								VFMADD231PS ymm15, ymm11, ymm14
								VMOVUPS[rdi], ymm15

								MOV rdi, rdx
								ADD rdi, 32

								SUB ecx, 8
								JMP cloops
								cloope :
						}

						float o11 = pOut[0 * N + 0];
						float o12 = pOut[0 * N + 1];
						float o13 = pOut[0 * N + 2];
						float o14 = pOut[0 * N + 3];
						float o21 = pOut[1 * N + 0];
						float o22 = pOut[1 * N + 1];
						float o23 = pOut[1 * N + 2];
						float o24 = pOut[1 * N + 3];
						float o31 = pOut[2 * N + 0];
						float o32 = pOut[2 * N + 1];
						float o33 = pOut[2 * N + 2];
						float o34 = pOut[2 * N + 3];

						for (int c = C8; c < C; c++) {
							float i1 = pImg[0 * CS + c];
							float i2 = pImg[1 * CS + c];
							float i3 = pImg[2 * CS + c];

							pKern[0 * C + c] += i1 * o11 + i2 * o21 + i3 * o31;
							pKern[1 * C + c] += i1 * o12 + i2 * o22 + i3 * o32;
							pKern[2 * C + c] += i1 * o13 + i2 * o23 + i3 * o33;
							pKern[3 * C + c] += i1 * o14 + i2 * o24 + i3 * o34;
						}
					}
					for (int nw = NWh; nw < NW; nw++) {
						int w = nw * stride + y;

						float* pImg = img + b * HWC + h * WC + w * C;
						float* pKern = kern + x * KyNC + y * NC + n * C;
						float* pOut = out + b * NHNWN + nh * NWN + nw * N + n;


						__asm {

							MOV r12, C4

							MOV rsi, pOut
							VBROADCASTSS ymm0, [rsi]
							VBROADCASTSS ymm1, [rsi + 4]
							VBROADCASTSS ymm2, [rsi + 8]
							VBROADCASTSS ymm3, [rsi + 12]

							MOV rsi, pImg
							MOV rdi, pKern

							MOV ecx, C8
							cloops :
							CMP ecx, 0
								JE cloope

								VMOVUPS ymm4, [rsi]
								ADD rsi, 32

								MOV rdx, rdi
								VMOVUPS ymm5, [rdi]
								VFMADD231PS ymm5, ymm4, ymm0
								VMOVUPS[rdi], ymm5
								ADD rdi, r12

								VMOVUPS ymm5, [rdi]
								VFMADD231PS ymm5, ymm4, ymm1
								VMOVUPS[rdi], ymm5
								ADD rdi, r12

								VMOVUPS ymm5, [rdi]
								VFMADD231PS ymm5, ymm4, ymm2
								VMOVUPS[rdi], ymm5
								ADD rdi, r12

								VMOVUPS ymm5, [rdi]
								VFMADD231PS ymm5, ymm4, ymm3
								VMOVUPS[rdi], ymm5
								MOV rdi, rdx
								ADD rdi, 32

								SUB ecx, 8
								JMP cloops
								cloope :

						}

						float o1 = pOut[0];
						float o2 = pOut[1];
						float o3 = pOut[2];
						float o4 = pOut[3];
						for (int c = C8; c < C; c++) {
							float i1 = pImg[c];

							pKern[0 * C + c] += i1 * o1;
							pKern[1 * C + c] += i1 * o2;
							pKern[2 * C + c] += i1 * o3;
							pKern[3 * C + c] += i1 * o4;
						}
					}
				}
			}
		}
		else {
			for (int b = 0; b < batchsize; b++) {
				for (int nh = 0; nh < NH; nh++) {
					int h = nh * stride + x;
					for (int nw = 0; nw < NWh; nw += 3) {
						int w = nw * stride + y;
						for (int n = nl; n < N; n++) {

							float* pImg = img + b * HWC + h * WC + w * C;
							float* pKern = kern + x * KyNC + y * NC + n * C;
							float* pOut = out + b * NHNWN + nh * NWN + nw * N + n;

							__asm {

								MOV r12, N4

								MOV rsi, pOut
								VBROADCASTSS ymm0, [rsi + 0]
								ADD rsi, r12
								VBROADCASTSS ymm4, [rsi + 0]
								ADD rsi, r12
								VBROADCASTSS ymm8, [rsi + 0]

								MOV r12, C4
								MOV r13, CS4

								MOV rsi, pImg
								MOV rdi, pKern

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									MOV rdx, rsi
									VMOVUPS ymm12, [rsi]
									ADD rsi, r13
									VMOVUPS ymm13, [rsi]
									ADD rsi, r13
									VMOVUPS ymm14, [rsi]
									MOV rsi, rdx
									ADD rsi, 32

									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm15, ymm0, ymm12
									VFMADD231PS ymm15, ymm4, ymm13
									VFMADD231PS ymm15, ymm8, ymm14
									VMOVUPS[rdi], ymm15
									ADD rdi, 32

									SUB ecx, 8
									JMP cloops
									cloope :
							}

							float o11 = pOut[0 * N + 0];
							float o21 = pOut[1 * N + 0];
							float o31 = pOut[2 * N + 0];

							for (int c = C8; c < C; c++) {
								float i1 = pImg[0 * CS + c];
								float i2 = pImg[1 * CS + c];
								float i3 = pImg[2 * CS + c];

								pKern[0 * C + c] += i1 * o11 + i2 * o21 + i3 * o31;
							}
						}
					}
					for (int nw = NWh; nw < NW; nw++) {
						int w = nw * stride + y;
						for (int n = nl; n < N; n++) {

							float* pImg = img + b * HWC + h * WC + w * C;
							float* pKern = kern + x * KyNC + y * NC + n * C;
							float* pOut = out + b * NHNWN + nh * NWN + nw * N + n;

							__asm {

								MOV rsi, pOut
								VBROADCASTSS ymm0, [rsi]

								MOV rsi, pImg
								MOV rdi, pKern

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									VMOVUPS ymm4, [rsi]
									ADD rsi, 32

									VMOVUPS ymm5, [rdi]
									VFMADD231PS ymm5, ymm4, ymm0
									VMOVUPS[rdi], ymm5
									ADD rdi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

							}

							float o1 = pOut[0];
							for (int c = C8; c < C; c++) {
								float i1 = pImg[c];

								pKern[0 * C + c] += i1 * o1;
							}
						}
					}
				}
			}
		}
	}
}






DLLEXPORT void avx2_conv3dtranspose(float* img, float* kern, float* out, int H, int W, int C, int N, int Kx, int Ky, int stride, int padding, int batchsize) {

	int NH = (H - 1) * stride + Kx;
	int NW = (W - 1) * stride + Ky;

	memset(out, 0, sizeof(float) * batchsize * NH * NW * N);

	int Nh = (N / 4) * 4;
	int Wh = (W / 3) * 3;
	int C8 = (C / 8) * 8;

	int WC = W * C;
	int HWC = H * WC;
	int NC = N * C;
	int KyNC = Ky * NC;
	int NWN = NW * N;
	int NHNWN = NH * NWN;

	long long C4 = C * 4;
	int NS = N * stride;
	long long NS4 = NS * 4;

	/*
	#pragma omp parallel for
	for (int n = 0; n < N; n++) {
		for (int b = 0; b < batchsize; b++) {
			for (int h = 0; h < H; h++) {
				for (int x = 0; x < Kx; x++) {
					int nh = h * stride + x;
					for (int y = 0; y < Ky; y++) {
						for (int w = 0; w < Wh; w += 3) {
							int nw = w * stride + y;

							float* pImg = img + (b * HWC + h * WC + w * C);
							float* pKern = kern + (x * KyNC + y * NC + n * C);
							float* pOut = out + (b * NHNWN + nh * NWN + nw * N + n);

							float s1, s2, s3;
							s1 = s2 = s3 = 0;
							for (int c = 0; c < C; c++) {
								float k1 = pKern[c];

								s1 += k1 * pImg[0 * C + c];
								s2 += k1 * pImg[1 * C + c];
								s3 += k1 * pImg[2 * C + c];
							}
							pOut[0 * NS] += s1;
							pOut[1 * NS] += s2;
							pOut[2 * NS] += s3;
						}
						for (int w = Wh; w < W; w++) {
							int nw = w * stride + y;

							float* pImg = img + (b * HWC + h * WC + w * C);
							float* pKern = kern + (x * KyNC + y * NC + n * C);
							float* pOut = out + (b * NHNWN + nh * NWN + nw * N + n);

							float s1 = 0;
							for (int c = 0; c < C; c++) {
								float k1 = pKern[c];

								s1 += k1 * pImg[0 * C + c];
							}
							pOut[0 * NS] += s1;
						}
					}
				}
			}
		}
	}
	*/

	#pragma omp parallel for
	for (int nh = 0; nh < NH; nh++) {
		float* mt = (float*)_aligned_malloc(sizeof(float) * 8 * 12, 32);
		int xl = nh % stride;
		int t = nh - (H - 1) * stride;
		xl = (t < xl) ? xl : t;
		int xh = (nh + 1 < Kx) ? nh + 1 : Kx;
		for (int b = 0; b < batchsize; b++) {
			for (int x = xl; x < xh; x += stride) {
				int h = (nh - x) / stride;
				for (int w = 0; w < Wh; w += 3) {
					for (int y = 0; y < Ky; y++) {
						int nw = w * stride + y;
						for (int n = 0; n < Nh; n += 4) {

							float* pImg = img + b * HWC + h * WC + w * C;
							float* pKern = kern + x * KyNC + y * NC + n * C;
							float* pOut = out + b * NHNWN + nh * NWN + nw * N + n;

							__asm {

								VXORPS ymm0, ymm0, ymm0
								VXORPS ymm1, ymm1, ymm1
								VXORPS ymm2, ymm2, ymm2
								VXORPS ymm3, ymm3, ymm3
								VXORPS ymm4, ymm4, ymm4
								VXORPS ymm5, ymm5, ymm5
								VXORPS ymm6, ymm6, ymm6
								VXORPS ymm7, ymm7, ymm7
								VXORPS ymm8, ymm8, ymm8
								VXORPS ymm9, ymm9, ymm9
								VXORPS ymm10, ymm10, ymm10
								VXORPS ymm11, ymm11, ymm11

								MOV rsi, pImg
								MOV rdi, pKern

								MOV r12, C4

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									MOV rdx, rsi
									VMOVUPS ymm12, [rsi]
									ADD rsi, r12
									VMOVUPS ymm13, [rsi]
									ADD rsi, r12
									VMOVUPS ymm14, [rsi]
									MOV rsi, rdx
									ADD rsi, 32

									MOV rdx, rdi
									VMOVUPS ymm15, [rdi]
									ADD rdi, r12
									VFMADD231PS ymm0, ymm15, ymm12
									VFMADD231PS ymm1, ymm15, ymm13
									VFMADD231PS ymm2, ymm15, ymm14
									VMOVUPS ymm15, [rdi]
									ADD rdi, r12
									VFMADD231PS ymm3, ymm15, ymm12
									VFMADD231PS ymm4, ymm15, ymm13
									VFMADD231PS ymm5, ymm15, ymm14
									VMOVUPS ymm15, [rdi]
									ADD rdi, r12
									VFMADD231PS ymm6, ymm15, ymm12
									VFMADD231PS ymm7, ymm15, ymm13
									VFMADD231PS ymm8, ymm15, ymm14
									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm9, ymm15, ymm12
									VFMADD231PS ymm10, ymm15, ymm13
									VFMADD231PS ymm11, ymm15, ymm14
									MOV rdi, rdx
									ADD rdi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

								MOV rsi, mt
									VMOVUPS[rsi], ymm0
									ADD rsi, 32
									VMOVUPS[rsi], ymm1
									ADD rsi, 32
									VMOVUPS[rsi], ymm2
									ADD rsi, 32
									VMOVUPS[rsi], ymm3
									ADD rsi, 32
									VMOVUPS[rsi], ymm4
									ADD rsi, 32
									VMOVUPS[rsi], ymm5
									ADD rsi, 32
									VMOVUPS[rsi], ymm6
									ADD rsi, 32
									VMOVUPS[rsi], ymm7
									ADD rsi, 32
									VMOVUPS[rsi], ymm8
									ADD rsi, 32
									VMOVUPS[rsi], ymm9
									ADD rsi, 32
									VMOVUPS[rsi], ymm10
									ADD rsi, 32
									VMOVUPS[rsi], ymm11
							}

							float o11, o12, o13, o14, o21, o22, o23, o24, o31, o32, o33, o34;
							o11 = o12 = o13 = o14 = o21 = o22 = o23 = o24 = o31 = o32 = o33 = o34 = 0;
							for (int c = C8; c < C; c++) {
								float w1 = pImg[0 * C + c];
								float w2 = pImg[1 * C + c];
								float w3 = pImg[2 * C + c];

								float n1 = pKern[0 * C + c];
								float n2 = pKern[1 * C + c];
								float n3 = pKern[2 * C + c];
								float n4 = pKern[3 * C + c];

								o11 += w1 * n1;
								o12 += w1 * n2;
								o13 += w1 * n3;
								o14 += w1 * n4;
								o21 += w2 * n1;
								o22 += w2 * n2;
								o23 += w2 * n3;
								o24 += w2 * n4;
								o31 += w3 * n1;
								o32 += w3 * n2;
								o33 += w3 * n3;
								o34 += w3 * n4;
							}
							float* t = mt;
							pOut[0 * NS + 0] += o11 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[1 * NS + 0] += o21 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[2 * NS + 0] += o31 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[0 * NS + 1] += o12 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[1 * NS + 1] += o22 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[2 * NS + 1] += o32 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[0 * NS + 2] += o13 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[1 * NS + 2] += o23 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[2 * NS + 2] += o33 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[0 * NS + 3] += o14 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[1 * NS + 3] += o24 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[2 * NS + 3] += o34 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
						}
						for (int n = Nh; n < N; n++) {

							float* pImg = img + b * HWC + h * WC + w * C;
							float* pKern = kern + x * KyNC + y * NC + n * C;
							float* pOut = out + b * NHNWN + nh * NWN + nw * N + n;

							__asm {

								VXORPS ymm0, ymm0, ymm0
								VXORPS ymm1, ymm1, ymm1
								VXORPS ymm2, ymm2, ymm2

								MOV rsi, pImg
								MOV rdi, pKern

								MOV r12, C4

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									MOV rdx, rsi
									VMOVUPS ymm12, [rsi]
									ADD rsi, r12
									VMOVUPS ymm13, [rsi]
									ADD rsi, r12
									VMOVUPS ymm14, [rsi]
									MOV rsi, rdx
									ADD rsi, 32

									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm0, ymm15, ymm12
									VFMADD231PS ymm1, ymm15, ymm13
									VFMADD231PS ymm2, ymm15, ymm14
									ADD rdi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

								MOV rsi, mt
									VMOVUPS[rsi], ymm0
									ADD rsi, 32
									VMOVUPS[rsi], ymm1
									ADD rsi, 32
									VMOVUPS[rsi], ymm2
							}

							float o11, o21, o31;
							o11 = o21 = o31 = 0;
							for (int c = C8; c < C; c++) {
								float w1 = pImg[0 * C + c];
								float w2 = pImg[1 * C + c];
								float w3 = pImg[2 * C + c];

								float n1 = pKern[0 * C + c];

								o11 += w1 * n1;
								o21 += w2 * n1;
								o31 += w3 * n1;
							}
							float* t = mt;
							pOut[0 * NS + 0] += o11 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[1 * NS + 0] += o21 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[2 * NS + 0] += o31 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
						}
					}
				}
				for (int w = Wh; w < W; w++) {
					for (int y = 0; y < Ky; y++) {
						int nw = w * stride + y;
						for (int n = 0; n < Nh; n += 4) {

							float* pImg = img + b * HWC + h * WC + w * C;
							float* pKern = kern + x * KyNC + y * NC + n * C;
							float* pOut = out + b * NHNWN + nh * NWN + nw * N + n;

							__asm {

								VXORPS ymm0, ymm0, ymm0
								VXORPS ymm1, ymm1, ymm1
								VXORPS ymm2, ymm2, ymm2
								VXORPS ymm3, ymm3, ymm3

								MOV rsi, pImg
								MOV rdi, pKern

								MOV r12, C4

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									VMOVUPS ymm12, [rsi]
									ADD rsi, 32

									MOV rdx, rdi
									VMOVUPS ymm15, [rdi]
									ADD rdi, r12
									VFMADD231PS ymm0, ymm15, ymm12
									VMOVUPS ymm15, [rdi]
									ADD rdi, r12
									VFMADD231PS ymm1, ymm15, ymm12
									VMOVUPS ymm15, [rdi]
									ADD rdi, r12
									VFMADD231PS ymm2, ymm15, ymm12
									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm3, ymm15, ymm12
									MOV rdi, rdx
									ADD rdi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

								MOV rsi, mt
									VMOVUPS[rsi], ymm0
									ADD rsi, 32
									VMOVUPS[rsi], ymm1
									ADD rsi, 32
									VMOVUPS[rsi], ymm2
									ADD rsi, 32
									VMOVUPS[rsi], ymm3
							}
							float o11, o12, o13, o14;
							o11 = o12 = o13 = o14 = 0;
							for (int c = C8; c < C; c++) {
								float w1 = pImg[0 * C + c];

								float n1 = pKern[0 * C + c];
								float n2 = pKern[1 * C + c];
								float n3 = pKern[2 * C + c];
								float n4 = pKern[3 * C + c];

								o11 += w1 * n1;
								o12 += w1 * n2;
								o13 += w1 * n3;
								o14 += w1 * n4;
							}
							float* t = mt;
							pOut[0 * NS + 0] += o11 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[0 * NS + 1] += o12 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[0 * NS + 2] += o13 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							t += 8;
							pOut[0 * NS + 3] += o14 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
						}
						for (int n = Nh; n < N; n++) {

							float* pImg = img + b * HWC + h * WC + w * C;
							float* pKern = kern + x * KyNC + y * NC + n * C;
							float* pOut = out + b * NHNWN + nh * NWN + nw * N + n;

							__asm {

								VXORPS ymm0, ymm0, ymm0

								MOV rsi, pImg
								MOV rdi, pKern

								MOV r12, C4

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									VMOVUPS ymm12, [rsi]
									ADD rsi, 32

									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm0, ymm15, ymm12
									ADD rdi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

								MOV rsi, mt
									VMOVUPS[rsi], ymm0
							}
							float o11;
							o11 = 0;
							for (int c = C8; c < C; c++) {
								float w1 = pImg[0 * C + c];

								float n1 = pKern[0 * C + c];

								o11 += w1 * n1;
							}
							float* t = mt;
							pOut[0 * NS + 0] += o11 + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
						}
					}
				}
			}
		}
		_aligned_free(mt);
	}
}


DLLEXPORT void avx2_conv3dtranspose_back_i(float* img, float* kern, float* out, int H, int W, int C, int N, int Kx, int Ky, int stride, int padding, int batchsize) {

	int NH = (H - 1) * stride + Kx;
	int NW = (W - 1) * stride + Ky;

	memset(img, 0, sizeof(float) * batchsize * H * W * C);

	int Nh = (N / 4) * 4;
	int C8 = (C / 8) * 8;
	int Wh = (W / 3) * 3;

	int WC = W * C;
	int HWC = H * WC;
	int NC = N * C;
	int KyNC = Ky * NC;
	int NWN = NW * N;
	int NHNWN = NH * NWN;

	long long C4 = C * 4;
	int NS = N * stride;
	long long NS4 = NS * 4;

	/*
	#pragma omp parallel for
	for (int h = 0; h < H; h++) {
		for (int b = 0; b < batchsize; b++) {
			for (int x = 0; x < Kx; x++) {
				int nh = h * stride + x;
				for (int w = 0; w < W; w++) {
					for (int y = 0; y < Ky; y++) {
						int nw = w * stride + y;
						for (int n = 0; n < N; n++) {
							for (int c = 0; c < C; c++) {
								img[b * HWC + h * WC + w * C + c] += kern[x * KyNC + y * NC + n * C + c] * out[b * NHNWN + nh * NWN + nw * N + n];
							}
						}
					}
				}
			}
		}
	}
	*/

	#pragma omp parallel for
	for (int h = 0; h < H; h++) {
		for (int b = 0; b < batchsize; b++) {
			for (int x = 0; x < Kx; x++) {
				int nh = h * stride + x;
				for (int w = 0; w < Wh; w += 3) {
					for (int y = 0; y < Ky; y++) {
						int nw = w * stride + y;
						for (int n = 0; n < Nh; n += 4) {

							float* pImg = img + b * HWC + h * WC + w * C;
							float* pKern = kern + x * KyNC + y * NC + n * C;
							float* pOut = out + b * NHNWN + nh * NWN + nw * N + n;

							__asm {

								MOV r12, NS4

								MOV rsi, pOut
								VBROADCASTSS ymm0, [rsi + 0]
								VBROADCASTSS ymm1, [rsi + 4]
								VBROADCASTSS ymm2, [rsi + 8]
								VBROADCASTSS ymm3, [rsi + 12]
								ADD rsi, r12
								VBROADCASTSS ymm4, [rsi + 0]
								VBROADCASTSS ymm5, [rsi + 4]
								VBROADCASTSS ymm6, [rsi + 8]
								VBROADCASTSS ymm7, [rsi + 12]
								ADD rsi, r12
								VBROADCASTSS ymm8, [rsi + 0]
								VBROADCASTSS ymm9, [rsi + 4]
								VBROADCASTSS ymm10, [rsi + 8]
								VBROADCASTSS ymm11, [rsi + 12]

								MOV rsi, pImg
								MOV rdi, pKern

								MOV r12, C4

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									MOV rdx, rsi
									VMOVUPS ymm12, [rsi]
									ADD rsi, r12
									VMOVUPS ymm13, [rsi]
									ADD rsi, r12
									VMOVUPS ymm14, [rsi]
									MOV rsi, rdx

									MOV rdx, rdi
									VMOVUPS ymm15, [rdi]
									ADD rdi, r12
									VFMADD231PS ymm12, ymm15, ymm0
									VFMADD231PS ymm13, ymm15, ymm4
									VFMADD231PS ymm14, ymm15, ymm8
									VMOVUPS ymm15, [rdi]
									ADD rdi, r12
									VFMADD231PS ymm12, ymm15, ymm1
									VFMADD231PS ymm13, ymm15, ymm5
									VFMADD231PS ymm14, ymm15, ymm9
									VMOVUPS ymm15, [rdi]
									ADD rdi, r12
									VFMADD231PS ymm12, ymm15, ymm2
									VFMADD231PS ymm13, ymm15, ymm6
									VFMADD231PS ymm14, ymm15, ymm10
									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm12, ymm15, ymm3
									VFMADD231PS ymm13, ymm15, ymm7
									VFMADD231PS ymm14, ymm15, ymm11
									MOV rdi, rdx
									ADD rdi, 32

									MOV rdx, rsi
									VMOVUPS[rsi], ymm12
									ADD rsi, r12
									VMOVUPS[rsi], ymm13
									ADD rsi, r12
									VMOVUPS[rsi], ymm14
									MOV rsi, rdx
									ADD rsi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

							}

							float o11 = pOut[0 * NS + 0];
							float o12 = pOut[0 * NS + 1];
							float o13 = pOut[0 * NS + 2];
							float o14 = pOut[0 * NS + 3];
							float o21 = pOut[1 * NS + 0];
							float o22 = pOut[1 * NS + 1];
							float o23 = pOut[1 * NS + 2];
							float o24 = pOut[1 * NS + 3];
							float o31 = pOut[2 * NS + 0];
							float o32 = pOut[2 * NS + 1];
							float o33 = pOut[2 * NS + 2];
							float o34 = pOut[2 * NS + 3];
							for (int c = C8; c < C; c++) {

								float n1 = pKern[0 * C + c];
								float n2 = pKern[1 * C + c];
								float n3 = pKern[2 * C + c];
								float n4 = pKern[3 * C + c];

								pImg[0 * C + c] += o11 * n1 + o12 * n2 + o13 * n3 + o14 * n4;
								pImg[1 * C + c] += o21 * n1 + o22 * n2 + o23 * n3 + o24 * n4;
								pImg[2 * C + c] += o31 * n1 + o32 * n2 + o33 * n3 + o34 * n4;
							}
						}
						for (int n = Nh; n < N; n++) {

							float* pImg = img + b * HWC + h * WC + w * C;
							float* pKern = kern + x * KyNC + y * NC + n * C;
							float* pOut = out + b * NHNWN + nh * NWN + nw * N + n;

							__asm {

								MOV r12, NS4

								MOV rsi, pOut
								VBROADCASTSS ymm0, [rsi + 0]
								ADD rsi, r12
								VBROADCASTSS ymm4, [rsi + 0]
								ADD rsi, r12
								VBROADCASTSS ymm8, [rsi + 0]

								MOV rsi, pImg
								MOV rdi, pKern

								MOV r12, C4

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									MOV rdx, rsi
									VMOVUPS ymm12, [rsi]
									ADD rsi, r12
									VMOVUPS ymm13, [rsi]
									ADD rsi, r12
									VMOVUPS ymm14, [rsi]
									MOV rsi, rdx

									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm12, ymm15, ymm0
									VFMADD231PS ymm13, ymm15, ymm4
									VFMADD231PS ymm14, ymm15, ymm8
									ADD rdi, 32

									VMOVUPS[rsi], ymm12
									ADD rsi, r12
									VMOVUPS[rsi], ymm13
									ADD rsi, r12
									VMOVUPS[rsi], ymm14
									MOV rsi, rdx
									ADD rsi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

							}

							float o11 = pOut[0 * NS + 0];
							float o21 = pOut[1 * NS + 0];
							float o31 = pOut[2 * NS + 0];
							for (int c = C8; c < C; c++) {

								float n1 = pKern[0 * C + c];

								pImg[0 * C + c] += o11 * n1;
								pImg[1 * C + c] += o21 * n1;
								pImg[2 * C + c] += o31 * n1;
							}
						}
					}
				}
				for (int w = Wh; w < W; w++) {
					for (int y = 0; y < Ky; y++) {
						int nw = w * stride + y;
						for (int n = 0; n < Nh; n += 4) {

							float* pImg = img + b * HWC + h * WC + w * C;
							float* pKern = kern + x * KyNC + y * NC + n * C;
							float* pOut = out + b * NHNWN + nh * NWN + nw * N + n;

							__asm {

								MOV r12, NS4

								MOV rsi, pOut
								VBROADCASTSS ymm0, [rsi + 0]
								VBROADCASTSS ymm1, [rsi + 4]
								VBROADCASTSS ymm2, [rsi + 8]
								VBROADCASTSS ymm3, [rsi + 12]

								MOV rsi, pImg
								MOV rdi, pKern

								MOV r12, C4

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									VMOVUPS ymm12, [rsi]

									MOV rdx, rdi
									VMOVUPS ymm15, [rdi]
									ADD rdi, r12
									VFMADD231PS ymm12, ymm15, ymm0
									VMOVUPS ymm15, [rdi]
									ADD rdi, r12
									VFMADD231PS ymm12, ymm15, ymm1
									VMOVUPS ymm15, [rdi]
									ADD rdi, r12
									VFMADD231PS ymm12, ymm15, ymm2
									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm12, ymm15, ymm3
									MOV rdi, rdx
									ADD rdi, 32

									VMOVUPS[rsi], ymm12
									ADD rsi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

							}

							float o11 = pOut[0 * NS + 0];
							float o12 = pOut[0 * NS + 1];
							float o13 = pOut[0 * NS + 2];
							float o14 = pOut[0 * NS + 3];
							for (int c = C8; c < C; c++) {

								float n1 = pKern[0 * C + c];
								float n2 = pKern[1 * C + c];
								float n3 = pKern[2 * C + c];
								float n4 = pKern[3 * C + c];

								pImg[0 * C + c] += o11 * n1 + o12 * n2 + o13 * n3 + o14 * n4;
							}
						}
						for (int n = Nh; n < N; n++) {

							float* pImg = img + b * HWC + h * WC + w * C;
							float* pKern = kern + x * KyNC + y * NC + n * C;
							float* pOut = out + b * NHNWN + nh * NWN + nw * N + n;

							__asm {

								MOV r12, NS4

								MOV rsi, pOut
								VBROADCASTSS ymm0, [rsi + 0]

								MOV rsi, pImg
								MOV rdi, pKern

								MOV r12, C4

								MOV ecx, C8
								cloops :
								CMP ecx, 0
									JE cloope

									VMOVUPS ymm12, [rsi]

									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm12, ymm15, ymm0
									ADD rdi, 32

									VMOVUPS[rsi], ymm12
									ADD rsi, 32

									SUB ecx, 8
									JMP cloops
									cloope :

							}

							float o11 = pOut[0 * NS + 0];
							for (int c = C8; c < C; c++) {

								float n1 = pKern[0 * C + c];

								pImg[0 * C + c] += o11 * n1;
							}
						}
					}
				}
			}
		}
	}
}


DLLEXPORT void avx2_conv3dtranspose_back_k(float* img, float* kern, float* out, int H, int W, int C, int N, int Kx, int Ky, int stride, int padding, int batchsize) {

	int NH = (H - 1) * stride + Kx;
	int NW = (W - 1) * stride + Ky;

	memset(kern, 0, sizeof(float) * Kx * Ky * N * C);

	int Nh = (N / 4) * 4;
	int C8 = (C / 8) * 8;
	int Wh = (W / 3) * 3;

	int WC = W * C;
	int HWC = H * WC;
	int NWN = NW * N;
	int NHNWN = NH * NWN;
	int NC = N * C;
	int KyNC = Ky * NC;

	int NS = N * stride;
	long long C4 = C * 4;
	long long N4 = N * 4;
	long long NS4 = stride * N4;

	/*
	#pragma omp parallel for
	for (int n = 0; n < N; n++) {
		for (int b = 0; b < batchsize; b++) {
			for (int h = 0; h < H; h++) {
				for (int x = 0; x < Kx; x++) {
					int nh = h * stride + x;
					for (int y = 0; y < Ky; y++) {
						for (int w = 0; w < Wh; w += 3) {
							int nw = w * stride + y;

							float* pKern = kern + (x * KyNC + y * NC + n * C);
							float* pImg = img + (b * HWC + h * WC + w * C);
							float* pOut = out + (b * NHNWN + nh * NWN + nw * N + n);

							float o1 = pOut[0 * NS];
							float o2 = pOut[1 * NS];
							float o3 = pOut[2 * NS];
							for (int c = 0; c < C; c++) {
								pKern[c] += pImg[0 * C + c] * o1 + pImg[1 * C + c] * o2 + pImg[2 * C + c] * o3;
							}
						}
						for (int w = Wh; w < W; w++) {
							int nw = w * stride + y;

							float* pKern = kern + (x * KyNC + y * NC + n * C);
							float* pImg = img + (b * HWC + h * WC + w * C);
							float* pOut = out + (b * NHNWN + nh * NWN + nw * N + n);

							float o1 = pOut[0 * NS];
							for (int c = 0; c < C; c++) {
								pKern[c] += pImg[0 * C + c] * o1;
							}
						}
					}
				}
			}
		}
	}
	*/

	int Nm = ceil((float)N / 4);
	int T = Kx * Ky * Nm;
	#pragma omp parallel for
	for (int t = 0; t < T; t++) {
		int x = t / (Ky * Nm);
		int t1 = t % (Ky * Nm);
		int y = t1 / Nm;
		int n = (t1 % Nm) * 4;
		int nl = n;

		if (n + 4 <= N) {
			for (int b = 0; b < batchsize; b++) {
				for (int h = 0; h < H; h++) {
					int nh = h * stride + x;
					for (int c = 0; c < C8; c += 8) {

						float* pImg = img + b * HWC + h * WC + c;
						float* pKern = kern + x * KyNC + y * NC + n * C + c;
						float* pOut = out + b * NHNWN + nh * NWN + y * N + n;

						__asm {

							MOV r12, C4
							MOV r13, NS4

							MOV rsi, pKern
							VMOVUPS ymm0, [rsi]
							ADD rsi, r12
							VMOVUPS ymm1, [rsi]
							ADD rsi, r12
							VMOVUPS ymm2, [rsi]
							ADD rsi, r12
							VMOVUPS ymm3, [rsi]

							MOV rsi, pImg
							MOV rdi, pOut

							MOV ecx, W
							wloops :
							CMP ecx, 0
								JE wloope

								VMOVUPS ymm4, [rsi]
								ADD rsi, r12

								VBROADCASTSS ymm7, [rdi + 0]
								VBROADCASTSS ymm8, [rdi + 4]
								VBROADCASTSS ymm9, [rdi + 8]
								VBROADCASTSS ymm10, [rdi + 12]
								VFMADD231PS ymm0, ymm7, ymm4
								VFMADD231PS ymm1, ymm8, ymm4
								VFMADD231PS ymm2, ymm9, ymm4
								VFMADD231PS ymm3, ymm10, ymm4
								ADD rdi, r13

								DEC ecx
								JMP wloops
								wloope :

							MOV rsi, pKern
								VMOVUPS[rsi], ymm0
								ADD rsi, r12
								VMOVUPS[rsi], ymm1
								ADD rsi, r12
								VMOVUPS[rsi], ymm2
								ADD rsi, r12
								VMOVUPS[rsi], ymm3
						}

					}
					for (int c = C8; c < C; c++) {

						float* pImg = img + b * HWC + h * WC + c;
						float* pKern = kern + x * KyNC + y * NC + n * C + c;
						float* pOut = out + b * NHNWN + nh * NWN + y * N + n;

						float s1, s2, s3, s4;
						s1 = s2 = s3 = s4 = 0;
						for (int w = 0; w < W; w++) {
							float i1 = pImg[w * C];

							s1 += i1 * pOut[w * NS + 0];
							s2 += i1 * pOut[w * NS + 1];
							s3 += i1 * pOut[w * NS + 2];
							s4 += i1 * pOut[w * NS + 3];
						}
						pKern[0 * C] += s1;
						pKern[1 * C] += s2;
						pKern[2 * C] += s3;
						pKern[3 * C] += s4;
					}
				}
			}
		}
		else {
			for (int b = 0; b < batchsize; b++) {
				for (int h = 0; h < H; h++) {
					int nh = h * stride + x;
					for (int n = nl; n < N; n++) {
						for (int c = 0; c < C8; c += 8) {

							float* pImg = img + b * HWC + h * WC + c;
							float* pKern = kern + x * KyNC + y * NC + n * C + c;
							float* pOut = out + b * NHNWN + nh * NWN + y * N + n;

							__asm {

								MOV r12, C4
								MOV r13, NS4

								MOV rsi, pKern
								VMOVUPS ymm0, [rsi]

								MOV rsi, pImg
								MOV rdi, pOut

								MOV ecx, W
								wloops :
								CMP ecx, 0
									JE wloope

									VMOVUPS ymm4, [rsi]
									ADD rsi, r12

									VBROADCASTSS ymm7, [rdi + 0]
									VFMADD231PS ymm0, ymm7, ymm4
									ADD rdi, r13

									DEC ecx
									JMP wloops
									wloope :

								MOV rsi, pKern
									VMOVUPS[rsi], ymm0
							}

						}
						for (int c = C8; c < C; c++) {

							float* pImg = img + b * HWC + h * WC + c;
							float* pKern = kern + x * KyNC + y * NC + n * C + c;
							float* pOut = out + b * NHNWN + nh * NWN + y * N + n;

							float s1 = 0;
							for (int w = 0; w < W; w++) {
								float i1 = pImg[w * C];

								s1 += i1 * pOut[w * NS + 0];
							}
							pKern[0 * C] += s1;
						}
					}
				}
			}
		}
	}
}
