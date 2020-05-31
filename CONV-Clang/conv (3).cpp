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
	int Ch = (C / 8) * 8;

	int WC = W * C;
	int HWC = H * WC;
	int NWN = NW * N;
	int NHNWN = NH * NWN;
	int NC = N * C;
	int KyNC = Ky * NC;

	#pragma omp parallel for
	for (int nh = 0; nh < NH; nh++) {
		for (int b = 0; b < batchsize; b++) {
			for (int x = 0; x < Kx; x++) {
				int h = nh * stride + x;
				for (int y = 0; y < Ky; y++) {
					for (int nw = 0; nw < NWh; nw += 3) {
						int w = nw * stride + y;
						for (int n = 0; n < Nh; n += 4) {
							float s11, s12, s13, s14, s21, s22, s23, s24, s31, s32, s33, s34;
							s11 = s12 = s13 = s14 = s21 = s22 = s23 = s24 = s31 = s32 = s33 = s34 = 0;
							for (int c = 0; c < C; c++) {
								float w1 = img[b * HWC + h * WC + w * C + c];
								float w2 = img[b * HWC + h * WC + (w + stride) * C + c];
								float w3 = img[b * HWC + h * WC + (w + 2 * stride) * C + c];

								float k1 = kern[x * KyNC + y * NC + n * C + c];
								float k2 = kern[x * KyNC + y * NC + (n + 1) * C + c];
								float k3 = kern[x * KyNC + y * NC + (n + 2) * C + c];
								float k4 = kern[x * KyNC + y * NC + (n + 3) * C + c];

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
							out[b * NHNWN + nh * NWN + nw * N + n] += s11;
							out[b * NHNWN + nh * NWN + nw * N + n + 1] += s12;
							out[b * NHNWN + nh * NWN + nw * N + n + 2] += s13;
							out[b * NHNWN + nh * NWN + nw * N + n + 3] += s14;
							out[b * NHNWN + nh * NWN + (nw + 1) * N + n] += s21;
							out[b * NHNWN + nh * NWN + (nw + 1) * N + n + 1] += s22;
							out[b * NHNWN + nh * NWN + (nw + 1) * N + n + 2] += s23;
							out[b * NHNWN + nh * NWN + (nw + 1) * N + n + 3] += s24;
							out[b * NHNWN + nh * NWN + (nw + 2) * N + n] += s31;
							out[b * NHNWN + nh * NWN + (nw + 2) * N + n + 1] += s32;
							out[b * NHNWN + nh * NWN + (nw + 2) * N + n + 2] += s33;
							out[b * NHNWN + nh * NWN + (nw + 2) * N + n + 3] += s34;
						}
						for (int n = Nh; n < N; n++) {
							float s1, s2, s3;
							s1 = s2 = s3 = 0;
							for (int c = 0; c < C; c++) {
								float k1 = kern[x * KyNC + y * NC + n * C + c];
								s1 += img[b * HWC + h * WC + w * C + c] * k1;
								s2 += img[b * HWC + h * WC + (w + stride) * C + c] * k1;
								s3 += img[b * HWC + h * WC + (w + 2 * stride) * C + c] * k1;
							}
							out[b * NHNWN + nh * NWN + nw * N + n] += s1;
							out[b * NHNWN + nh * NWN + (nw + 1) * N + n] += s2;
							out[b * NHNWN + nh * NWN + (nw + 2) * N + n] += s3;
						}
					}
					for (int nw = NWh; nw < NW; nw++) {
						int w = nw * stride + y;
						for (int n = 0; n < Nh; n += 4) {
							float s1, s2, s3, s4;
							s1 = s2 = s3 = s4 = 0;
							for (int c = 0; c < C; c++) {
								float w1 = img[b * HWC + h * WC + w * C + c];

								float k1 = kern[x * KyNC + y * NC + n * C + c];
								float k2 = kern[x * KyNC + y * NC + (n + 1) * C + c];
								float k3 = kern[x * KyNC + y * NC + (n + 2) * C + c];
								float k4 = kern[x * KyNC + y * NC + (n + 3) * C + c];

								s1 += w1 * k1;
								s2 += w1 * k2;
								s3 += w1 * k3;
								s4 += w1 * k4;
							}
							out[b * NHNWN + nh * NWN + nw * N + n] += s1;
							out[b * NHNWN + nh * NWN + nw * N + n + 1] += s2;
							out[b * NHNWN + nh * NWN + nw * N + n + 2] += s3;
							out[b * NHNWN + nh * NWN + nw * N + n + 3] += s4;
						}
						for (int n = Nh; n < N; n++) {
							float s1;
							s1 = 0;
							for (int c = 0; c < C; c++) {
								float w1 = img[b * HWC + h * WC + w * C + c];

								float k1 = kern[x * KyNC + y * NC + n * C + c];

								s1 += w1 * k1;
							}
							out[b * NHNWN + nh * NWN + nw * N + n] += s1;
						}
					}
				}
			}
		}
	}

}


DLLEXPORT void avx2_conv3d_back_i(float* img, float* kern, float* out, int H, int W, int C, int N, int Kx, int Ky, int stride, int padding, int batchsize) {

	int NH = (H - Kx) / stride + 1;
	int NW = (W - Ky) / stride + 1;

	memset(img, 0, sizeof(float) * batchsize * C * H * W);

	int NWh = (NW / 6) * 6;
	int Nh = (N / 4) * 4;
	int Ch = (C / 16) * 16;

	int NWN = NW * N;
	int NHNWN = NH * NWN;
	int NC = N * C;
	int WC = W * C;
	int KyNC = Ky * NC;
	int HWC = H * WC;


	#pragma omp parallel for
	for (int h = 0; h < H; h++) {
		int xl = (h - (NH - 1) < 0) ? Kx : (h - (NH - 1));
		int xh = (h < Kx) ? h : Kx;
		for (int b = 0; b < batchsize; b++) {
			for (int x = xl; x < xh; x += stride) {
				int nh = (h - x) / stride;
				for (int y = 0; y < Ky; y++) {
					for (int nw = 0; nw < NWh; nw += 3) {
						int w = nw * stride + y;

						float* pi1 = img + (b * HWC + h * WC + w * C);
						float* pi2 = img + (b * HWC + h * WC + (w + stride) * C);
						float* pi3 = img + (b * HWC + h * WC + (w + 2 * stride) * C);

						for (int n = 0; n < Nh; n += 4) {
							float o11 = out[b * NHNWN + nh * NWN + nw * N + n];
							float o12 = out[b * NHNWN + nh * NWN + nw * N + n + 1];
							float o13 = out[b * NHNWN + nh * NWN + nw * N + n + 2];
							float o14 = out[b * NHNWN + nh * NWN + nw * N + n + 3];
							float o21 = out[b * NHNWN + nh * NWN + (nw + 1) * N + n];
							float o22 = out[b * NHNWN + nh * NWN + (nw + 1) * N + n + 1];
							float o23 = out[b * NHNWN + nh * NWN + (nw + 1) * N + n + 2];
							float o24 = out[b * NHNWN + nh * NWN + (nw + 1) * N + n + 3];
							float o31 = out[b * NHNWN + nh * NWN + (nw + 2) * N + n];
							float o32 = out[b * NHNWN + nh * NWN + (nw + 2) * N + n + 1];
							float o33 = out[b * NHNWN + nh * NWN + (nw + 2) * N + n + 2];
							float o34 = out[b * NHNWN + nh * NWN + (nw + 2) * N + n + 3];

							for (int c = 0; c < C; c++) {
								float k1 = kern[x * KyNC + y * NC + n * C + c];
								float k2 = kern[x * KyNC + y * NC + (n + 1) * C + c];
								float k3 = kern[x * KyNC + y * NC + (n + 2) * C + c];
								float k4 = kern[x * KyNC + y * NC + (n + 3) * C + c];

								pi1[c] += o11 * k1 + o12 * k2 + o13 * k3 + o14 * k4;
								pi2[c] += o21 * k1 + o22 * k2 + o23 * k3 + o24 * k4;
								pi3[c] += o31 * k1 + o32 * k2 + o33 * k3 + o34 * k4;
							}
						}
						for (int n = Nh; n < N; n++) {
							float o11 = out[b * NHNWN + nh * NWN + nw * N + n];
							float o21 = out[b * NHNWN + nh * NWN + (nw + 1) * N + n];
							float o31 = out[b * NHNWN + nh * NWN + (nw + 2) * N + n];

							for (int c = 0; c < C; c++) {
								float k1 = kern[x * KyNC + y * NC + n * C + c];

								pi1[c] += o11 * k1;
								pi2[c] += o21 * k1;
								pi3[c] += o31 * k1;
							}
						}
					}
					for (int nw = NWh; nw < NW; nw++) {
						int w = nw * stride + y;

						float* pi1 = img + (b * HWC + h * WC + w * C);

						for (int n = 0; n < Nh; n += 4) {
							float o11 = out[b * NHNWN + nh * NWN + nw * N + n];
							float o12 = out[b * NHNWN + nh * NWN + nw * N + n + 1];
							float o13 = out[b * NHNWN + nh * NWN + nw * N + n + 2];
							float o14 = out[b * NHNWN + nh * NWN + nw * N + n + 3];

							for (int c = 0; c < C; c++) {
								float k1 = kern[x * KyNC + y * NC + n * C + c];
								float k2 = kern[x * KyNC + y * NC + (n + 1) * C + c];
								float k3 = kern[x * KyNC + y * NC + (n + 2) * C + c];
								float k4 = kern[x * KyNC + y * NC + (n + 3) * C + c];

								pi1[c] += o11 * k1 + o12 * k2 + o13 * k3 + o14 * k4;
							}
						}
						for (int n = Nh; n < N; n++) {
							float o11 = out[b * NHNWN + nh * NWN + nw * N + n];

							for (int c = 0; c < C; c++) {
								float k1 = kern[x * KyNC + y * NC + n * C + c];

								pi1[c] += o11 * k1;
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

	int Nh = (N / 6) * 6;
	int Ch = (C / 16) * 16;
	int NWh = (NW / 8) * 8;

	int WC = W * C;
	int HWC = H * WC;
	int NWN = NW * N;
	int NHNWN = NH * NWN;

	int NC = N * C;
	int KyNC = Ky * NC;

	#pragma omp parallel for
	for (int n = 0; n < N; n++) {
		for (int b = 0; b < batchsize; b++) {
			for (int nh = 0; nh < NH; nh++) {
				for (int x = 0; x < Kx; x++) {
					int h = nh * stride + x;
					for (int nw = 0; nw < NWh; nw += 8) {
						for (int y = 0; y < Ky; y++) {
							int w = nw * stride + y;

							float o1 = out[b * NHNWN + nh * NWN + nw * N + n];
							float o2 = out[b * NHNWN + nh * NWN + (nw + 1) * N + n];
							float o3 = out[b * NHNWN + nh * NWN + (nw + 2) * N + n];
							float o4 = out[b * NHNWN + nh * NWN + (nw + 3) * N + n];
							float o5 = out[b * NHNWN + nh * NWN + (nw + 4) * N + n];
							float o6 = out[b * NHNWN + nh * NWN + (nw + 5) * N + n];
							float o7 = out[b * NHNWN + nh * NWN + (nw + 6) * N + n];
							float o8 = out[b * NHNWN + nh * NWN + (nw + 7) * N + n];

							float* pk1 = kern + (x * KyNC + y * NC + n * C);

							float* pi1 = &img[b * HWC + h * WC + (w + 0) * C];
							float* pi2 = &img[b * HWC + h * WC + (w + 1 * stride) * C];
							float* pi3 = &img[b * HWC + h * WC + (w + 2 * stride) * C];
							float* pi4 = &img[b * HWC + h * WC + (w + 3 * stride) * C];
							float* pi5 = &img[b * HWC + h * WC + (w + 4 * stride) * C];
							float* pi6 = &img[b * HWC + h * WC + (w + 5 * stride) * C];
							float* pi7 = &img[b * HWC + h * WC + (w + 6 * stride) * C];
							float* pi8 = &img[b * HWC + h * WC + (w + 7 * stride) * C];

							for (int c = 0; c < C; c++) {
								pk1[c] += pi1[c] * o1 + pi2[c] * o2 + pi3[c] * o3 + pi4[c] * o4 + pi5[c] * o5 + pi6[c] * o6 + pi7[c] * o7 + pi8[c] * o8;
							}
						}
					}
					for (int nw = NWh; nw < NW; nw++) {
						for (int y = 0; y < Ky; y++) {
							int w = nw * stride + y;

							float o1 = out[b * NHNWN + nh * NWN + nw * N + n];
							float* pk1 = kern + (x * KyNC + y * NC + n * C);
							float* pi1 = &img[b * HWC + h * WC + (w + 0) * C];

							for (int c = 0; c < C; c++) {
								pk1[c] += pi1[c] * o1;
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
	int Ch = (C / 8) * 8;
	int Wh = (W / 3) * 3;
	int C8 = Ch;

	int WC = W * C;
	int HWC = H * WC;
	int NC = N * C;
	int KyNC = Ky * NC;
	int NWN = NW * N;
	int NHNWN = NH * NWN;

	long long C4 = C * 4;
	int NS = N * stride;
	long long NS4 = NS * 4;

	if (Nh != 0 && Ch != 0 && Wh != 0) {

		int NB = (Nh / 16 < THREADS) ? 4 : 16;
		int CB = 512;

		int Nb = ceil((float)Nh / NB);
		int Cb = ceil((float)Ch / CB);

		#pragma omp parallel for
		for (int nb = 0; nb < Nb; nb++) {
			float* mt = (float*)_aligned_malloc(sizeof(float) * 8 * 12, 32);
			int Nl = nb * NB;
			int Nm = (Nl + NB > Nh) ? Nh - Nl : NB;

			float* kern1 = kern + nb * NB * C;
			float* out1 = out + nb * NB;
			for (int cb = 0; cb < Cb; cb++) {
				int Cl = cb * CB;
				int Cm = (Cl + CB > Ch) ? Ch - Cl : CB;

				float* img1 = img + cb * CB;
				float* kern2 = kern1 + cb * CB;
				for (int b = 0; b < batchsize; b++) {
					float* img2 = img1 + b * HWC;
					float* out2 = out1 + b * NHNWN;
					for (int h = 0; h < H; h++) {
						float* img3 = img2 + h * WC;
						for (int x = 0; x < Kx; x++) {
							int nh = (h * stride + x);
							float* kern3 = kern2 + x * KyNC;
							float* out3 = out2 + nh * NWN;
							for (int y = 0; y < Ky; y++) {
								float* kern4 = kern3 + y * NC;
								for (int w = 0; w < Wh; w += 3) {
									int nw = (w * stride + y);
									float* pImg = img3 + w * C;
									float* out4 = out3 + nw * N;
									for (int n = 0; n < Nm; n += 4) {

										float* pKern = kern4 + n * C;
										float* pOut = out4 + n;

										__asm {

											MOV r12, C4

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

											MOV ecx, Cm
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
										{
											float* t = mt;
											pOut[0 * NS + 0] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[1 * NS + 0] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[2 * NS + 0] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[0 * NS + 1] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[1 * NS + 1] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[2 * NS + 1] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[0 * NS + 2] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[1 * NS + 2] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[2 * NS + 2] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[0 * NS + 3] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[1 * NS + 3] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[2 * NS + 3] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
										}

									}
								}
							}
						}
					}
				}

			}
		}
	}

	if (W != Wh) {

		int NB = (Nh / 16 < THREADS) ? 4 : 16;
		int CB = 512;

		int Nb = ceil((float)Nh / NB);
		int Cb = ceil((float)Ch / CB);

		#pragma omp parallel for
		for (int nb = 0; nb < Nb; nb++) {
			float* mt = (float*)_aligned_malloc(sizeof(float) * 8 * 4, 32);
			int Nl = nb * NB;
			int Nm = (Nl + NB > Nh) ? Nh - Nl : NB;

			float* kern1 = kern + nb * NB * C;
			float* out1 = out + nb * NB;
			for (int cb = 0; cb < Cb; cb++) {
				int Cl = cb * CB;
				int Cm = (Cl + CB > Ch) ? Ch - Cl : CB;

				float* img1 = img + cb * CB;
				float* kern2 = kern1 + cb * CB;
				for (int b = 0; b < batchsize; b++) {
					float* img2 = img1 + b * HWC;
					float* out2 = out1 + b * NHNWN;
					for (int h = 0; h < H; h++) {
						float* img3 = img2 + h * WC;
						for (int x = 0; x < Kx; x++) {
							int nh = (h * stride + x);
							float* kern3 = kern2 + x * KyNC;
							float* out3 = out2 + nh * NWN;
							for (int y = 0; y < Ky; y++) {
								float* kern4 = kern3 + y * NC;
								for (int w = Wh; w < W; w++) {
									int nw = (w * stride + y);
									float* pImg = img3 + w * C;
									float* out4 = out3 + nw * N;
									for (int n = 0; n < Nm; n += 4) {

										float* pKern = kern4 + n * C;
										float* pOut = out4 + n;

										__asm {

											MOV r12, C4

											VXORPS ymm0, ymm0, ymm0
											VXORPS ymm1, ymm1, ymm1
											VXORPS ymm2, ymm2, ymm2
											VXORPS ymm3, ymm3, ymm3

											MOV rsi, pImg
											MOV rdi, pKern

											MOV ecx, Cm
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
										{
											float* t = mt;
											pOut[0] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[1] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[2] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[3] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
										}
										{
											float s1, s2, s3, s4;
											s1 = s2 = s3 = s4 = 0;
											for (int c = Ch; c < C; c++) {
												float w1 = pImg[0 * C + c];

												float k1 = pKern[0 * C + c];
												float k2 = pKern[1 * C + c];
												float k3 = pKern[2 * C + c];
												float k4 = pKern[3 * C + c];

												s1 += w1 * k1;
												s2 += w1 * k2;
												s3 += w1 * k3;
												s4 += w1 * k4;
											}
											pOut[0] += s1;
											pOut[1] += s2;
											pOut[2] += s3;
											pOut[3] += s4;
										}
									}
								}
							}
						}
					}
				}

			}
		}

		if (N != Nh){
			#pragma omp parallel for
			for (int h = 0; h < H; h++) {
				float* mt = (float*)_aligned_malloc(sizeof(float) * 8, 32);
				float* img1 = img + h * WC;
				for (int b = 0; b < batchsize; b++) {
					float* img2 = img1 + b * HWC;
					float* out1 = out + b * NHNWN;
					for (int x = 0; x < Kx; x++) {
						int nh = (h * stride + x);
						float* kern1 = kern + x * KyNC;
						float* out2 = out1 + nh * NWN;
						for (int y = 0; y < Ky; y++) {
							float* kern2 = kern1 + y * NC;
							for (int w = Wh; w < W; w++) {
								int nw = (w * stride + y);
								float* pImg = img2 + w * C;
								float* out4 = out2 + nw * N;
								for (int n = Nh; n < N; n++) {

									float* pKern = kern2 + n * C;
									float* pOut = out4 + n;

									__asm {

										VXORPS ymm0, ymm0, ymm0

										MOV rsi, pImg
										MOV rdi, pKern

										MOV ecx, C8
										cloops :
										CMP ecx, 0
											JE cloope

											VMOVUPS ymm12, [rsi]
											ADD rsi, 32

											VMOVUPS ymm15, [rdi]
											ADD rdi, 32

											VFMADD231PS ymm0, ymm15, ymm12

											SUB ecx, 8
											JMP cloops
											cloope :

										MOV rsi, mt
											VMOVUPS[rsi], ymm0
									}
									{
										float* t = mt;
										pOut[0] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
									}
									{
										float s1 = 0;
										for (int c = Ch; c < C; c++) {
											float w1 = pImg[0 * C + c];
											float k1 = pKern[0 * C + c];
											s1 += w1 * k1;
										}
										pOut[0] += s1;
									}
								}
							}
						}
					}


				}
			}
		}
	}

	if (N != Nh) {

		#pragma omp parallel for
		for (int nh = 0; nh < NH; nh++) {
			float* mt = (float*)_aligned_malloc(sizeof(float) * 8 * 3, 32);
			int xl = nh % stride;
			xl = ((nh - xl) / stride >= H) ? nh - (H - 1) * stride : xl;
			int xh = (nh + stride > Kx) ? Kx : nh + stride;
			float* out1 = out + nh * NWN;
			for (int b = 0; b < batchsize; b++) {
				float* out2 = out1 + b * NHNWN;
				float* img1 = img + b * HWC;
				for (int x = xl; x < xh; x += stride) {
					int h = (nh - x) / stride;
					float* img2 = img1 + h * WC;
					float* kern1 = kern + x * KyNC;
					for (int y = 0; y < Ky; y++) {
						float* kern2 = kern1 + y * NC;
						for (int w = 0; w < Wh; w += 3) {
							int nw = w * stride + y;
							float* pImg = img2 + w * C;
							float* out3 = out2 + nw * N;
							for (int n = Nh; n < N; n++) {
								float* pKern = kern2 + n * C;
								float* pOut = out3 + n;

								__asm {

									MOV r12, C4
									MOV r13, NS4

									VXORPS ymm0, ymm0, ymm0
									VXORPS ymm1, ymm1, ymm1
									VXORPS ymm2, ymm2, ymm2

									MOV rsi, pImg
									MOV rdi, pKern

									MOV ecx, C8
									cloops:
									CMP ecx, 0
									JE cloope

									VMOVUPS ymm3, [rdi]
									ADD rdi, 32

									MOV rdx, rsi
									VMOVUPS ymm4, [rsi]
									ADD rsi, r12
									VFMADD231PS ymm0, ymm3, ymm4
									VMOVUPS ymm5, [rsi]
									ADD rsi, r12
									VFMADD231PS ymm1, ymm3, ymm5
									VMOVUPS ymm6, [rsi]
									MOV rsi, rdx
									ADD rsi, 32
									VFMADD231PS ymm2, ymm3, ymm6

									SUB ecx, 8
									JMP cloops
									cloope:

									MOV rsi, mt
									VMOVUPS[rsi], ymm0
									ADD rsi, 32
									VMOVUPS[rsi], ymm1
									ADD rsi, 32
									VMOVUPS[rsi], ymm2

								}
								{
									float* t = mt;
									pOut[0 * NS] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
									t += 8;
									pOut[1 * NS] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
									t += 8;
									pOut[2 * NS] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								}
								{
									float s1, s2, s3;
									s1 = s2 = s3 = 0;
									for (int c = Ch; c < C; c++) {
										float k1 = pKern[c];
										s1 += pImg[0 * C + c] * k1;
										s2 += pImg[1 * C + c] * k1;
										s3 += pImg[2 * C + c] * k1;
									}
									pOut[0 * NS] += s1;
									pOut[1 * NS] += s2;
									pOut[2 * NS] += s3;
								}
							}
						}
					}
				}
			}
		}

	}

	if (C != Ch) {

		#pragma omp parallel for
		for (int nh = 0; nh < NH; nh++) {
			float* mt = (float*)_aligned_malloc(sizeof(float) * 8 * 3, 32);
			int xl = nh % stride;
			xl = ((nh - xl) / stride >= H) ? nh - (H - 1) * stride : xl;
			int xh = (nh + stride > Kx) ? Kx : nh + stride;
			float* out1 = out + nh * NWN;
			for (int b = 0; b < batchsize; b++) {
				float* out2 = out1 + b * NHNWN;
				float* img1 = img + b * HWC;
				for (int x = xl; x < xh; x += stride) {
					int h = (nh - x) / stride;
					float* img2 = img1 + h * WC;
					float* kern1 = kern + x * KyNC;
					for (int y = 0; y < Ky; y++) {
						float* kern2 = kern1 + y * NC;
						for (int w = 0; w < Wh; w += 3) {
							int nw = w * stride + y;
							float* pImg = img2 + w * C;
							float* out3 = out2 + nw * N;
							for (int n = 0; n < Nh; n += 4) {
								float* pKern = kern2 + n * C;
								float* pOut = out3 + n;

								float s11, s12, s13, s14, s21, s22, s23, s24, s31, s32, s33, s34;
								s11 = s12 = s13 = s14 = s21 = s22 = s23 = s24 = s31 = s32 = s33 = s34 = 0;
								for (int c = Ch; c < C; c++) {
									float w1 = pImg[0 * C + c];
									float w2 = pImg[1 * C + c];
									float w3 = pImg[2 * C + c];

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
								pOut[0 * NS + 0] += s11;
								pOut[0 * NS + 1] += s12;
								pOut[0 * NS + 2] += s13;
								pOut[0 * NS + 3] += s14;
								pOut[1 * NS + 0] += s21;
								pOut[1 * NS + 1] += s22;
								pOut[1 * NS + 2] += s23;
								pOut[1 * NS + 3] += s24;
								pOut[2 * NS + 0] += s31;
								pOut[2 * NS + 1] += s32;
								pOut[2 * NS + 2] += s33;
								pOut[2 * NS + 3] += s34;
							}
						}
					}
				}
			}
		}

	}
}


DLLEXPORT void avx2_conv3dtranspose_back_i(float* img, float* kern, float* out, int H, int W, int C, int N, int Kx, int Ky, int stride, int padding, int batchsize) {

	int NH = (H - 1) * stride + Kx;
	int NW = (W - 1) * stride + Ky;

	memset(img, 0, sizeof(float) * batchsize * H * W * C);

	int Nh = (N / 4) * 4;
	int Ch = (C / 8) * 8;
	int Wh = (W / 3) * 3;
	int C8 = Ch;

	int WC = W * C;
	int HWC = H * WC;
	int NC = N * C;
	int KyNC = Ky * NC;
	int NWN = NW * N;
	int NHNWN = NH * NWN;

	long long C4 = C * 4;
	int NS = N * stride;
	long long NS4 = NS * 4;

	if (Nh != 0 && Ch != 0 && Wh != 0) {

		int NB = 16;
		int CB = 512;

		int Nb = ceil((float)Nh / NB);
		int Cb = ceil((float)Ch / CB);

		#pragma omp parallel for
		for (int h = 0; h < H; h++) {
			float* mt = (float*)_aligned_malloc(sizeof(float) * 8 * 12, 32);
			float* img1 = img + h * WC;
			for (int nb = 0; nb < Nb; nb++) {
				int Nl = nb * NB;
				int Nm = (Nl + NB > Nh) ? Nh - Nl : NB;

				float* kern1 = kern + nb * NB * C;
				float* out1 = out + nb * NB;
				for (int cb = 0; cb < Cb; cb++) {
					int Cl = cb * CB;
					int Cm = (Cl + CB > Ch) ? Ch - Cl : CB;

					float* img2 = img1 + cb * CB;
					float* kern2 = kern1 + cb * CB;
					for (int b = 0; b < batchsize; b++) {
						float* img3 = img2 + b * HWC;
						float* out2 = out1 + b * NHNWN;
						for (int x = 0; x < Kx; x++) {
							int nh = (h * stride + x);
							float* kern3 = kern2 + x * KyNC;
							float* out3 = out2 + nh * NWN;
							for (int y = 0; y < Ky; y++) {
								float* kern4 = kern3 + y * NC;
								for (int w = 0; w < Wh; w += 3) {
									int nw = (w * stride + y);
									float* pImg = img3 + w * C;
									float* out4 = out3 + nw * N;
									for (int n = 0; n < Nm; n += 2) {

										float* pKern = kern4 + n * C;
										float* pOut = out4 + n;

										__asm {

											MOV r12, NS4
											MOV r13, C4

											MOV rsi, pOut
											VBROADCASTSS ymm0, [rsi]
											VBROADCASTSS ymm1, [rsi + 4]
											ADD rsi, r12
											VBROADCASTSS ymm2, [rsi]
											VBROADCASTSS ymm3, [rsi + 4]
											ADD rsi, r12
											VBROADCASTSS ymm4, [rsi]
											VBROADCASTSS ymm5, [rsi + 4]

											MOV rsi, pKern
											MOV rdi, pImg

											MOV ecx, Cm
											cloops :
											CMP ecx, 0
												JE cloope

												MOV rdx, rsi
												VMOVUPS ymm7, [rsi]
												ADD rsi, r13
												VMOVUPS ymm8, [rsi]
												MOV rsi, rdx
												ADD rsi, 32

												MOV rdx, rdi
												VMOVUPS ymm9, [rdi]
												VFMADD231PS ymm9, ymm7, ymm0
												VFMADD231PS ymm9, ymm8, ymm1
												VMOVUPS[rdi], ymm9
												ADD rdi, r13

												VMOVUPS ymm9, [rdi]
												VFMADD231PS ymm9, ymm7, ymm2
												VFMADD231PS ymm9, ymm8, ymm3
												VMOVUPS[rdi], ymm9
												ADD rdi, r13

												VMOVUPS ymm9, [rdi]
												VFMADD231PS ymm9, ymm7, ymm4
												VFMADD231PS ymm9, ymm8, ymm5
												VMOVUPS[rdi], ymm9
												MOV rdi, rdx
												ADD rdi, 32

												SUB ecx, 8
												JMP cloops
												cloope :
										}

									}
								}
							}
						}
					}
				}
			}
		}
	}

	if (W != Wh) {

		int NB = (Nh / 16 < THREADS) ? 4 : 16;
		int CB = 512;

		int Nb = ceil((float)Nh / NB);
		int Cb = ceil((float)Ch / CB);

		#pragma omp parallel for
		for (int h = 0; h < H; h++) {
			float* mt = (float*)_aligned_malloc(sizeof(float) * 8 * 12, 32);
			float* img1 = img + h * WC;
			for (int nb = 0; nb < Nb; nb++) {
				int Nl = nb * NB;
				int Nm = (Nl + NB > Nh) ? Nh - Nl : NB;

				float* kern1 = kern + nb * NB * C;
				float* out1 = out + nb * NB;
				for (int cb = 0; cb < Cb; cb++) {
					int Cl = cb * CB;
					int Cm = (Cl + CB > Ch) ? Ch - Cl : CB;

					float* img2 = img1 + cb * CB;
					float* kern2 = kern1 + cb * CB;
					for (int b = 0; b < batchsize; b++) {
						float* img3 = img2 + b * HWC;
						float* out2 = out1 + b * NHNWN;
						for (int x = 0; x < Kx; x++) {
							int nh = (h * stride + x);
							float* kern3 = kern2 + x * KyNC;
							float* out3 = out2 + nh * NWN;
							for (int y = 0; y < Ky; y++) {
								float* kern4 = kern3 + y * NC;
								for (int w = Wh; w < W; w++) {
									int nw = (w * stride + y);
									float* pImg = img3 + w * C;
									float* out4 = out3 + nw * N;
									for (int n = 0; n < Nm; n += 4) {

										float* pKern = kern4 + n * C;
										float* pOut = out4 + n;

										__asm {

											MOV r13, C4

											MOV rsi, pOut
											VBROADCASTSS ymm0, [rsi]
											VBROADCASTSS ymm1, [rsi + 4]
											VBROADCASTSS ymm2, [rsi + 8]
											VBROADCASTSS ymm3, [rsi + 12]

											MOV rsi, pKern
											MOV rdi, pImg

											MOV ecx, Cm
											cloops :
											CMP ecx, 0
												JE cloope

												MOV rdx, rsi
												VMOVUPS ymm4, [rsi]
												ADD rsi, r13
												VMOVUPS ymm5, [rsi]
												ADD rsi, r13
												VMOVUPS ymm6, [rsi]
												ADD rsi, r13
												VMOVUPS ymm7, [rsi]
												MOV rsi, rdx
												ADD rsi, 32

												VMOVUPS ymm9, [rdi]
												VFMADD231PS ymm9, ymm0, ymm4
												VFMADD231PS ymm9, ymm1, ymm5
												VFMADD231PS ymm9, ymm2, ymm6
												VFMADD231PS ymm9, ymm3, ymm7
												VMOVUPS[rdi], ymm9
												ADD rdi, 32

												SUB ecx, 8
												JMP cloops
												cloope :
										}
										{
											float n1 = pOut[0];
											float n2 = pOut[1];
											float n3 = pOut[2];
											float n4 = pOut[3];
											for (int c = Ch; c < C; c++) {
												float k1 = pKern[0 * C + c];
												float k2 = pKern[1 * C + c];
												float k3 = pKern[2 * C + c];
												float k4 = pKern[3 * C + c];

												pImg[c] += n1 * k1 + n2 * k2 + n3 * k3 + n4 * k4;
											}
										}

									}
									for (int n = Nh; n < N; n++) {
										float* pKern = kern4 + n * C;
										float* pOut = out4 + n;

										__asm {

											MOV r13, C4

											MOV rsi, pOut
											VBROADCASTSS ymm0, [rsi]

											MOV rsi, pKern
											MOV rdi, pImg

											MOV ecx, Cm
											cloops :
											CMP ecx, 0
												JE cloope

												VMOVUPS ymm4, [rsi]
												ADD rsi, 32

												VMOVUPS ymm9, [rdi]
												VFMADD231PS ymm9, ymm0, ymm4
												VMOVUPS[rdi], ymm9
												ADD rdi, 32

												SUB ecx, 8
												JMP cloops
												cloope :
										}
										{
											float n1 = pOut[0];
											for (int c = Ch; c < C; c++) {
												pImg[c] += n1 * pKern[c];
											}
										}
									}
								}
							}
						}
					}
				}

			}
			if (N != Nh) {
				for (int nb = 0; nb < Nb; nb++) {
					int Nl = nb * NB;
					int Nm = (Nl + NB > Nh) ? Nh - Nl : NB;

					float* kern1 = kern + nb * NB * C;
					float* out1 = out + nb * NB;
					for (int cb = 0; cb < Cb; cb++) {
						int Cl = cb * CB;
						int Cm = (Cl + CB > Ch) ? Ch - Cl : CB;

						float* img2 = img1 + cb * CB;
						float* kern2 = kern1 + cb * CB;
						for (int b = 0; b < batchsize; b++) {
							float* img3 = img2 + b * HWC;
							float* out2 = out1 + b * NHNWN;
							for (int x = 0; x < Kx; x++) {
								int nh = (h * stride + x);
								float* kern3 = kern2 + x * KyNC;
								float* out3 = out2 + nh * NWN;
								for (int y = 0; y < Ky; y++) {
									float* kern4 = kern3 + y * NC;
									for (int w = Wh; w < W; w++) {
										int nw = (w * stride + y);
										float* pImg = img3 + w * C;
										float* out4 = out3 + nw * N;
										for (int n = Nh; n < N; n++) {
											float* pKern = kern4 + n * C;
											float* pOut = out4 + n;

											__asm {

												MOV r13, C4

												MOV rsi, pOut
												VBROADCASTSS ymm0, [rsi]

												MOV rsi, pKern
												MOV rdi, pImg

												MOV ecx, Cm
												cloops :
												CMP ecx, 0
													JE cloope

													VMOVUPS ymm4, [rsi]
													ADD rsi, 32

													VMOVUPS ymm9, [rdi]
													VFMADD231PS ymm9, ymm0, ymm4
													VMOVUPS[rdi], ymm9
													ADD rdi, 32

													SUB ecx, 8
													JMP cloops
													cloope :
											}
											{
												float n1 = pOut[0];
												for (int c = Ch; c < C; c++) {
													pImg[c] += n1 * pKern[c];
												}
											}
										}
									}
								}
							}
						}
					}

				}

			}
		}
	}

	if (N != Nh) {

		#pragma omp parallel for
		for (int h = 0; h < H; h++) {
			float* mt = (float*)_aligned_malloc(sizeof(float) * 8 * 12, 32);
			float* img1 = img + h * WC;
			for (int b = 0; b < batchsize; b++) {
				float* img2 = img1 + b * HWC;
				float* out1 = out + b * NHNWN;
				for (int x = 0; x < Kx; x++) {
					int nh = (h * stride + x);
					float* kern1 = kern + x * KyNC;
					float* out2 = out1 + nh * NWN;
					for (int y = 0; y < Ky; y++) {
						float* kern2 = kern1 + y * NC;
						for (int w = 0; w < Wh; w += 3) {
							int nw = (w * stride + y);
							float* pImg = img2 + w * C;
							float* out3 = out2 + nw * N;
							for (int n = Nh; n < N; n++) {

								float* pKern = kern2 + n * C;
								float* pOut = out3 + n;

								__asm {

									MOV r12, NS4
									MOV r13, C4

									MOV rsi, pOut
									VBROADCASTSS ymm0, [rsi]
									ADD rsi, r12
									VBROADCASTSS ymm1, [rsi]
									ADD rsi, r12
									VBROADCASTSS ymm2, [rsi]

									MOV rsi, pKern
									MOV rdi, pImg

									MOV ecx, C8
									cloops :
									CMP ecx, 0
										JE cloope

										VMOVUPS ymm3, [rsi]
										ADD rsi, 32

										MOV rdx, rdi
										VMOVUPS ymm4, [rdi]
										VFMADD231PS ymm4, ymm3, ymm0
										VMOVUPS[rdi], ymm4
										ADD rdi, r13

										VMOVUPS ymm4, [rdi]
										VFMADD231PS ymm4, ymm3, ymm1
										VMOVUPS[rdi], ymm4
										ADD rdi, r13

										VMOVUPS ymm4, [rdi]
										VFMADD231PS ymm4, ymm3, ymm2
										VMOVUPS[rdi], ymm4
										MOV rdi, rdx
										ADD rdi, 32

										SUB ecx, 8
										JMP cloops
										cloope :
								}
								{
									float n1 = pOut[0 * NS + n];
									float n2 = pOut[1 * NS + n];
									float n3 = pOut[2 * NS + n];
									for (int c = Ch; c < C; c++) {
										pImg[0 * C + c] += n1 * pKern[c];
										pImg[1 * C + c] += n2 * pKern[c];
										pImg[2 * C + c] += n3 * pKern[c];
									}
								}

							}
						}
					}
				}
			}
		}

	}

	if (C != Ch) {

		int NB = (Nh / 16 < THREADS) ? 4 : 16;
		int CB = 512;

		int Nb = ceil((float)Nh / NB);
		int Cb = ceil((float)Ch / CB);

		#pragma omp parallel for
		for (int h = 0; h < H; h++) {
			float* img1 = img + h * WC;
			for (int b = 0; b < batchsize; b++) {
				float* img2 = img1 + b * HWC;
				float* out1 = out + b * NHNWN;
				for (int x = 0; x < Kx; x++) {
					int nh = (h * stride + x);
					float* kern1 = kern + x * KyNC;
					float* out2 = out1 + nh * NWN;
					for (int y = 0; y < Ky; y++) {
						float* kern2 = kern1 + y * NC;
						for (int w = 0; w < Wh; w += 3) {
							int nw = (w * stride + y);
							float* pImg = img2 + w * C;
							float* out3 = out2 + nw * N;
							for (int n = 0; n < Nh; n += 4) {

								float* pKern = kern2 + n * C;
								float* pOut = out3 + n;

								float n11 = pOut[0 * NS + 0];
								float n12 = pOut[0 * NS + 1];
								float n13 = pOut[0 * NS + 2];
								float n14 = pOut[0 * NS + 3];
								float n21 = pOut[1 * NS + 0];
								float n22 = pOut[1 * NS + 1];
								float n23 = pOut[1 * NS + 2];
								float n24 = pOut[1 * NS + 3];
								float n31 = pOut[2 * NS + 0];
								float n32 = pOut[2 * NS + 1];
								float n33 = pOut[2 * NS + 2];
								float n34 = pOut[2 * NS + 3];
								for (int c = Ch; c < C; c++) {
									float k1 = pKern[0 * C + c];
									float k2 = pKern[1 * C + c];
									float k3 = pKern[2 * C + c];
									float k4 = pKern[3 * C + c];

									pImg[0 * C + c] += n11 * k1 + n12 * k2 + n13 * k3 + n14 * k4;
									pImg[1 * C + c] += n21 * k1 + n22 * k2 + n23 * k3 + n24 * k4;
									pImg[2 * C + c] += n31 * k1 + n32 * k2 + n33 * k3 + n34 * k4;
								}

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
	int Ch = (C / 8) * 8;
	int Wh = (W / 3) * 3;
	int C8 = Ch;

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

	if (Ch != 0 && Nh != 0 && Wh != 0) {
		int T = Kx * Ky * Nh / 4;
		#pragma omp parallel for
		for (int t = 0; t < T; t++) {

			int x = t / (Ky * Nh / 4);
			int t1 = t % (Ky * Nh / 4);
			int y = t1 / (Nh / 4);
			int n = (t1 % (Nh / 4)) * 4;

			float* kern1 = kern + x * KyNC + y * NC + n * C;
			float* out1 = out + y * N + n;
			for (int b = 0; b < batchsize; b++) {
				float* img1 = img + b * HWC;
				float* out2 = out1 + b * NHNWN;
				for (int h = 0; h < H; h++) {
					int nh = (h * stride + x);
					float* img2 = img1 + h * WC;
					float* pOut = out2 + nh * NWN;
					for (int c = 0; c < Ch; c += 8) {
						float* pImg = img2 + c;
						float* pKern = kern1 + c;

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

							MOV ecx, Wh
							cloops :
							CMP ecx, 0
								JE cloope

								VMOVUPS ymm4, [rsi]
								ADD rsi, r12
								VMOVUPS ymm5, [rsi]
								ADD rsi, r12
								VMOVUPS ymm6, [rsi]
								ADD rsi, r12

								VBROADCASTSS ymm7, [rdi]
								VBROADCASTSS ymm8, [rdi + 4]
								VBROADCASTSS ymm9, [rdi + 8]
								VBROADCASTSS ymm10, [rdi + 12]
								ADD rdi, r13
								VFMADD231PS ymm0, ymm4, ymm7
								VFMADD231PS ymm1, ymm4, ymm8
								VFMADD231PS ymm2, ymm4, ymm9
								VFMADD231PS ymm3, ymm4, ymm10

								VBROADCASTSS ymm7, [rdi]
								VBROADCASTSS ymm8, [rdi + 4]
								VBROADCASTSS ymm9, [rdi + 8]
								VBROADCASTSS ymm10, [rdi + 12]
								ADD rdi, r13
								VFMADD231PS ymm0, ymm5, ymm7
								VFMADD231PS ymm1, ymm5, ymm8
								VFMADD231PS ymm2, ymm5, ymm9
								VFMADD231PS ymm3, ymm5, ymm10

								VBROADCASTSS ymm7, [rdi]
								VBROADCASTSS ymm8, [rdi + 4]
								VBROADCASTSS ymm9, [rdi + 8]
								VBROADCASTSS ymm10, [rdi + 12]
								ADD rdi, r13
								VFMADD231PS ymm0, ymm6, ymm7
								VFMADD231PS ymm1, ymm6, ymm8
								VFMADD231PS ymm2, ymm6, ymm9
								VFMADD231PS ymm3, ymm6, ymm10

								SUB ecx, 3
								JMP cloops
								cloope :

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
				}
			}
		}
	}

	if (W != Wh) {
		int Wr = W - Wh;
		int T = Kx * Ky * Nh / 4;
		#pragma omp parallel for
		for (int t = 0; t < T; t++) {

			int x = t / (Ky * Nh / 4);
			int t1 = t % (Ky * Nh / 4);
			int y = t1 / (Nh / 4);
			int n = (t1 % (Nh / 4)) * 4;

			float* kern1 = kern + x * KyNC + y * NC + n * C;
			float* out1 = out + (Wh * stride + y) * N + n;
			for (int b = 0; b < batchsize; b++) {
				float* img1 = img + b * HWC + Wh * C;
				float* out2 = out1 + b * NHNWN;
				for (int h = 0; h < H; h++) {
					int nh = (h * stride + x);
					float* img2 = img1 + h * WC;
					float* pOut = out2 + nh * NWN;
					for (int c = 0; c < Ch; c += 8) {
						float* pImg = img2 + c;
						float* pKern = kern1 + c;

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

							MOV ecx, Wr
							cloops :
							CMP ecx, 0
								JE cloope

								VMOVUPS ymm4, [rsi]
								ADD rsi, r12

								VBROADCASTSS ymm7, [rdi]
								VBROADCASTSS ymm8, [rdi + 4]
								VBROADCASTSS ymm9, [rdi + 8]
								VBROADCASTSS ymm10, [rdi + 12]
								ADD rdi, r13
								VFMADD231PS ymm0, ymm4, ymm7
								VFMADD231PS ymm1, ymm4, ymm8
								VFMADD231PS ymm2, ymm4, ymm9
								VFMADD231PS ymm3, ymm4, ymm10

								DEC ecx
								JMP cloops
								cloope :

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
					for (int c = Ch; c < C; c++) {
						float* pImg = img2 + c;
						float* pKern = kern1 + c;

						float s1, s2, s3, s4;
						s1 = s2 = s3 = s4 = 0;
						for (int w = 0; w < Wr; w++) {
							float i1 = pImg[w * C];

							float o1 = pOut[w * NS + 0];
							float o2 = pOut[w * NS + 1];
							float o3 = pOut[w * NS + 2];
							float o4 = pOut[w * NS + 3];

							s1 += i1 * o1;
							s2 += i1 * o2;
							s3 += i1 * o3;
							s4 += i1 * o4;
						}
						pKern[0 * C] += s1;
						pKern[1 * C] += s2;
						pKern[2 * C] += s3;
						pKern[3 * C] += s4;
					}
				}
			}
		}

		int Nr = (N - Nh);
		T = Kx * Ky * Nr;
		#pragma omp parallel for
		for (int t = 0; t < T; t++) {

			int x = t / (Ky * Nr);
			int t1 = t % (Ky * Nr);
			int y = t1 / Nr;
			int n = (t1 % Nr) + Nh;

			float* kern1 = kern + x * KyNC + y * NC + n * C;
			float* out1 = out + (Wh * stride + y) * N + n;
			for (int b = 0; b < batchsize; b++) {
				float* img1 = img + b * HWC + Wh * C;
				float* out2 = out1 + b * NHNWN;
				for (int h = 0; h < H; h++) {
					int nh = (h * stride + x);
					float* img2 = img1 + h * WC;
					float* pOut = out2 + nh * NWN;
					for (int c = 0; c < Ch; c += 8) {
						float* pImg = img2 + c;
						float* pKern = kern1 + c;

						__asm {

							MOV r12, C4
							MOV r13, NS4

							MOV rsi, pKern
							VMOVUPS ymm0, [rsi]

							MOV rsi, pImg
							MOV rdi, pOut

							MOV ecx, Wr
							cloops :
							CMP ecx, 0
								JE cloope

								VMOVUPS ymm4, [rsi]
								ADD rsi, r12

								VBROADCASTSS ymm7, [rdi]
								ADD rdi, r13
								VFMADD231PS ymm0, ymm4, ymm7

								DEC ecx
								JMP cloops
								cloope :

							MOV rsi, pKern
								VMOVUPS[rsi], ymm0
						}
					}
					for (int c = Ch; c < C; c++) {
						float* pImg = img2 + c;
						float* pKern = kern1 + c;

						float s1 = 0;
						for (int w = 0; w < Wr; w++) {
							float i1 = pImg[w * C];

							float o1 = pOut[w * NS];

							s1 += i1 * o1;
						}
						pKern[c] += s1;
					}
				}
			}
		}

	}
	
	if (N != Nh) {
		int Nr = (N - Nh);
		int T = Kx * Ky * Nr;
		#pragma omp parallel for
		for (int t = 0; t < T; t++) {

			int x = t / (Ky * Nr);
			int t1 = t % (Ky * Nr);
			int y = t1 / Nr;
			int n = (t1 % Nr) + Nh;

			float* kern1 = kern + x * KyNC + y * NC + n * C;
			float* out1 = out + y * N + n;
			for (int b = 0; b < batchsize; b++) {
				float* img1 = img + b * HWC;
				float* out2 = out1 + b * NHNWN;
				for (int h = 0; h < H; h++) {
					int nh = (h * stride + x);
					float* img2 = img1 + h * WC;
					float* pOut = out2 + nh * NWN;
					for (int c = 0; c < Ch; c += 8) {
						float* pImg = img2 + c;
						float* pKern = kern1 + c;

						__asm {

							MOV r12, C4
							MOV r13, NS4

							MOV rsi, pKern
							VMOVUPS ymm0, [rsi]

							MOV rsi, pImg
							MOV rdi, pOut

							MOV ecx, Wh
							cloops :
							CMP ecx, 0
								JE cloope

								VMOVUPS ymm4, [rsi]
								ADD rsi, r12
								VMOVUPS ymm5, [rsi]
								ADD rsi, r12
								VMOVUPS ymm6, [rsi]
								ADD rsi, r12

								VBROADCASTSS ymm7, [rdi]
								ADD rdi, r13
								VFMADD231PS ymm0, ymm4, ymm7

								VBROADCASTSS ymm7, [rdi]
								ADD rdi, r13
								VFMADD231PS ymm0, ymm5, ymm7

								VBROADCASTSS ymm7, [rdi]
								ADD rdi, r13
								VFMADD231PS ymm0, ymm6, ymm7

								SUB ecx, 3
								JMP cloops
								cloope :

							MOV rsi, pKern
								VMOVUPS[rsi], ymm0
						}
					}
					for (int c = Ch; c < C; c++) {
						float* pImg = img2 + c;
						float* pKern = kern1 + c;

						float s = 0;
						for (int w = 0; w < Wh; w++) {
							s += pImg[w * C] * pOut[w * NS];
						}
						pKern[0] += s;
					}
				}
			}
		}
	}

	if (C != Ch) {
		int T = Kx * Ky * Nh / 4;
		#pragma omp parallel for
		for (int t = 0; t < T; t++) {

			int x = t / (Ky * Nh / 4);
			int t1 = t % (Ky * Nh / 4);
			int y = t1 / (Nh / 4);
			int n = (t1 % (Nh / 4)) * 4;

			float* kern1 = kern + x * KyNC + y * NC + n * C;
			float* out1 = out + y * N + n;
			for (int b = 0; b < batchsize; b++) {
				float* img1 = img + b * HWC;
				float* out2 = out1 + b * NHNWN;
				for (int h = 0; h < H; h++) {
					int nh = (h * stride + x);
					float* img2 = img1 + h * WC;
					float* pOut = out2 + nh * NWN;
					for (int c = Ch; c < C; c++) {
						float* pImg = img2 + c;
						float* pKern = kern1 + c;

						float s11, s12, s13, s14;
						s11 = s12 = s13 = s14 = 0;
						for (int w = 0; w < Wh; w += 3) {
							float i1 = pImg[(w + 0) * C];
							float i2 = pImg[(w + 1) * C];
							float i3 = pImg[(w + 2) * C];

							float o1 = pOut[(w + 0) * NS + 0];
							float o2 = pOut[(w + 0) * NS + 1];
							float o3 = pOut[(w + 0) * NS + 2];
							float o4 = pOut[(w + 0) * NS + 3];

							s11 += i1 * o1;
							s12 += i1 * o2;
							s13 += i1 * o3;
							s14 += i1 * o4;

							o1 = pOut[(w + 1) * NS + 0];
							o2 = pOut[(w + 1) * NS + 1];
							o3 = pOut[(w + 1) * NS + 2];
							o4 = pOut[(w + 1) * NS + 3];

							s11 += i2 * o1;
							s12 += i2 * o2;
							s13 += i2 * o3;
							s14 += i2 * o4;

							o1 = pOut[(w + 2) * NS + 0];
							o2 = pOut[(w + 2) * NS + 1];
							o3 = pOut[(w + 2) * NS + 2];
							o4 = pOut[(w + 2) * NS + 3];

							s11 += i2 * o1;
							s12 += i2 * o2;
							s13 += i2 * o3;
							s14 += i2 * o4;
						}
						pKern[0 * C] += s11;
						pKern[1 * C] += s12;
						pKern[2 * C] += s13;
						pKern[3 * C] += s14;
					}
				}
			}
		}
	}
}
