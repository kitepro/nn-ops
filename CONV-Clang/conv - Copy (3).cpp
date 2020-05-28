#include <immintrin.h>
#include <time.h>
#include <omp.h>
#include <iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

#define THREADS 16

DLLEXPORT float* avx2_malloc(unsigned long long size) {
	return (float*)_aligned_malloc(size, 32);
}


DLLEXPORT void avx2_conv3d(float* img, float* kern, float* out, int H, int W, int C, int N, int Kx, int Ky, int stride, int padding) {

	int NH = (H - Kx) / stride + 1;
	int NW = (W - Ky) / stride + 1;

	memset(out, 0, sizeof(float) * N * NH * NW);


	int NWh = (NW / 3) * 3;
	int Nh = (N / 4) * 4;
	int Ch = (C / 8) * 8;
	int C8 = Ch;

	if (Ch != 0 && NWh != 0 && Nh != 0) {

		int NB = (Nh / 16 < THREADS) ? 4 : 16;
		int CB = 512;
		int NWB = 3;

		int Nb = ceil((float)Nh / NB);
		int Cb = ceil((float)Ch / CB);
		int NWb = ceil((float)NWh / NWB);

		int WCB = W * CB;
		int HWCB = H * WCB;
		int NWNB = NW * NB;
		int NHNWNB = NH * NWNB;
		int NBCB = CB * NB;
		int KyNBCB = Ky * NBCB;
		int KxKyNBCB = Kx * KyNBCB;
		int CbKxKyNBCB = Cb * KxKyNBCB;

		int WC = W * C;
		int NWN = NW * N;
		int NC = N * C;
		int KyNC = Ky * NC;

		#pragma omp parallel for
		for (int nb = 0; nb < Nb; nb++) {
			float* mt = (float*)_aligned_malloc(sizeof(float) * 8 * 12, 32);
			int Nl = nb * NB;
			int Nm = (Nl + NB > Nh) ? Nh - Nl : NB;
			for (int cb = 0; cb < Cb; cb++) {
				int Cl = cb * CB;
				int Cm = (Cl + CB > Ch) ? Ch - Cl : CB;
				for (int nh = 0; nh < NH; nh++) {
					for (int nwb = 0; nwb < NWb; nwb++) {
						int NWl = nwb * NWB;
						int NWm = (NWl + NWB > NWh) ? NWh : NWl + NWB;
						for (int x = 0; x < Kx; x++) {
							for (int y = 0; y < Ky; y++) {
								for (int nw = NWl; nw < NWm; nw += 3) {
									for (int n = 0; n < Nm; n += 4) {

										float* pImg = img + (nh * stride + x) * WC + (nw * stride + y) * C + Cl;
										float* pOut = out + nh * NWN + nw * N + Nl + n;
										float* pKern = kern + x * KyNC + y * NC + (Nl + n) * C + Cl;

										__asm {

											MOV rax, 0
											MOV eax, C
											MOV ecx, 4
											MUL ecx
											MOV r12, rax

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
											VFMADD231PS ymm0, ymm12, ymm15
											VFMADD231PS ymm1, ymm13, ymm15
											VFMADD231PS ymm2, ymm14, ymm15

											VMOVUPS ymm15, [rdi]
											ADD rdi, r12
											VFMADD231PS ymm3, ymm12, ymm15
											VFMADD231PS ymm4, ymm13, ymm15
											VFMADD231PS ymm5, ymm14, ymm15

											VMOVUPS ymm15, [rdi]
											ADD rdi, r12
											VFMADD231PS ymm6, ymm12, ymm15
											VFMADD231PS ymm7, ymm13, ymm15
											VFMADD231PS ymm8, ymm14, ymm15

											VMOVUPS ymm15, [rdi]
											VFMADD231PS ymm9, ymm12, ymm15
											VFMADD231PS ymm10, ymm13, ymm15
											VFMADD231PS ymm11, ymm14, ymm15

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
											pOut[0 * N + 0] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[1 * N + 0] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[2 * N + 0] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[0 * N + 1] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[1 * N + 1] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[2 * N + 1] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[0 * N + 2] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[1 * N + 2] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[2 * N + 2] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[0 * N + 3] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[1 * N + 3] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
											t += 8;
											pOut[2 * N + 3] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
										}
									}
								}
							}
						}
					}
				}
			}
		}


		/*
		#pragma omp parallel for
		for (int nh = 0; nh < NH; nh++) {
			float* mt = (float*)_aligned_malloc(sizeof(float) * 8 * 3 * 4, 32);
			float* pout1 = out + nh * NW * N;
			for (int x = 0; x < Kx; x++) {
				float* pkern1 = kern + x * Ky * N * C;
				float* pimg1 = img + (nh * stride + x) * W * C;
				for (int y = 0; y < Ky; y++) {
					float* pkern2 = pkern1 + y * N * C;
					for (int nw = 0; nw < NWh; nw += 3) {
						float* pimg2 = pimg1 + (nw * stride + y) * C;
						float* pout2 = pout1 + nw * N;
						for (int n = 0; n < Nh; n += 4) {
							float* pkern3 = pkern2 + n * C;
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

								MOV rdi, pkern3
								MOV rsi, pimg2
								MOV rax, 0
								MOV eax, C
								MOV ecx, 4
								MUL ecx
								MOV r12, rax


								MOV ecx, C8
								loop_c_begin :
								CMP ecx, 0
									JE loop_c_end

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
									VFMADD231PS ymm0, ymm12, ymm15
									VFMADD231PS ymm1, ymm13, ymm15
									VFMADD231PS ymm2, ymm14, ymm15

									VMOVUPS ymm15, [rdi]
									ADD rdi, r12
									VFMADD231PS ymm3, ymm12, ymm15
									VFMADD231PS ymm4, ymm13, ymm15
									VFMADD231PS ymm5, ymm14, ymm15

									VMOVUPS ymm15, [rdi]
									ADD rdi, r12
									VFMADD231PS ymm6, ymm12, ymm15
									VFMADD231PS ymm7, ymm13, ymm15
									VFMADD231PS ymm8, ymm14, ymm15

									VMOVUPS ymm15, [rdi]
									VFMADD231PS ymm9, ymm12, ymm15
									VFMADD231PS ymm10, ymm13, ymm15
									VFMADD231PS ymm11, ymm14, ymm15

									MOV rdi, rdx
									ADD rdi, 32

									SUB ecx, 8
									JMP loop_c_begin
									loop_c_end :

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
								pout2[0 * N + 0] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pout2[1 * N + 0] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pout2[2 * N + 0] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pout2[0 * N + 1] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pout2[1 * N + 1] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pout2[2 * N + 1] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pout2[0 * N + 2] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pout2[1 * N + 2] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pout2[2 * N + 2] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pout2[0 * N + 3] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pout2[1 * N + 3] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pout2[2 * N + 3] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							}
							pout2 += 4;
						}
					}
				}
			}
		}
		*/

	}

	if (NW != NWh) {
		int Nh = (N / 7) * 7;
		int C8 = (C / 8) * 8;
		for (int x = 0; x < Kx; x++) {
			for (int y = 0; y < Ky; y++) {
				#pragma omp parallel for
				for (int nh = 0; nh < NH; nh++) {
					float* mt = (float*)_aligned_malloc(sizeof(float) * 8 * 7, 32);
					for (int nw = NWh; nw < NW; nw++) {
						for (int n = 0; n < Nh; n += 7) {
							float* pImg = img + (nh * stride + x) * W * C + (nw * stride + y) * C;
							float* pKern = kern + x * Ky * N * C + y * N * C + n * C;
							float* pOut = out + nh * NW * N + nw * N + n;
							{
								__asm {

									VXORPS ymm0, ymm0, ymm0
									VXORPS ymm1, ymm1, ymm1
									VXORPS ymm2, ymm2, ymm2
									VXORPS ymm3, ymm3, ymm3
									VXORPS ymm4, ymm4, ymm4
									VXORPS ymm5, ymm5, ymm5
									VXORPS ymm6, ymm6, ymm6

									MOV rax, 0
									MOV eax, C
									MOV ecx, 4
									MUL ecx
									MOV r12, rax

									MOV rsi, pImg
									MOV rdi, pKern

									MOV ecx, C8
									cloops :
									CMP ecx, 0
										JE cloope

										VMOVUPS ymm7, [rsi]
										ADD rsi, 32

										MOV rdx, rdi
										VMOVUPS ymm8, [rdi]
										ADD rdi, r12
										VMOVUPS ymm9, [rdi]
										ADD rdi, r12
										VMOVUPS ymm10, [rdi]
										ADD rdi, r12
										VMOVUPS ymm11, [rdi]
										ADD rdi, r12
										VMOVUPS ymm12, [rdi]
										ADD rdi, r12
										VMOVUPS ymm13, [rdi]
										ADD rdi, r12
										VMOVUPS ymm14, [rdi]
										MOV rdi, rdx
										ADD rdi, 32

										VFMADD231PS ymm0, ymm7, ymm8
										VFMADD231PS ymm1, ymm7, ymm9
										VFMADD231PS ymm2, ymm7, ymm10
										VFMADD231PS ymm3, ymm7, ymm11
										VFMADD231PS ymm4, ymm7, ymm12
										VFMADD231PS ymm5, ymm7, ymm13
										VFMADD231PS ymm6, ymm7, ymm14

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
								}
								float* t = mt;
								pOut[0] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pOut[1] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pOut[2] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pOut[3] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pOut[4] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pOut[5] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pOut[6] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							}
							for (int c = Ch; c < C; c++) {
								float kt = pImg[c];
								pOut[0] += kt * pKern[0 * C + c];
								pOut[1] += kt * pKern[1 * C + c];
								pOut[2] += kt * pKern[2 * C + c];
								pOut[3] += kt * pKern[3 * C + c];
								pOut[4] += kt * pKern[4 * C + c];
								pOut[5] += kt * pKern[5 * C + c];
								pOut[6] += kt * pKern[6 * C + c];
							}
						}
						for (int n = Nh; n < N; n++) {
							float* pImg = img + (nh * stride + x) * W * C + (nw * stride + y) * C;
							float* pKern = kern + x * Ky * N * C + y * N * C + n * C;
							float* pOut = out + nh * NW * N + nw * N + n;
							__asm {

								VXORPS ymm0, ymm0, ymm0

								MOV rsi, pImg
								MOV rdi, pKern

								MOV ecx, C8
								cloops:
								CMP ecx, 0
								JE cloope

								VMOVUPS ymm7, [rsi]
								ADD rsi, 32

								VMOVUPS ymm8, [rdi]
								ADD rdi, 32

								VFMADD231PS ymm0, ymm7, ymm8

								SUB ecx, 8
								JMP cloops
								cloope:

								MOV rsi, mt
								VMOVUPS[rsi], ymm0
							}
							float* t = mt;
							pOut[0] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							for (int c = Ch; c < C; c++) {
								pOut[0] += pImg[c] * pKern[c];
							}
						}
					}
				}
			}
		}
	}

	if (N != Nh) {
		int NWhh = (NWh / 7) * 7;
		int C8 = (C / 8) * 8;
		for (int x = 0; x < Kx; x++) {
			for (int y = 0; y < Ky; y++) {
				#pragma omp parallel for
				for (int nh = 0; nh < NH; nh++) {
					float* mt = (float*)_aligned_malloc(sizeof(float) * 8 * 7, 32);
					for (int n = Nh; n < N; n++) {
						for (int nw = 0; nw < NWhh; nw += 7) {
							float* pImg = img + (nh * stride + x) * W * C + (nw * stride + y) * C;
							float* pKern = kern + x * Ky * N * C + y * N * C + n * C;
							float* pOut = out + nh * NW * N + nw * N + n;
							{
								__asm {

									VXORPS ymm0, ymm0, ymm0
									VXORPS ymm1, ymm1, ymm1
									VXORPS ymm2, ymm2, ymm2
									VXORPS ymm3, ymm3, ymm3
									VXORPS ymm4, ymm4, ymm4
									VXORPS ymm5, ymm5, ymm5
									VXORPS ymm6, ymm6, ymm6

									MOV rax, 0
									MOV eax, C
									MOV ecx, 4
									MUL ecx
									MOV ecx, stride
									MUL ecx
									MOV r12, rax

									MOV rsi, pImg
									MOV rdi, pKern

									MOV ecx, C8
									cloops :
									CMP ecx, 0
										JE cloope

										VMOVUPS ymm7, [rdi]
										ADD rdi, 32

										MOV rdx, rsi
										VMOVUPS ymm8, [rsi]
										ADD rsi, r12
										VMOVUPS ymm9, [rsi]
										ADD rsi, r12
										VMOVUPS ymm10, [rsi]
										ADD rsi, r12
										VMOVUPS ymm11, [rsi]
										ADD rsi, r12
										VMOVUPS ymm12, [rsi]
										ADD rsi, r12
										VMOVUPS ymm13, [rsi]
										ADD rsi, r12
										VMOVUPS ymm14, [rsi]
										MOV rsi, rdx
										ADD rsi, 32

										VFMADD231PS ymm0, ymm7, ymm8
										VFMADD231PS ymm1, ymm7, ymm9
										VFMADD231PS ymm2, ymm7, ymm10
										VFMADD231PS ymm3, ymm7, ymm11
										VFMADD231PS ymm4, ymm7, ymm12
										VFMADD231PS ymm5, ymm7, ymm13
										VFMADD231PS ymm6, ymm7, ymm14

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
								}
								float* t = mt;
								pOut[0 * N] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pOut[1 * N] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pOut[2 * N] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pOut[3 * N] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pOut[4 * N] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pOut[5 * N] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
								t += 8;
								pOut[6 * N] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							}
							for (int c = Ch; c < C; c++) {
								float kt = pKern[c];
								pOut[0 * N] += kt * pImg[0 * stride * C + c];
								pOut[1 * N] += kt * pImg[1 * stride * C + c];
								pOut[2 * N] += kt * pImg[2 * stride * C + c];
								pOut[3 * N] += kt * pImg[3 * stride * C + c];
								pOut[4 * N] += kt * pImg[4 * stride * C + c];
								pOut[5 * N] += kt * pImg[5 * stride * C + c];
								pOut[6 * N] += kt * pImg[6 * stride * C + c];
							}
						}
						for (int nw = NWhh; nw < NWh; nw++) {
							float* pImg = img + (nh * stride + x) * W * C + (nw * stride + y) * C;
							float* pKern = kern + x * Ky * N * C + y * N * C + n * C;
							float* pOut = out + nh * NW * N + nw * N + n;
							{
								__asm {

									VXORPS ymm0, ymm0, ymm0

									MOV rsi, pImg
									MOV rdi, pKern

									MOV ecx, C8
									cloops :
									CMP ecx, 0
										JE cloope

										VMOVUPS ymm7, [rdi]
										ADD rdi, 32

										VMOVUPS ymm8, [rsi]
										ADD rsi, 32

										VFMADD231PS ymm0, ymm7, ymm8

										SUB ecx, 8
										JMP cloops
										cloope :

									MOV rsi, mt
										VMOVUPS[rsi], ymm0
								}
								float* t = mt;
								pOut[0] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
							}
							for (int c = Ch; c < C; c++) {
								pOut[0] += pKern[c] * pImg[c];
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
			float* mt = (float*)_aligned_malloc(sizeof(float) * 8 * 7, 32);
			for (int x = 0; x < Kx; x++) {
				for (int y = 0; y < Ky; y++) {
					for (int nw = 0; nw < NWh; nw++) {
						float* pImg = img + (nh * stride + x) * W * C + (nw * stride + y) * C;
						float* pOut = out + nh * NW * N + nw * N;
						for (int n = 0; n < Nh; n++) {
							float* pKern = kern + x * Ky * N * C + y * N * C + n * C;
							float s = 0;
							for (int c = Ch; c < C; c++) {
								s += pImg[c] * pKern[c];
							}
							pOut[n] += s;
						}
					}
				}
			}
		}
	}
}


DLLEXPORT void avx2_conv3d_back_i(float* img, float* kern, float* out, int H, int W, int C, int N, int Kx, int Ky, int stride, int padding) {

	int NH = (H - Kx) / stride + 1;
	int NW = (W - Ky) / stride + 1;

	memset(img, 0, sizeof(float) * C * H * W);

	int NWh = (NW / 6) * 6;
	int Ch = (C / 16) * 16;

	int NWN = NW * N;
	int NC = N * C;
	int WC = W * C;
	int KyNC = Ky * NC;

	if (Ch != 0 && NWh != 0) {
		#pragma omp parallel for
		for (int nh = 0; nh < NH; nh++) {
			float* out1 = out + nh * NWN;
			for (int x = 0; x < Kx; x++) {
				float* img1 = img + (nh * stride + x) * WC;
				float* kern1 = kern + (x * KyNC);
				for (int y = 0; y < Ky; y++) {
					float* kern2 = kern1 + (y * NC);
					for (int nw = 0; nw < NWh; nw += 6) {
						float* img2 = img1 + (nw * stride + y) * C;
						float* pOut = out1 + nw * N;
						for (int c = 0; c < Ch; c += 16) {

							float* pImg = img2 + c;
							float* pKern = kern2 + c;

							__asm {

								MOV rax, 0
								MOV eax, C
								MOV ecx, 4
								MUL ecx
								MOV r12, rax
								MOV ecx, stride
								MUL ecx
								MOV r13, rax

								MOV rsi, pImg
								VMOVUPS ymm0, [rsi]
								VMOVUPS ymm1, [rsi + 32]
								ADD rsi, r13
								VMOVUPS ymm2, [rsi]
								VMOVUPS ymm3, [rsi + 32]
								ADD rsi, r13
								VMOVUPS ymm4, [rsi]
								VMOVUPS ymm5, [rsi + 32]
								ADD rsi, r13
								VMOVUPS ymm6, [rsi]
								VMOVUPS ymm7, [rsi + 32]
								ADD rsi, r13
								VMOVUPS ymm8, [rsi]
								VMOVUPS ymm9, [rsi + 32]
								ADD rsi, r13
								VMOVUPS ymm10, [rsi]
								VMOVUPS ymm11, [rsi + 32]

								MOV rax, 0
								MOV eax, N
								MOV ecx, 4
								MUL ecx

								MOV rsi, pKern
								MOV rdi, pOut

								MOV ecx, N
								nloops :
								CMP ecx, 0
									JE nloope

									VMOVUPS ymm12, [rsi]
									VMOVUPS ymm13, [rsi + 32]
									ADD rsi, r12

									MOV rdx, rdi
									VBROADCASTSS ymm14, [rdi]
									ADD rdi, rax
									VFMADD231PS ymm0, ymm12, ymm14
									VFMADD231PS ymm1, ymm13, ymm14

									VBROADCASTSS ymm14, [rdi]
									ADD rdi, rax
									VFMADD231PS ymm2, ymm12, ymm14
									VFMADD231PS ymm3, ymm13, ymm14

									VBROADCASTSS ymm14, [rdi]
									ADD rdi, rax
									VFMADD231PS ymm4, ymm12, ymm14
									VFMADD231PS ymm5, ymm13, ymm14

									VBROADCASTSS ymm14, [rdi]
									ADD rdi, rax
									VFMADD231PS ymm6, ymm12, ymm14
									VFMADD231PS ymm7, ymm13, ymm14

									VBROADCASTSS ymm14, [rdi]
									ADD rdi, rax
									VFMADD231PS ymm8, ymm12, ymm14
									VFMADD231PS ymm9, ymm13, ymm14

									VBROADCASTSS ymm14, [rdi]
									VFMADD231PS ymm10, ymm12, ymm14
									VFMADD231PS ymm11, ymm13, ymm14

									MOV rdi, rdx
									ADD rdi, 4

									DEC ecx
									JMP nloops
									nloope :

								MOV rsi, pImg
									VMOVUPS[rsi], ymm0
									VMOVUPS[rsi + 32], ymm1
									ADD rsi, r13
									VMOVUPS[rsi], ymm2
									VMOVUPS[rsi + 32], ymm3
									ADD rsi, r13
									VMOVUPS[rsi], ymm4
									VMOVUPS[rsi + 32], ymm5
									ADD rsi, r13
									VMOVUPS[rsi], ymm6
									VMOVUPS[rsi + 32], ymm7
									ADD rsi, r13
									VMOVUPS[rsi], ymm8
									VMOVUPS[rsi + 32], ymm9
									ADD rsi, r13
									VMOVUPS[rsi], ymm10
									VMOVUPS[rsi + 32], ymm11

							}
						}
					}
				}
			}
		}
	}

	if (NW != NWh) {
		#pragma omp parallel for
		for (int nh = 0; nh < NH; nh++) {
			float* out1 = out + nh * NWN;
			for (int x = 0; x < Kx; x++) {
				float* img1 = img + (nh * stride + x) * WC;
				float* kern1 = kern + (x * KyNC);
				for (int y = 0; y < Ky; y++) {
					float* kern2 = kern1 + (y * NC);
					for (int nw = NWh; nw < NW; nw++) {
						float* img2 = img1 + (nw * stride + y) * C;
						float* pOut = out1 + nw * N;
						for (int c = 0; c < Ch; c += 16) {

							float* pImg = img2 + c;
							float* pKern = kern2 + c;

							__asm {

								MOV rax, 0
								MOV eax, C
								MOV ecx, 4
								MUL ecx
								MOV r12, rax

								MOV rsi, pImg
								VMOVUPS ymm0, [rsi]
								VMOVUPS ymm1, [rsi + 32]

								MOV rsi, pKern
								MOV rdi, pOut

								MOV ecx, N
								nloops :
								CMP ecx, 0
									JE nloope

									VMOVUPS ymm12, [rsi]
									VMOVUPS ymm13, [rsi + 32]
									ADD rsi, r12

									VBROADCASTSS ymm14, [rdi]
									VFMADD231PS ymm0, ymm12, ymm14
									VFMADD231PS ymm1, ymm13, ymm14
									ADD rdi, 4

									DEC ecx
									JMP nloops
									nloope :

								MOV rsi, pImg
									VMOVUPS[rsi], ymm0
									VMOVUPS[rsi + 32], ymm1
							}
						}
						for (int c = Ch; c < C; c++) {
							float s = 0;
							for (int n = 0; n < N; n++) {
								s += pOut[n] * kern2[n * C + c];
							}
							img2[c] += s;
						}
					}
				}
			}
		}
	}

	if (C != Ch) {
		#pragma omp parallel for
		for (int nh = 0; nh < NH; nh++) {
			float* out1 = out + nh * NWN;
			for (int x = 0; x < Kx; x++) {
				float* img1 = img + (nh * stride + x) * WC;
				float* kern1 = kern + (x * KyNC);
				for (int y = 0; y < Ky; y++) {
					float* kern2 = kern1 + (y * NC);
					for (int nw = 0; nw < NWh; nw += 6) {
						float* img2 = img1 + (nw * stride + y) * C;
						float* pOut = out1 + nw * N;
						for (int c = Ch; c < C; c++) {
							float* pKern = kern2 + c;
							float s1, s2, s3, s4, s5, s6;
							s1 = s2 = s3 = s4 = s5 = s6 = 0;
							for (int n = 0; n < N; n++) {
								float t = pKern[n * C];
								s1 += pOut[0 * N + n] * t;
								s2 += pOut[1 * N + n] * t;
								s3 += pOut[2 * N + n] * t;
								s4 += pOut[3 * N + n] * t;
								s5 += pOut[4 * N + n] * t;
								s6 += pOut[5 * N + n] * t;
							}
							img2[0 * C * stride + c] += s1;
							img2[1 * C * stride + c] += s2;
							img2[2 * C * stride + c] += s3;
							img2[3 * C * stride + c] += s4;
							img2[4 * C * stride + c] += s5;
							img2[5 * C * stride + c] += s6;
						}
					}
				}
			}
		}
	}

}


DLLEXPORT void avx2_conv3d_back_k(float* img, float* kern, float* out, int H, int W, int C, int N, int Kx, int Ky, int stride, int padding) {

	int NH = (H - Kx) / stride + 1;
	int NW = (W - Ky) / stride + 1;

	//memset(kern, 0, sizeof(float) * Kx * Ky * N * C);

	int Nh = (N / 6) * 6;
	int Ch = (C / 16) * 16;

	if (Ch != 0 && Nh != 0) {
		int T = Kx * Ky * Nh / 6;
		#pragma omp parallel for
		for (int t = 0; t < T; t++) {
			int x = t / (Ky * Nh / 6);
			int t1 = t % (Ky * Nh / 6);
			int y = t1 / (Nh / 6);
			int nb = t1 % (Nh / 6);
			int n = nb * 6;

			float* kern1 = kern + (x * Ky * N * C) + (y * N * C) + (n * C);
			for (int nh = 0; nh < NH; nh++) {
				float* img1 = img + (nh + x) * W * C + y * C;
				float* pOut = out + nh * NW * N + n;
				for (int c = 0; c < Ch; c += 16) {

					float* pImg = img1 + c;
					float* pKern = kern1 + c;

					__asm {

						MOV rax, 0
						MOV eax, C
						MOV ecx, 4
						MUL ecx
						MOV r12, rax
						MOV ecx, stride
						MUL ecx
						MOV r14, rax

						MOV rax, 0
						MOV eax, N
						MUL ecx
						MOV r13, rax

						MOV rsi, pKern
						VMOVUPS ymm0, [rsi]
						VMOVUPS ymm1, [rsi + 32]
						ADD rsi, r12
						VMOVUPS ymm2, [rsi]
						VMOVUPS ymm3, [rsi + 32]
						ADD rsi, r12
						VMOVUPS ymm4, [rsi]
						VMOVUPS ymm5, [rsi + 32]
						ADD rsi, r12
						VMOVUPS ymm6, [rsi]
						VMOVUPS ymm7, [rsi + 32]
						ADD rsi, r12
						VMOVUPS ymm8, [rsi]
						VMOVUPS ymm9, [rsi + 32]
						ADD rsi, r12
						VMOVUPS ymm10, [rsi]
						VMOVUPS ymm11, [rsi + 32]

						MOV rsi, pImg
						MOV rdi, pOut

						MOV ecx, NW
						wloops :
						CMP ecx, 0
							JE wloope

							VMOVUPS ymm12, [rsi]
							VMOVUPS ymm13, [rsi + 32]
							ADD rsi, r14

							VBROADCASTSS ymm14, [rdi]
							VFMADD231PS ymm0, ymm14, ymm12
							VFMADD231PS ymm1, ymm14, ymm13

							VBROADCASTSS ymm14, [rdi + 4]
							VFMADD231PS ymm2, ymm14, ymm12
							VFMADD231PS ymm3, ymm14, ymm13

							VBROADCASTSS ymm14, [rdi + 8]
							VFMADD231PS ymm4, ymm14, ymm12
							VFMADD231PS ymm5, ymm14, ymm13

							VBROADCASTSS ymm14, [rdi + 12]
							VFMADD231PS ymm6, ymm14, ymm12
							VFMADD231PS ymm7, ymm14, ymm13

							VBROADCASTSS ymm14, [rdi + 16]
							VFMADD231PS ymm8, ymm14, ymm12
							VFMADD231PS ymm9, ymm14, ymm13

							VBROADCASTSS ymm14, [rdi + 20]
							VFMADD231PS ymm10, ymm14, ymm12
							VFMADD231PS ymm11, ymm14, ymm13

							ADD rdi, r13

							DEC ecx
							JMP wloops
							wloope :


						MOV rsi, pKern
							VMOVUPS[rsi], ymm0
							VMOVUPS[rsi + 32], ymm1
							ADD rsi, r12
							VMOVUPS[rsi], ymm2
							VMOVUPS[rsi + 32], ymm3
							ADD rsi, r12
							VMOVUPS[rsi], ymm4
							VMOVUPS[rsi + 32], ymm5
							ADD rsi, r12
							VMOVUPS[rsi], ymm6
							VMOVUPS[rsi + 32], ymm7
							ADD rsi, r12
							VMOVUPS[rsi], ymm8
							VMOVUPS[rsi + 32], ymm9
							ADD rsi, r12
							VMOVUPS[rsi], ymm10
							VMOVUPS[rsi + 32], ymm11

					}
				}
			}
		}
	}

	if (N != Nh) {
		int T = Kx * Ky * (N - Nh);
		#pragma omp parallel for
		for (int t = 0; t < T; t++) {
			int x = t / (Ky * (N - Nh));
			int t1 = t % (Ky * (N - Nh));
			int y = t1 / (N - Nh);
			int nb = t1 % (N - Nh);
			int n = Nh + nb;

			float* kern1 = kern + (x * Ky * N * C) + (y * N * C) + (n * C);
			for (int nh = 0; nh < NH; nh++) {
				float* img1 = img + (nh + x) * W * C + y * C;
				float* pOut = out + nh * NW * N + n;
				for (int c = 0; c < Ch; c += 16) {

					float* pImg = img1 + c;
					float* pKern = kern1 + c;

					__asm {

						MOV rax, 0
						MOV eax, C
						MOV ecx, 4
						MUL ecx
						MOV ecx, stride
						MUL ecx
						MOV r14, rax

						MOV rax, 0
						MOV eax, N
						MOV ecx, 4
						MUL ecx
						MOV r13, rax

						MOV rsi, pKern
						VMOVUPS ymm0, [rsi]
						VMOVUPS ymm1, [rsi + 32]

						MOV rsi, pImg
						MOV rdi, pOut

						MOV ecx, NW
						wloops :
						CMP ecx, 0
							JE wloope

							VMOVUPS ymm12, [rsi]
							VMOVUPS ymm13, [rsi + 32]
							ADD rsi, r14

							VBROADCASTSS ymm14, [rdi]
							VFMADD231PS ymm0, ymm14, ymm12
							VFMADD231PS ymm1, ymm14, ymm13

							ADD rdi, r13

							DEC ecx
							JMP wloops
							wloope :


						MOV rsi, pKern
							VMOVUPS[rsi], ymm0
							VMOVUPS[rsi + 32], ymm1

					}
				}
				for (int c = Ch; c < C; c++) {
					float s = 0;
					for (int nw = 0; nw < NW; nw++) {
						s += pOut[nw * N] * img1[nw * stride * C + c];
					}
					kern1[c] += s;
				}
			}
		}
	}

	if (C != Ch) {
		int T = Kx * Ky * Nh / 6;
		#pragma omp parallel for
		for (int t = 0; t < T; t++) {
			int x = t / (Ky * Nh / 6);
			int t1 = t % (Ky * Nh / 6);
			int y = t1 / (Nh / 6);
			int nb = t1 % (Nh / 6);
			int n = nb * 6;

			float* kern1 = kern + (x * Ky * N * C) + (y * N * C) + (n * C);
			for (int nh = 0; nh < NH; nh++) {
				float* img1 = img + (nh + x) * W * C + y * C;
				float* pOut = out + nh * NW * N + n;
				for (int c = Ch; c < C; c++) {
					float s = 0;
					for (int nw = 0; nw < NW; nw++) {
						s += img1[nw * stride * C + c] * pOut[nw * N];
					}
					kern1[c] += s;
				}
			}
		}
	}

}