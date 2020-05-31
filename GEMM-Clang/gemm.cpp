#include <immintrin.h>
#include <time.h>
#include <omp.h>
#include <iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

#define THREADS 16

DLLEXPORT float* avx2_malloc(unsigned long long size) {
	return (float*)_aligned_malloc(size, 32);
}

DLLEXPORT void avx2_gemm(float* A, float* B, float* C, int M, int N, int K, int trans_a, int trans_b, float* tempmem) {

	memset(C, 0, sizeof(float) * M * N);

	int MB = 126;
	int KB = 1024;
	int MR = 6;
	int NR = 16;

	int Mb = M / MB;
	int Kb = K / KB;

	if (Mb < THREADS) {
		MB = 36;
		Mb = M / MB;
	}

	int Nr = N / NR;
	int Mr = MB / MR;

	int Mh = M % MB;
	int Nh = N % NR;
	int Kh = K % KB;

	if (trans_a <= 0 && trans_b <= 0) {
		float* Bp = tempmem;// (float*)_aligned_malloc(sizeof(float) * Nr * KB * NR, 32);
		for (int kb = 0; kb < Kb; kb++) {

			#pragma omp parallel for
			for (int k = 0; k < KB; k++) {
				for (int nr = 0; nr < Nr; nr++) {
					for (int n = 0; n < NR; n++) {
						Bp[nr * KB * NR + k * NR + n] = B[(kb * KB + k) * N + nr * NR + n];
					}
				}
			}

			#pragma omp parallel for
			for (int mb = 0; mb < Mb; mb++) {
				float* Ap = tempmem + Nr * KB * NR + mb * Mr * KB * MR;// (float*)_aligned_malloc(sizeof(float) * Mr * KB * MR, 32);

				for (int mr = 0; mr < Mr; mr++) {
					for (int m = 0; m < MR; m++) {
						for (int k = 0; k < KB; k++) {
							Ap[mr * KB * MR + k * MR + m] = A[(mb * MB + mr * MR + m) * K + (kb * KB + k)];
						}
					}
				}

				for (int nr = 0; nr < Nr; nr++) {
					for (int mr = 0; mr < Mr; mr++) {

						float* pA = Ap + (mr * KB * MR);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + (mb * MB + mr * MR) * N + (nr * NR);

						// 6x16 Asm
						__asm {

							MOV rax, 0
							MOV eax, N
							MOV ecx, 4
							MUL ecx

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm2, [rsi]
							VMOVUPS ymm3, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm4, [rsi]
							VMOVUPS ymm5, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm6, [rsi]
							VMOVUPS ymm7, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm8, [rsi]
							VMOVUPS ymm9, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm10, [rsi]
							VMOVUPS ymm11, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, KB
							kloops :
							CMP ecx, 0
								JE kloope

								VMOVUPS ymm12, [rdi]
								VMOVUPS ymm13, [rdi + 32]

								VBROADCASTSS ymm14, [rsi]
								VBROADCASTSS ymm15, [rsi + 4]
								VFMADD231PS ymm0, ymm14, ymm12
								VFMADD231PS ymm1, ymm14, ymm13
								VFMADD231PS ymm2, ymm15, ymm12
								VFMADD231PS ymm3, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 8]
								VBROADCASTSS ymm15, [rsi + 12]
								VFMADD231PS ymm4, ymm14, ymm12
								VFMADD231PS ymm5, ymm14, ymm13
								VFMADD231PS ymm6, ymm15, ymm12
								VFMADD231PS ymm7, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 16]
								VBROADCASTSS ymm15, [rsi + 20]
								VFMADD231PS ymm8, ymm14, ymm12
								VFMADD231PS ymm9, ymm14, ymm13
								VFMADD231PS ymm10, ymm15, ymm12
								VFMADD231PS ymm11, ymm15, ymm13

								ADD rsi, 24
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm1
								ADD rsi, rax
								VMOVUPS[rsi], ymm2
								VMOVUPS[rsi + 32], ymm3
								ADD rsi, rax
								VMOVUPS[rsi], ymm4
								VMOVUPS[rsi + 32], ymm5
								ADD rsi, rax
								VMOVUPS[rsi], ymm6
								VMOVUPS[rsi + 32], ymm7
								ADD rsi, rax
								VMOVUPS[rsi], ymm8
								VMOVUPS[rsi + 32], ymm9
								ADD rsi, rax
								VMOVUPS[rsi], ymm10
								VMOVUPS[rsi + 32], ymm11
						}

					}
				}
				for (int n = 0; n < Nh; n++) {
					for (int mr = 0; mr < Mr; mr++) {
						float s0, s1, s2, s3, s4, s5;
						s0 = s1 = s2 = s3 = s4 = s5 = 0;
						for (int k = 0; k < KB; k++) {
							float t = B[(kb * KB + k) * N + Nr * NR + n];
							s0 += t * Ap[mr * KB * MR + k * MR + 0];
							s1 += t * Ap[mr * KB * MR + k * MR + 1];
							s2 += t * Ap[mr * KB * MR + k * MR + 2];
							s3 += t * Ap[mr * KB * MR + k * MR + 3];
							s4 += t * Ap[mr * KB * MR + k * MR + 4];
							s5 += t * Ap[mr * KB * MR + k * MR + 5];
						}
						C[(mb * MB + mr * MR + 0) * N + Nr * NR + n] += s0;
						C[(mb * MB + mr * MR + 1) * N + Nr * NR + n] += s1;
						C[(mb * MB + mr * MR + 2) * N + Nr * NR + n] += s2;
						C[(mb * MB + mr * MR + 3) * N + Nr * NR + n] += s3;
						C[(mb * MB + mr * MR + 4) * N + Nr * NR + n] += s4;
						C[(mb * MB + mr * MR + 5) * N + Nr * NR + n] += s5;
					}
				}

			}

			if (Mh != 0) {
				#pragma omp parallel for
				for (int m = 0; m < Mh; m++) {

					for (int nr = 0; nr < Nr; nr++) {

						float* pA = A + ((Mb * MB + m) * K);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + ((Mb * MB + m) * N + nr * NR);

						__asm {

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, KB
							kloops :
							CMP ecx, 0
								JE kloope

								VBROADCASTSS ymm2, [rsi]

								VMOVUPS ymm3, [rdi]
								VMOVUPS ymm4, [rdi + 32]

								VFMADD231PS ymm0, ymm2, ymm3
								VFMADD231PS ymm1, ymm2, ymm4

								ADD rsi, 4
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm0

						}

					}
					for (int n = 0; n < Nh; n++) {
						float s = 0;
						for (int k = 0; k < KB; k++) {
							s += B[(kb * KB + k) * N + Nr * NR + n] * A[(Mb * MB + m) * K + (kb * KB + k)];
						}
						C[(Mb * MB + m) * N + Nr * NR + n] += s;
					}
				}
			}
		}
		if (Kh != 0) {
			#pragma omp parallel for
			for (int k = 0; k < Kh; k++) {
				for (int nr = 0; nr < Nr; nr++) {
					for (int n = 0; n < NR; n++) {
						Bp[nr * KB * NR + k * NR + n] = B[(Kb * KB + k) * N + nr * NR + n];
					}
				}
			}
			#pragma omp parallel for
			for (int mb = 0; mb < Mb; mb++) {
				float* Ap = tempmem + Nr * KB * NR + mb * Mr * KB * MR;// (float*)_aligned_malloc(sizeof(float) * Mr * Kh * MR, 32);

				for (int mr = 0; mr < Mr; mr++) {
					for (int m = 0; m < MR; m++) {
						for (int k = 0; k < Kh; k++) {
							Ap[mr * Kh * MR + k * MR + m] = A[(mb * MB + mr * MR + m) * K + (Kb * KB + k)];
						}
					}
				}

				for (int nr = 0; nr < Nr; nr++) {
					for (int mr = 0; mr < Mr; mr++) {

						float* pA = Ap + (mr * Kh * MR);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + (mb * MB + mr * MR) * N + (nr * NR);

						// 6x16 Asm
						__asm {

							MOV rax, 0
							MOV eax, N
							MOV ecx, 4
							MUL ecx

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm2, [rsi]
							VMOVUPS ymm3, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm4, [rsi]
							VMOVUPS ymm5, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm6, [rsi]
							VMOVUPS ymm7, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm8, [rsi]
							VMOVUPS ymm9, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm10, [rsi]
							VMOVUPS ymm11, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, Kh
							kloops :
							CMP ecx, 0
								JE kloope

								VMOVUPS ymm12, [rdi]
								VMOVUPS ymm13, [rdi + 32]

								VBROADCASTSS ymm14, [rsi]
								VBROADCASTSS ymm15, [rsi + 4]
								VFMADD231PS ymm0, ymm14, ymm12
								VFMADD231PS ymm1, ymm14, ymm13
								VFMADD231PS ymm2, ymm15, ymm12
								VFMADD231PS ymm3, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 8]
								VBROADCASTSS ymm15, [rsi + 12]
								VFMADD231PS ymm4, ymm14, ymm12
								VFMADD231PS ymm5, ymm14, ymm13
								VFMADD231PS ymm6, ymm15, ymm12
								VFMADD231PS ymm7, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 16]
								VBROADCASTSS ymm15, [rsi + 20]
								VFMADD231PS ymm8, ymm14, ymm12
								VFMADD231PS ymm9, ymm14, ymm13
								VFMADD231PS ymm10, ymm15, ymm12
								VFMADD231PS ymm11, ymm15, ymm13

								ADD rsi, 24
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm1
								ADD rsi, rax
								VMOVUPS[rsi], ymm2
								VMOVUPS[rsi + 32], ymm3
								ADD rsi, rax
								VMOVUPS[rsi], ymm4
								VMOVUPS[rsi + 32], ymm5
								ADD rsi, rax
								VMOVUPS[rsi], ymm6
								VMOVUPS[rsi + 32], ymm7
								ADD rsi, rax
								VMOVUPS[rsi], ymm8
								VMOVUPS[rsi + 32], ymm9
								ADD rsi, rax
								VMOVUPS[rsi], ymm10
								VMOVUPS[rsi + 32], ymm11
						}

					}
				}
				for (int n = 0; n < Nh; n++) {
					for (int mr = 0; mr < Mr; mr++) {
						float s0, s1, s2, s3, s4, s5;
						s0 = s1 = s2 = s3 = s4 = s5 = 0;
						for (int k = 0; k < Kh; k++) {
							float t = B[(Kb * KB + k) * N + Nr * NR + n];
							s0 += t * Ap[mr * Kh * MR + k * MR + 0];
							s1 += t * Ap[mr * Kh * MR + k * MR + 1];
							s2 += t * Ap[mr * Kh * MR + k * MR + 2];
							s3 += t * Ap[mr * Kh * MR + k * MR + 3];
							s4 += t * Ap[mr * Kh * MR + k * MR + 4];
							s5 += t * Ap[mr * Kh * MR + k * MR + 5];
						}
						C[(mb * MB + mr * MR + 0) * N + Nr * NR + n] += s0;
						C[(mb * MB + mr * MR + 1) * N + Nr * NR + n] += s1;
						C[(mb * MB + mr * MR + 2) * N + Nr * NR + n] += s2;
						C[(mb * MB + mr * MR + 3) * N + Nr * NR + n] += s3;
						C[(mb * MB + mr * MR + 4) * N + Nr * NR + n] += s4;
						C[(mb * MB + mr * MR + 5) * N + Nr * NR + n] += s5;
					}
				}

			}

			if (Mh != 0) {
				#pragma omp parallel for
				for (int m = 0; m < Mh; m++) {

					for (int nr = 0; nr < Nr; nr++) {

						float* pA = A + ((Mb * MB + m) * K);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + ((Mb * MB + m) * N + nr * NR);

						__asm {

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, Kh
							kloops :
							CMP ecx, 0
								JE kloope

								VBROADCASTSS ymm2, [rsi]

								VMOVUPS ymm3, [rdi]
								VMOVUPS ymm4, [rdi + 32]

								VFMADD231PS ymm0, ymm2, ymm3
								VFMADD231PS ymm1, ymm2, ymm4

								ADD rsi, 4
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm0

						}

					}
					for (int n = 0; n < Nh; n++) {
						float s = 0;
						for (int k = 0; k < Kh; k++) {
							s += B[(Kb * KB + k) * N + Nr * NR + n] * A[(Mb * MB + m) * K + (Kb * KB + k)];
						}
						C[(Mb * MB + m) * N + Nr * NR + n] += s;
					}
				}
			}
		}
	}
	else if (trans_a > 0 && trans_b > 0) {
		float* Bp = tempmem;// (float*)_aligned_malloc(sizeof(float) * Nr * KB* NR, 32);
		for (int kb = 0; kb < Kb; kb++) {

			#pragma omp parallel for
			for (int k = 0; k < KB; k++) {
				for (int nr = 0; nr < Nr; nr++) {
					for (int n = 0; n < NR; n++) {
						Bp[nr * KB * NR + k * NR + n] = B[(nr * NR + n) * K  + (kb * KB + k)];
					}
				}
			}

			#pragma omp parallel for
			for (int mb = 0; mb < Mb; mb++) {
				float* Ap = tempmem + Nr * KB * NR + mb * Mr * KB * MR; //(float*)_aligned_malloc(sizeof(float) * Mr * KB * MR, 32);

				for (int mr = 0; mr < Mr; mr++) {
					for (int m = 0; m < MR; m++) {
						for (int k = 0; k < KB; k++) {
							Ap[mr * KB * MR + k * MR + m] = A[(kb * KB + k) * M + (mb * MB + mr * MR + m)];
						}
					}
				}

				for (int nr = 0; nr < Nr; nr++) {
					for (int mr = 0; mr < Mr; mr++) {

						float* pA = Ap + (mr * KB * MR);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + (mb * MB + mr * MR) * N + (nr * NR);

						// 6x16 Asm
						__asm {

							MOV rax, 0
							MOV eax, N
							MOV ecx, 4
							MUL ecx

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm2, [rsi]
							VMOVUPS ymm3, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm4, [rsi]
							VMOVUPS ymm5, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm6, [rsi]
							VMOVUPS ymm7, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm8, [rsi]
							VMOVUPS ymm9, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm10, [rsi]
							VMOVUPS ymm11, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, KB
							kloops :
							CMP ecx, 0
								JE kloope

								VMOVUPS ymm12, [rdi]
								VMOVUPS ymm13, [rdi + 32]

								VBROADCASTSS ymm14, [rsi]
								VBROADCASTSS ymm15, [rsi + 4]
								VFMADD231PS ymm0, ymm14, ymm12
								VFMADD231PS ymm1, ymm14, ymm13
								VFMADD231PS ymm2, ymm15, ymm12
								VFMADD231PS ymm3, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 8]
								VBROADCASTSS ymm15, [rsi + 12]
								VFMADD231PS ymm4, ymm14, ymm12
								VFMADD231PS ymm5, ymm14, ymm13
								VFMADD231PS ymm6, ymm15, ymm12
								VFMADD231PS ymm7, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 16]
								VBROADCASTSS ymm15, [rsi + 20]
								VFMADD231PS ymm8, ymm14, ymm12
								VFMADD231PS ymm9, ymm14, ymm13
								VFMADD231PS ymm10, ymm15, ymm12
								VFMADD231PS ymm11, ymm15, ymm13

								ADD rsi, 24
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm1
								ADD rsi, rax
								VMOVUPS[rsi], ymm2
								VMOVUPS[rsi + 32], ymm3
								ADD rsi, rax
								VMOVUPS[rsi], ymm4
								VMOVUPS[rsi + 32], ymm5
								ADD rsi, rax
								VMOVUPS[rsi], ymm6
								VMOVUPS[rsi + 32], ymm7
								ADD rsi, rax
								VMOVUPS[rsi], ymm8
								VMOVUPS[rsi + 32], ymm9
								ADD rsi, rax
								VMOVUPS[rsi], ymm10
								VMOVUPS[rsi + 32], ymm11
						}

					}
				}
				for (int n = 0; n < Nh; n++) {
					for (int mr = 0; mr < Mr; mr++) {
						float s0, s1, s2, s3, s4, s5;
						s0 = s1 = s2 = s3 = s4 = s5 = 0;
						for (int k = 0; k < KB; k++) {
							float t = B[(Nr * NR + n) * K + (kb * KB + k)];
							s0 += t * Ap[mr * KB * MR + k * MR + 0];
							s1 += t * Ap[mr * KB * MR + k * MR + 1];
							s2 += t * Ap[mr * KB * MR + k * MR + 2];
							s3 += t * Ap[mr * KB * MR + k * MR + 3];
							s4 += t * Ap[mr * KB * MR + k * MR + 4];
							s5 += t * Ap[mr * KB * MR + k * MR + 5];
						}
						C[(mb * MB + mr * MR + 0) * N + Nr * NR + n] += s0;
						C[(mb * MB + mr * MR + 1) * N + Nr * NR + n] += s1;
						C[(mb * MB + mr * MR + 2) * N + Nr * NR + n] += s2;
						C[(mb * MB + mr * MR + 3) * N + Nr * NR + n] += s3;
						C[(mb * MB + mr * MR + 4) * N + Nr * NR + n] += s4;
						C[(mb * MB + mr * MR + 5) * N + Nr * NR + n] += s5;
					}
				}

			}

			if (Mh != 0) {
				#pragma omp parallel for
				for (int m = 0; m < Mh; m++) {

					for (int nr = 0; nr < Nr; nr++) {

						float* pA = A + (Mb * MB + m);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + ((Mb * MB + m) * N + nr * NR);

						__asm {

							MOV rax, 0
							MOV eax, M
							MOV ecx, 4
							MUL ecx
							MOV r12, rax

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, KB
							kloops :
							CMP ecx, 0
								JE kloope

								VBROADCASTSS ymm2, [rsi]

								VMOVUPS ymm3, [rdi]
								VMOVUPS ymm4, [rdi + 32]

								VFMADD231PS ymm0, ymm2, ymm3
								VFMADD231PS ymm1, ymm2, ymm4

								ADD rsi, r12
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm0

						}

					}
					for (int n = 0; n < Nh; n++) {
						float s = 0;
						for (int k = 0; k < KB; k++) {
							s += B[(Nr * NR + n) * K + (kb * KB + k)] * A[(kb * KB + k) * M + (Mb * MB + m)];
						}
						C[(Mb * MB + m) * N + Nr * NR + n] += s;
					}
				}
			}
		}
		if (Kh != 0) {
			#pragma omp parallel for
			for (int k = 0; k < Kh; k++) {
				for (int nr = 0; nr < Nr; nr++) {
					for (int n = 0; n < NR; n++) {
						Bp[nr * KB * NR + k * NR + n] = B[(nr * NR + n) * K + (Kb * KB + k)];
					}
				}
			}
			#pragma omp parallel for
			for (int mb = 0; mb < Mb; mb++) {
				float* Ap = tempmem + Nr * KB * NR + mb * Mr * KB * MR; //(float*)_aligned_malloc(sizeof(float) * Mr * Kh * MR, 32);

				for (int mr = 0; mr < Mr; mr++) {
					for (int m = 0; m < MR; m++) {
						for (int k = 0; k < Kh; k++) {
							Ap[mr * Kh * MR + k * MR + m] = A[(Kb * KB + k) * M + (mb * MB + mr * MR + m)];
						}
					}
				}

				for (int nr = 0; nr < Nr; nr++) {
					for (int mr = 0; mr < Mr; mr++) {

						float* pA = Ap + (mr * Kh * MR);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + (mb * MB + mr * MR) * N + (nr * NR);

						// 6x16 Asm
						__asm {

							MOV rax, 0
							MOV eax, N
							MOV ecx, 4
							MUL ecx

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm2, [rsi]
							VMOVUPS ymm3, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm4, [rsi]
							VMOVUPS ymm5, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm6, [rsi]
							VMOVUPS ymm7, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm8, [rsi]
							VMOVUPS ymm9, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm10, [rsi]
							VMOVUPS ymm11, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, Kh
							kloops :
							CMP ecx, 0
								JE kloope

								VMOVUPS ymm12, [rdi]
								VMOVUPS ymm13, [rdi + 32]

								VBROADCASTSS ymm14, [rsi]
								VBROADCASTSS ymm15, [rsi + 4]
								VFMADD231PS ymm0, ymm14, ymm12
								VFMADD231PS ymm1, ymm14, ymm13
								VFMADD231PS ymm2, ymm15, ymm12
								VFMADD231PS ymm3, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 8]
								VBROADCASTSS ymm15, [rsi + 12]
								VFMADD231PS ymm4, ymm14, ymm12
								VFMADD231PS ymm5, ymm14, ymm13
								VFMADD231PS ymm6, ymm15, ymm12
								VFMADD231PS ymm7, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 16]
								VBROADCASTSS ymm15, [rsi + 20]
								VFMADD231PS ymm8, ymm14, ymm12
								VFMADD231PS ymm9, ymm14, ymm13
								VFMADD231PS ymm10, ymm15, ymm12
								VFMADD231PS ymm11, ymm15, ymm13

								ADD rsi, 24
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm1
								ADD rsi, rax
								VMOVUPS[rsi], ymm2
								VMOVUPS[rsi + 32], ymm3
								ADD rsi, rax
								VMOVUPS[rsi], ymm4
								VMOVUPS[rsi + 32], ymm5
								ADD rsi, rax
								VMOVUPS[rsi], ymm6
								VMOVUPS[rsi + 32], ymm7
								ADD rsi, rax
								VMOVUPS[rsi], ymm8
								VMOVUPS[rsi + 32], ymm9
								ADD rsi, rax
								VMOVUPS[rsi], ymm10
								VMOVUPS[rsi + 32], ymm11
						}

					}
				}
				for (int n = 0; n < Nh; n++) {
					for (int mr = 0; mr < Mr; mr++) {
						float s0, s1, s2, s3, s4, s5;
						s0 = s1 = s2 = s3 = s4 = s5 = 0;
						for (int k = 0; k < Kh; k++) {
							float t = B[(Nr * NR + n) * K + (Kb * KB + k)];
							s0 += t * Ap[mr * Kh * MR + k * MR + 0];
							s1 += t * Ap[mr * Kh * MR + k * MR + 1];
							s2 += t * Ap[mr * Kh * MR + k * MR + 2];
							s3 += t * Ap[mr * Kh * MR + k * MR + 3];
							s4 += t * Ap[mr * Kh * MR + k * MR + 4];
							s5 += t * Ap[mr * Kh * MR + k * MR + 5];
						}
						C[(mb * MB + mr * MR + 0) * N + Nr * NR + n] += s0;
						C[(mb * MB + mr * MR + 1) * N + Nr * NR + n] += s1;
						C[(mb * MB + mr * MR + 2) * N + Nr * NR + n] += s2;
						C[(mb * MB + mr * MR + 3) * N + Nr * NR + n] += s3;
						C[(mb * MB + mr * MR + 4) * N + Nr * NR + n] += s4;
						C[(mb * MB + mr * MR + 5) * N + Nr * NR + n] += s5;
					}
				}

			}
			if (Mh != 0) {
				#pragma omp parallel for
				for (int m = 0; m < Mh; m++) {

					for (int nr = 0; nr < Nr; nr++) {

						float* pA = A + (Mb * MB + m);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + ((Mb * MB + m) * N + nr * NR);

						__asm {

							MOV rax, 0
							MOV eax, M
							MOV ecx, 4
							MUL ecx
							MOV r12, rax

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, Kh
							kloops :
							CMP ecx, 0
								JE kloope

								VBROADCASTSS ymm2, [rsi]

								VMOVUPS ymm3, [rdi]
								VMOVUPS ymm4, [rdi + 32]

								VFMADD231PS ymm0, ymm2, ymm3
								VFMADD231PS ymm1, ymm2, ymm4

								ADD rsi, r12
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm0

						}

					}
					for (int n = 0; n < Nh; n++) {
						float s = 0;
						for (int k = 0; k < Kh; k++) {
							s += B[(Kb * KB + k) * N + Nr * NR + n] * A[(Kb * KB + k) * M + (Mb * MB + m)];
						}
						C[(Mb * MB + m) * N + Nr * NR + n] += s;
					}
				}
			}
		}
	}
	else if (trans_a > 0) {
		float* Bp = tempmem;// (float*)_aligned_malloc(sizeof(float)* Nr* KB* NR, 32);
		for (int kb = 0; kb < Kb; kb++) {

			#pragma omp parallel for
			for (int k = 0; k < KB; k++) {
				for (int nr = 0; nr < Nr; nr++) {
					for (int n = 0; n < NR; n++) {
						Bp[nr * KB * NR + k * NR + n] = B[(kb * KB + k) * N + nr * NR + n];
					}
				}
			}

			#pragma omp parallel for
			for (int mb = 0; mb < Mb; mb++) {
				float* Ap = tempmem + Nr * KB * NR + mb * Mr * KB * MR;//(float*)_aligned_malloc(sizeof(float) * Mr * KB * MR, 32);

				for (int mr = 0; mr < Mr; mr++) {
					for (int m = 0; m < MR; m++) {
						for (int k = 0; k < KB; k++) {
							Ap[mr * KB * MR + k * MR + m] = A[(kb * KB + k) * M + (mb * MB + mr * MR + m)];
						}
					}
				}

				for (int nr = 0; nr < Nr; nr++) {
					for (int mr = 0; mr < Mr; mr++) {

						float* pA = Ap + (mr * KB * MR);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + (mb * MB + mr * MR) * N + (nr * NR);

						// 6x16 Asm
						__asm {

							MOV rax, 0
							MOV eax, N
							MOV ecx, 4
							MUL ecx

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm2, [rsi]
							VMOVUPS ymm3, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm4, [rsi]
							VMOVUPS ymm5, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm6, [rsi]
							VMOVUPS ymm7, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm8, [rsi]
							VMOVUPS ymm9, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm10, [rsi]
							VMOVUPS ymm11, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, KB
							kloops :
							CMP ecx, 0
								JE kloope

								VMOVUPS ymm12, [rdi]
								VMOVUPS ymm13, [rdi + 32]

								VBROADCASTSS ymm14, [rsi]
								VBROADCASTSS ymm15, [rsi + 4]
								VFMADD231PS ymm0, ymm14, ymm12
								VFMADD231PS ymm1, ymm14, ymm13
								VFMADD231PS ymm2, ymm15, ymm12
								VFMADD231PS ymm3, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 8]
								VBROADCASTSS ymm15, [rsi + 12]
								VFMADD231PS ymm4, ymm14, ymm12
								VFMADD231PS ymm5, ymm14, ymm13
								VFMADD231PS ymm6, ymm15, ymm12
								VFMADD231PS ymm7, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 16]
								VBROADCASTSS ymm15, [rsi + 20]
								VFMADD231PS ymm8, ymm14, ymm12
								VFMADD231PS ymm9, ymm14, ymm13
								VFMADD231PS ymm10, ymm15, ymm12
								VFMADD231PS ymm11, ymm15, ymm13

								ADD rsi, 24
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm1
								ADD rsi, rax
								VMOVUPS[rsi], ymm2
								VMOVUPS[rsi + 32], ymm3
								ADD rsi, rax
								VMOVUPS[rsi], ymm4
								VMOVUPS[rsi + 32], ymm5
								ADD rsi, rax
								VMOVUPS[rsi], ymm6
								VMOVUPS[rsi + 32], ymm7
								ADD rsi, rax
								VMOVUPS[rsi], ymm8
								VMOVUPS[rsi + 32], ymm9
								ADD rsi, rax
								VMOVUPS[rsi], ymm10
								VMOVUPS[rsi + 32], ymm11
						}

					}
				}
				for (int n = 0; n < Nh; n++) {
					for (int mr = 0; mr < Mr; mr++) {
						float s0, s1, s2, s3, s4, s5;
						s0 = s1 = s2 = s3 = s4 = s5 = 0;
						for (int k = 0; k < KB; k++) {
							float t = B[(kb * KB + k) * N + Nr * NR + n];
							s0 += t * Ap[mr * KB * MR + k * MR + 0];
							s1 += t * Ap[mr * KB * MR + k * MR + 1];
							s2 += t * Ap[mr * KB * MR + k * MR + 2];
							s3 += t * Ap[mr * KB * MR + k * MR + 3];
							s4 += t * Ap[mr * KB * MR + k * MR + 4];
							s5 += t * Ap[mr * KB * MR + k * MR + 5];
						}
						C[(mb * MB + mr * MR + 0) * N + Nr * NR + n] += s0;
						C[(mb * MB + mr * MR + 1) * N + Nr * NR + n] += s1;
						C[(mb * MB + mr * MR + 2) * N + Nr * NR + n] += s2;
						C[(mb * MB + mr * MR + 3) * N + Nr * NR + n] += s3;
						C[(mb * MB + mr * MR + 4) * N + Nr * NR + n] += s4;
						C[(mb * MB + mr * MR + 5) * N + Nr * NR + n] += s5;
					}
				}

			}

			if (Mh != 0) {
				#pragma omp parallel for
				for (int m = 0; m < Mh; m++) {
					for (int nr = 0; nr < Nr; nr++) {

						float* pA = A + (Mb * MB + m);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + ((Mb * MB + m) * N + nr * NR);

						__asm {

							MOV rax, 0
							MOV eax, M
							MOV ecx, 4
							MUL ecx
							MOV r12, rax

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, KB
							kloops :
							CMP ecx, 0
								JE kloope

								VBROADCASTSS ymm2, [rsi]

								VMOVUPS ymm3, [rdi]
								VMOVUPS ymm4, [rdi + 32]

								VFMADD231PS ymm0, ymm2, ymm3
								VFMADD231PS ymm1, ymm2, ymm4

								ADD rsi, r12
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm0

						}

					}
					for (int n = 0; n < Nh; n++) {
						float s = 0;
						for (int k = 0; k < KB; k++) {
							s += B[(kb * KB + k) * N + Nr * NR + n] * A[(kb * KB + k) * M + (Mb * MB + m)];
						}
						C[(Mb * MB + m) * N + Nr * NR + n] += s;
					}
				}
			}
		}
		if (Kh != 0) {
			#pragma omp parallel for
			for (int k = 0; k < Kh; k++) {
				for (int nr = 0; nr < Nr; nr++) {
					for (int n = 0; n < NR; n++) {
						Bp[nr * KB * NR + k * NR + n] = B[(Kb * KB + k) * N + nr * NR + n];
					}
				}
			}
			#pragma omp parallel for
			for (int mb = 0; mb < Mb; mb++) {
				float* Ap = tempmem + Nr * KB * NR + mb * Mr * KB * MR;// (float*)_aligned_malloc(sizeof(float)* Mr* Kh* MR, 32);

				for (int mr = 0; mr < Mr; mr++) {
					for (int m = 0; m < MR; m++) {
						for (int k = 0; k < Kh; k++) {
							Ap[mr * Kh * MR + k * MR + m] = A[(Kb * KB + k) * M + (mb * MB + mr * MR + m)];
						}
					}
				}

				for (int nr = 0; nr < Nr; nr++) {
					for (int mr = 0; mr < Mr; mr++) {

						float* pA = Ap + (mr * Kh * MR);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + (mb * MB + mr * MR) * N + (nr * NR);

						// 6x16 Asm
						__asm {

							MOV rax, 0
							MOV eax, N
							MOV ecx, 4
							MUL ecx

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm2, [rsi]
							VMOVUPS ymm3, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm4, [rsi]
							VMOVUPS ymm5, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm6, [rsi]
							VMOVUPS ymm7, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm8, [rsi]
							VMOVUPS ymm9, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm10, [rsi]
							VMOVUPS ymm11, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, Kh
							kloops :
							CMP ecx, 0
								JE kloope

								VMOVUPS ymm12, [rdi]
								VMOVUPS ymm13, [rdi + 32]

								VBROADCASTSS ymm14, [rsi]
								VBROADCASTSS ymm15, [rsi + 4]
								VFMADD231PS ymm0, ymm14, ymm12
								VFMADD231PS ymm1, ymm14, ymm13
								VFMADD231PS ymm2, ymm15, ymm12
								VFMADD231PS ymm3, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 8]
								VBROADCASTSS ymm15, [rsi + 12]
								VFMADD231PS ymm4, ymm14, ymm12
								VFMADD231PS ymm5, ymm14, ymm13
								VFMADD231PS ymm6, ymm15, ymm12
								VFMADD231PS ymm7, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 16]
								VBROADCASTSS ymm15, [rsi + 20]
								VFMADD231PS ymm8, ymm14, ymm12
								VFMADD231PS ymm9, ymm14, ymm13
								VFMADD231PS ymm10, ymm15, ymm12
								VFMADD231PS ymm11, ymm15, ymm13

								ADD rsi, 24
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm1
								ADD rsi, rax
								VMOVUPS[rsi], ymm2
								VMOVUPS[rsi + 32], ymm3
								ADD rsi, rax
								VMOVUPS[rsi], ymm4
								VMOVUPS[rsi + 32], ymm5
								ADD rsi, rax
								VMOVUPS[rsi], ymm6
								VMOVUPS[rsi + 32], ymm7
								ADD rsi, rax
								VMOVUPS[rsi], ymm8
								VMOVUPS[rsi + 32], ymm9
								ADD rsi, rax
								VMOVUPS[rsi], ymm10
								VMOVUPS[rsi + 32], ymm11
						}

					}
				}
				for (int n = 0; n < Nh; n++) {
					for (int mr = 0; mr < Mr; mr++) {
						float s0, s1, s2, s3, s4, s5;
						s0 = s1 = s2 = s3 = s4 = s5 = 0;
						for (int k = 0; k < Kh; k++) {
							float t = B[(Kb * KB + k) * N + Nr * NR + n];
							s0 += t * Ap[mr * Kh * MR + k * MR + 0];
							s1 += t * Ap[mr * Kh * MR + k * MR + 1];
							s2 += t * Ap[mr * Kh * MR + k * MR + 2];
							s3 += t * Ap[mr * Kh * MR + k * MR + 3];
							s4 += t * Ap[mr * Kh * MR + k * MR + 4];
							s5 += t * Ap[mr * Kh * MR + k * MR + 5];
						}
						C[(mb * MB + mr * MR + 0) * N + Nr * NR + n] += s0;
						C[(mb * MB + mr * MR + 1) * N + Nr * NR + n] += s1;
						C[(mb * MB + mr * MR + 2) * N + Nr * NR + n] += s2;
						C[(mb * MB + mr * MR + 3) * N + Nr * NR + n] += s3;
						C[(mb * MB + mr * MR + 4) * N + Nr * NR + n] += s4;
						C[(mb * MB + mr * MR + 5) * N + Nr * NR + n] += s5;
					}
				}

			}
			if (Mh != 0) {
				#pragma omp parallel for
				for (int m = 0; m < Mh; m++) {

					for (int nr = 0; nr < Nr; nr++) {

						float* pA = A + (Mb * MB + m);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + ((Mb * MB + m) * N + nr * NR);

						__asm {

							MOV rax, 0
							MOV eax, M
							MOV ecx, 4
							MUL ecx
							MOV r12, rax

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, Kh
							kloops :
							CMP ecx, 0
								JE kloope

								VBROADCASTSS ymm2, [rsi]

								VMOVUPS ymm3, [rdi]
								VMOVUPS ymm4, [rdi + 32]

								VFMADD231PS ymm0, ymm2, ymm3
								VFMADD231PS ymm1, ymm2, ymm4

								ADD rsi, r12
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm0

						}

					}
					for (int n = 0; n < Nh; n++) {
						float s = 0;
						for (int k = 0; k < Kh; k++) {
							s += B[(Kb * KB + k) * N + Nr * NR + n] * A[(Kb * KB + k) * M + (Mb * MB + m)];
						}
						C[(Mb * MB + m) * N + Nr * NR + n] += s;
					}
				}
			}
		}
	}
	else {
		float* Bp = tempmem;// (float*)_aligned_malloc(sizeof(float)* Nr* KB* NR, 32);
		for (int kb = 0; kb < Kb; kb++) {

			#pragma omp parallel for
			for (int k = 0; k < KB; k++) {
				for (int nr = 0; nr < Nr; nr++) {
					for (int n = 0; n < NR; n++) {
						Bp[nr * KB * NR + k * NR + n] = B[(nr * NR + n) * K + (kb * KB + k)];
					}
				}
			}

			#pragma omp parallel for
			for (int mb = 0; mb < Mb; mb++) {
				float* Ap = tempmem + Nr * KB * NR + mb * Mr * KB * MR; //(float*)_aligned_malloc(sizeof(float) * Mr * KB * MR, 32);

				for (int mr = 0; mr < Mr; mr++) {
					for (int m = 0; m < MR; m++) {
						for (int k = 0; k < KB; k++) {
							Ap[mr * KB * MR + k * MR + m] = A[(mb * MB + mr * MR + m) * K + (kb * KB + k)];
						}
					}
				}

				for (int nr = 0; nr < Nr; nr++) {
					for (int mr = 0; mr < Mr; mr++) {

						float* pA = Ap + (mr * KB * MR);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + (mb * MB + mr * MR) * N + (nr * NR);

						// 6x16 Asm
						__asm {

							MOV rax, 0
							MOV eax, N
							MOV ecx, 4
							MUL ecx

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm2, [rsi]
							VMOVUPS ymm3, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm4, [rsi]
							VMOVUPS ymm5, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm6, [rsi]
							VMOVUPS ymm7, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm8, [rsi]
							VMOVUPS ymm9, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm10, [rsi]
							VMOVUPS ymm11, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, KB
							kloops :
							CMP ecx, 0
								JE kloope

								VMOVUPS ymm12, [rdi]
								VMOVUPS ymm13, [rdi + 32]

								VBROADCASTSS ymm14, [rsi]
								VBROADCASTSS ymm15, [rsi + 4]
								VFMADD231PS ymm0, ymm14, ymm12
								VFMADD231PS ymm1, ymm14, ymm13
								VFMADD231PS ymm2, ymm15, ymm12
								VFMADD231PS ymm3, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 8]
								VBROADCASTSS ymm15, [rsi + 12]
								VFMADD231PS ymm4, ymm14, ymm12
								VFMADD231PS ymm5, ymm14, ymm13
								VFMADD231PS ymm6, ymm15, ymm12
								VFMADD231PS ymm7, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 16]
								VBROADCASTSS ymm15, [rsi + 20]
								VFMADD231PS ymm8, ymm14, ymm12
								VFMADD231PS ymm9, ymm14, ymm13
								VFMADD231PS ymm10, ymm15, ymm12
								VFMADD231PS ymm11, ymm15, ymm13

								ADD rsi, 24
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm1
								ADD rsi, rax
								VMOVUPS[rsi], ymm2
								VMOVUPS[rsi + 32], ymm3
								ADD rsi, rax
								VMOVUPS[rsi], ymm4
								VMOVUPS[rsi + 32], ymm5
								ADD rsi, rax
								VMOVUPS[rsi], ymm6
								VMOVUPS[rsi + 32], ymm7
								ADD rsi, rax
								VMOVUPS[rsi], ymm8
								VMOVUPS[rsi + 32], ymm9
								ADD rsi, rax
								VMOVUPS[rsi], ymm10
								VMOVUPS[rsi + 32], ymm11
						}

					}
				}
				for (int n = 0; n < Nh; n++) {
					for (int mr = 0; mr < Mr; mr++) {
						float s0, s1, s2, s3, s4, s5;
						s0 = s1 = s2 = s3 = s4 = s5 = 0;
						for (int k = 0; k < KB; k++) {
							float t = B[(Nr * NR + n) * K + (kb * KB + k)];
							s0 += t * Ap[mr * KB * MR + k * MR + 0];
							s1 += t * Ap[mr * KB * MR + k * MR + 1];
							s2 += t * Ap[mr * KB * MR + k * MR + 2];
							s3 += t * Ap[mr * KB * MR + k * MR + 3];
							s4 += t * Ap[mr * KB * MR + k * MR + 4];
							s5 += t * Ap[mr * KB * MR + k * MR + 5];
						}
						C[(mb * MB + mr * MR + 0) * N + Nr * NR + n] += s0;
						C[(mb * MB + mr * MR + 1) * N + Nr * NR + n] += s1;
						C[(mb * MB + mr * MR + 2) * N + Nr * NR + n] += s2;
						C[(mb * MB + mr * MR + 3) * N + Nr * NR + n] += s3;
						C[(mb * MB + mr * MR + 4) * N + Nr * NR + n] += s4;
						C[(mb * MB + mr * MR + 5) * N + Nr * NR + n] += s5;
					}
				}

			}
			if (Mh != 0) {
				#pragma omp parallel for
				for (int m = 0; m < Mh; m++) {

					for (int nr = 0; nr < Nr; nr++) {

						float* pA = A + ((Mb * MB + m) * K);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + ((Mb * MB + m) * N + nr * NR);

						__asm {

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, KB
							kloops :
							CMP ecx, 0
								JE kloope

								VBROADCASTSS ymm2, [rsi]

								VMOVUPS ymm3, [rdi]
								VMOVUPS ymm4, [rdi + 32]

								VFMADD231PS ymm0, ymm2, ymm3
								VFMADD231PS ymm1, ymm2, ymm4

								ADD rsi, 4
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm0

						}

					}
					for (int n = 0; n < Nh; n++) {
						float s = 0;
						for (int k = 0; k < KB; k++) {
							s += B[(Nr * NR + n) * K + (kb * KB + k)] * A[(Mb * MB + m) * K + (kb * KB + k)];
						}
						C[(Mb * MB + m) * N + Nr * NR + n] += s;
					}
				}
			}
		}
		if (Kh != 0) {
			#pragma omp parallel for
			for (int k = 0; k < Kh; k++) {
				for (int nr = 0; nr < Nr; nr++) {
					for (int n = 0; n < NR; n++) {
						Bp[nr * KB * NR + k * NR + n] = B[(nr * NR + n) * K + (Kb * KB + k)];
					}
				}
			}
			#pragma omp parallel for
			for (int mb = 0; mb < Mb; mb++) {
				float* Ap = tempmem + Nr * KB * NR + mb * Mr * KB * MR; //(float*)_aligned_malloc(sizeof(float) * Mr * Kh * MR, 32);

				for (int mr = 0; mr < Mr; mr++) {
					for (int m = 0; m < MR; m++) {
						for (int k = 0; k < Kh; k++) {
							Ap[mr * Kh * MR + k * MR + m] = A[(mb * MB + mr * MR + m) * K + (Kb * KB + k)];
						}
					}
				}

				for (int nr = 0; nr < Nr; nr++) {
					for (int mr = 0; mr < Mr; mr++) {

						float* pA = Ap + (mr * Kh * MR);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + (mb * MB + mr * MR) * N + (nr * NR);

						// 6x16 Asm
						__asm {

							MOV rax, 0
							MOV eax, N
							MOV ecx, 4
							MUL ecx

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm2, [rsi]
							VMOVUPS ymm3, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm4, [rsi]
							VMOVUPS ymm5, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm6, [rsi]
							VMOVUPS ymm7, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm8, [rsi]
							VMOVUPS ymm9, [rsi + 32]
							ADD rsi, rax
							VMOVUPS ymm10, [rsi]
							VMOVUPS ymm11, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, Kh
							kloops :
							CMP ecx, 0
								JE kloope

								VMOVUPS ymm12, [rdi]
								VMOVUPS ymm13, [rdi + 32]

								VBROADCASTSS ymm14, [rsi]
								VBROADCASTSS ymm15, [rsi + 4]
								VFMADD231PS ymm0, ymm14, ymm12
								VFMADD231PS ymm1, ymm14, ymm13
								VFMADD231PS ymm2, ymm15, ymm12
								VFMADD231PS ymm3, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 8]
								VBROADCASTSS ymm15, [rsi + 12]
								VFMADD231PS ymm4, ymm14, ymm12
								VFMADD231PS ymm5, ymm14, ymm13
								VFMADD231PS ymm6, ymm15, ymm12
								VFMADD231PS ymm7, ymm15, ymm13

								VBROADCASTSS ymm14, [rsi + 16]
								VBROADCASTSS ymm15, [rsi + 20]
								VFMADD231PS ymm8, ymm14, ymm12
								VFMADD231PS ymm9, ymm14, ymm13
								VFMADD231PS ymm10, ymm15, ymm12
								VFMADD231PS ymm11, ymm15, ymm13

								ADD rsi, 24
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm1
								ADD rsi, rax
								VMOVUPS[rsi], ymm2
								VMOVUPS[rsi + 32], ymm3
								ADD rsi, rax
								VMOVUPS[rsi], ymm4
								VMOVUPS[rsi + 32], ymm5
								ADD rsi, rax
								VMOVUPS[rsi], ymm6
								VMOVUPS[rsi + 32], ymm7
								ADD rsi, rax
								VMOVUPS[rsi], ymm8
								VMOVUPS[rsi + 32], ymm9
								ADD rsi, rax
								VMOVUPS[rsi], ymm10
								VMOVUPS[rsi + 32], ymm11
						}

					}
				}
				for (int n = 0; n < Nh; n++) {
					for (int mr = 0; mr < Mr; mr++) {
						float s0, s1, s2, s3, s4, s5;
						s0 = s1 = s2 = s3 = s4 = s5 = 0;
						for (int k = 0; k < Kh; k++) {
							float t = B[(Nr * NR + n) * K + (Kb * KB + k)];
							s0 += t * Ap[mr * Kh * MR + k * MR + 0];
							s1 += t * Ap[mr * Kh * MR + k * MR + 1];
							s2 += t * Ap[mr * Kh * MR + k * MR + 2];
							s3 += t * Ap[mr * Kh * MR + k * MR + 3];
							s4 += t * Ap[mr * Kh * MR + k * MR + 4];
							s5 += t * Ap[mr * Kh * MR + k * MR + 5];
						}
						C[(mb * MB + mr * MR + 0) * N + Nr * NR + n] += s0;
						C[(mb * MB + mr * MR + 1) * N + Nr * NR + n] += s1;
						C[(mb * MB + mr * MR + 2) * N + Nr * NR + n] += s2;
						C[(mb * MB + mr * MR + 3) * N + Nr * NR + n] += s3;
						C[(mb * MB + mr * MR + 4) * N + Nr * NR + n] += s4;
						C[(mb * MB + mr * MR + 5) * N + Nr * NR + n] += s5;
					}
				}

			}
			if (Mh != 0) {
				#pragma omp parallel for
				for (int m = 0; m < Mh; m++) {

					for (int nr = 0; nr < Nr; nr++) {

						float* pA = A + ((Mb * MB + m) * K);
						float* pB = Bp + (nr * KB * NR);
						float* pC = C + ((Mb * MB + m) * N + nr * NR);

						__asm {

							MOV rsi, pC
							VMOVUPS ymm0, [rsi]
							VMOVUPS ymm1, [rsi + 32]

							MOV rsi, pA
							MOV rdi, pB

							MOV ecx, Kh
							kloops :
							CMP ecx, 0
								JE kloope

								VBROADCASTSS ymm2, [rsi]

								VMOVUPS ymm3, [rdi]
								VMOVUPS ymm4, [rdi + 32]

								VFMADD231PS ymm0, ymm2, ymm3
								VFMADD231PS ymm1, ymm2, ymm4

								ADD rsi, 4
								ADD rdi, 64

								DEC ecx
								JMP kloops
								kloope :

							MOV rsi, pC
								VMOVUPS[rsi], ymm0
								VMOVUPS[rsi + 32], ymm0

						}

					}
					for (int n = 0; n < Nh; n++) {
						float s = 0;
						for (int k = 0; k < Kh; k++) {
							s += B[(Nr * NR + n) * K + (Kb * KB + k)] * A[(Mb * MB + m) * K + (Kb * KB + k)];
						}
						C[(Mb * MB + m) * N + Nr * NR + n] += s;
					}
				}
			}
		}		
	}
}



