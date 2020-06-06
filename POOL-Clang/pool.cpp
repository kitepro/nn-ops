#include <immintrin.h>
#include <time.h>
#include <omp.h>
#include <iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

#define THREADS 16

DLLEXPORT float* avx2_malloc(unsigned long long size) {
	return (float*)_aligned_malloc(size, 32);
}

DLLEXPORT void maxpool2d(float* img, float* out, int H, int W, int C, int Kx, int Ky, int stride, int batchsize) {

	int NH = (H - Kx) / stride + 1;
	int NW = (W - Ky) / stride + 1;

	int WC = W * C;
	int HWC = H * WC;
	int NWC = NW * C;
	int NHNWC = NH * NWC;

	#pragma omp parallel for
	for (int nh = 0; nh < NH; nh++) {
		for (int b = 0; b < batchsize; b++) {
			for (int nw = 0; nw < NW; nw++) {
				for (int c = 0; c < C; c++) {
					float* pImg = img + b * HWC + nh * stride * WC + nw * stride * C + c;
					float* pOut = out + b * NHNWC + nh * NWC + nw * C + c;

					float m = -9999999999999;
					for (int x = 0; x < Kx; x++) {
						for (int y = 0; y < Ky; y++) {
							float v = pImg[x * WC + y * C];
							m = (m < v) ? v : m;
						}
					}
					pOut[0] = m;
				}
			}
		}
	}
}

DLLEXPORT void maxpool2d_back(float* img, float* grad, float* out, int H, int W, int C, int Kx, int Ky, int stride, int batchsize) {

	int NH = (H - Kx) / stride + 1;
	int NW = (W - Ky) / stride + 1;

	int WC = W * C;
	int HWC = H * WC;
	int NWC = NW * C;
	int NHNWC = NH * NWC;

	memset(out, 0, sizeof(float) * batchsize * HWC);

	#pragma omp parallel for
	for (int c = 0; c < C; c++) {
		for (int nh = 0; nh < NH; nh++) {
			for (int b = 0; b < batchsize; b++) {
				for (int nw = 0; nw < NW; nw++) {
					float* pImg = img + b * HWC + nh * stride * WC + nw * stride * C + c;
					float* pGrad = grad + b * NHNWC + nh * NWC + nw * C + c;
					float* pOut = out + b * HWC + nh * stride * WC + nw * stride * C + c;

					float m = -9999999999999;
					int mi = 0;
					for (int x = 0; x < Kx; x++) {
						for (int y = 0; y < Ky; y++) {
							int p = x * WC + y * C;
							float v = pImg[p];
							if (m < v) {
								m = v;
								mi = p;
							}
						}
					}
					pOut[mi] += pGrad[0];
				}
			}
		}
	}
}



DLLEXPORT void upsample2d(float* img, float* out, int H, int W, int C, int Kx, int Ky, int batchsize) {

	int NH = H * Kx;
	int NW = W * Ky;

	int WC = W * C;
	int HWC = H * WC;
	int NWC = NW * C;
	int NHNWC = NH * NWC;

	#pragma omp parallel for
	for (int c = 0; c < C; c++) {
		for (int b = 0; b < batchsize; b++) {
			for (int h = 0; h < H; h++) {
				for (int w = 0; w < W; w++) {

					float v = img[b * HWC + h * WC + w * C + c];

					for (int x = 0; x < Kx; x++) {
						int nh = h * Kx + x;
						for (int y = 0; y < Ky; y++) {
							int nw = w * Ky + y;
							out[b * NHNWC + nh * NWC + nw * C + c] = v;
						}
					}
				}
			}
		}
	}
}

DLLEXPORT void upsample2d_back(float* out, float* grad, int H, int W, int C, int Kx, int Ky, int batchsize) {

	int NH = H * Kx;
	int NW = W * Ky;

	int WC = W * C;
	int HWC = H * WC;
	int NWC = NW * C;
	int NHNWC = NH * NWC;

	#pragma omp parallel for
	for (int c = 0; c < C; c++) {
		for (int b = 0; b < batchsize; b++) {
			for (int h = 0; h < H; h++) {
				for (int w = 0; w < W; w++) {

					float s = 0;
					for (int x = 0; x < Kx; x++) {
						int nh = h * Kx + x;
						for (int y = 0; y < Ky; y++) {
							int nw = w * Ky + y;
							float v = grad[b * NHNWC + nh * NWC + nw * C + c];
							s += grad[b * NHNWC + nh * NWC + nw * C + c];
						}
					}
					out[b * HWC + h * WC + w * C + c] = s;
				}
			}
		}
	}
}
