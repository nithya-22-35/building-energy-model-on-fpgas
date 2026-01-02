#code
#include "model_params.h"

void anomaly_detector(float energy, float mean, float std, int *anomaly) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=energy
#pragma HLS INTERFACE s_axilite port=mean
#pragma HLS INTERFACE s_axilite port=std
#pragma HLS INTERFACE s_axilite port=anomaly

    float z_score = (energy - mean) / std;

    if (z_score > ANOMALY_THRESHOLD)
        *anomaly = 1;
    else
        *anomaly = 0;
}
