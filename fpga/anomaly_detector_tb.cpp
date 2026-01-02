#code
#include <iostream>
#include "model_params.h"

void anomaly_detector(float, float, float, int*);

int main() {
    int anomaly;

    anomaly_detector(500.0, 200.0, 50.0, &anomaly);

    if (anomaly)
        std::cout << "⚠️ ALERT: Abnormal energy usage detected!" << std::endl;
    else
        std::cout << "Normal usage" << std::endl;

    return 0;
}
