#include <iostream>
#include "engine.h"

int main() {
    int MatrixSize = 2048; // Let's go BIGGER to see the difference!
    std::cout << "Initializing Adaptive Engine (VRAM Allocation)..." << std::endl;
    AdaptiveComputeEngine engine(MatrixSize);

    // Scenario 1: Medical Analysis (Need high precision)
    engine.runInference("CRITICAL");

    // Scenario 2: Chatbot (Need speed)
    engine.runInference("CREATIVE");

    return 0;
}
