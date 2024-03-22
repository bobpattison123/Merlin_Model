#include "MerlinModel.h"

struct MultiClassMerlinMachine {
    int number_of_classes;

    struct MerlinMachine **merlin_machines;

    int number_of_patches;
    int number_of_ta_chunks;
    int number_of_state_bits;
};

struct MultiClassMerlinMachine *CreateMultiClassMerlinMachine(int number_of_classes, int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, double s_range);

void mc_mm_initialize(struct MultiClassMerlinMachine *mc_mm);

void mc_mm_predict(struct MultiClassMerlinMachine *mc_mm, unsigned int *X, int *y, int number_of_examples);

void mc_mm_fit(struct MultiClassMerlinMachine *mc_mm, unsigned int *X, int y[], int number_of_examples, int epochs);