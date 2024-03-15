#include <omp.h>

#define PREDICT 1
#define UPDATE 0

struct MerlinMachine {
    int number_of_features;
    int number_of_patches;

    omp_lock_t *literal_lock;

    unsigned int *ta_state;
    unsigned int *literal_outputs;
    unsigned int *feedback_to_la;
    unsigned int *literal_patch;
    
    int *output_one_patches;

    unsigned int *literal_weights;

    int number_of_ta_chunks;
    int number_of_state_bits;
    
    int T;

    double s;

    unsigned int filter;
};

struct MerlinMachine *CreateMerlinMachine(int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s);

void mm_initialize(struct MerlinMachine *mm);

int mm_ta_state(struct MerlinMachine *mm, int ta);
