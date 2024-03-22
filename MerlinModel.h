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

    unsigned int *literal_weights;
    int *feedback_choice; 

    int number_of_ta_chunks;
    int number_of_state_bits;
    
    int T;

    double s;
    double s_range;

    unsigned int filter;
};

struct MerlinMachine *CreateMerlinMachine(int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, double s_range);

void mm_initialize(struct MerlinMachine *mm);

int mm_ta_state(struct MerlinMachine *mm, int ta);

int mm_ta_action(struct MerlinMachine *mm, int ta);

void mm_update_literals(struct MerlinMachine *mm, unsigned int *Xi, int class_sum, int target);

void mm_update(struct MerlinMachine *mm, unsigned int *Xi, int target);

int mm_score(struct MerlinMachine *mm, unsigned int *Xi);