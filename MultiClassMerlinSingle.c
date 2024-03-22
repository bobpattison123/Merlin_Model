#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "MulticlassMerlin.h"

struct MultiClassMerlinMachine *CreateMultiClassMerlinMachine(int number_of_classes, int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, double s_range) {

    struct MultiClassMerlinMachine *mc_mm = NULL;

    mc_mm = (void *)malloc(sizeof(struct MultiClassMerlinMachine));

    mc_mm->number_of_classes = number_of_classes;
    mc_mm->merlin_machines = (void *)malloc(sizeof(struct MerlinMachine *)* number_of_classes);
    for (int class = 0; class < number_of_classes; class++) {
        mc_mm->merlin_machines[class] = CreateMerlinMachine(number_of_features, number_of_patches, number_of_ta_chunks, number_of_state_bits, T, s, s_range);
    }

    mc_mm->number_of_patches = number_of_patches;

    mc_mm->number_of_ta_chunks = number_of_ta_chunks;

    mc_mm->number_of_state_bits = number_of_state_bits;

    return mc_mm;
}


void mc_mm_initialize(struct MultiClassMerlinMachine *mc_mm) {
    for (int class = 0; class < mc_mm->number_of_classes; class++) {
        mm_initialize(mc_mm->merlin_machines[class]);
    }
}

void mc_mm_update(struct MultiClassMerlinMachine *mc_mm, unsigned int *Xi, int target_class) {
    
    mm_update(mc_mm->merlin_machines[target_class], Xi, 1);

    unsigned int negative_target_class = (unsigned int)mc_mm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
	while (negative_target_class == target_class) {
		negative_target_class = (unsigned int)mc_mm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
	}

    mm_update(mc_mm->merlin_machines[negative_target_class], Xi, 0);
}

void mc_mm_predict(struct MultiClassMerlinMachine *mc_mm, unsigned int *X, int *y, int number_of_examples) {
    unsigned int step_size = mc_mm->number_of_patches * mc_mm->number_of_ta_chunks;

    for (int example = 0; example < number_of_examples; example++) {
        unsigned int pos = example * step_size;
        int max_class_sum = mm_score(mc_mm->merlin_machines[0], &X[pos]);
        int max_class = 0;
        for (int class = 1; class < mc_mm->number_of_classes; class++) {
            int class_sum = mm_score(mc_mm->merlin_machines[class], &X[pos]);
            if (max_class_sum < class_sum) {
                max_class_sum = class_sum;
                max_class = class;
            }
        }
        y[example] = max_class;
    }
}

void mc_mm_fit(struct MultiClassMerlinMachine *mc_mm, unsigned int *X, int *y, int number_of_examples, int epochs) {
    unsigned int step_size = mc_mm->number_of_patches * mc_mm->number_of_ta_chunks;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int example = 0; example < number_of_examples; example++) {
            unsigned int pos = example * step_size;
            mc_mm_update(mc_mm, &X[pos], y[example]);
        }
    }
}

