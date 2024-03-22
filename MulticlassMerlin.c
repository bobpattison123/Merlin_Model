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

    int max_threads = omp_get_max_threads();
    struct MultiClassMerlinMachine **mc_mm_thread = (void *)malloc(sizeof(struct MultiClassMerlinMachine *) * max_threads);
    struct MerlinMachine *mm = mc_mm->merlin_machines[0];

    for (int thread = 0; thread < max_threads; thread++) {
        mc_mm_thread[thread] = CreateMultiClassMerlinMachine(mc_mm->number_of_classes, mm->number_of_features, mm->number_of_patches, mm->number_of_ta_chunks, mm->number_of_state_bits, mm->T, mm->s, mm->s_range);
        for (int class = 0; class < mc_mm->number_of_classes; class++) {
            free(mc_mm_thread[thread]->merlin_machines[class]->ta_state);
            mc_mm_thread[thread]->merlin_machines[class]->ta_state = mc_mm->merlin_machines[class]->ta_state;
            free(mc_mm_thread[thread]->merlin_machines[class]->literal_weights);
            mc_mm_thread[thread]->merlin_machines[class]->literal_weights = mc_mm->merlin_machines[class]->literal_weights;
        }
    }

    //#pragma omp parallel for
    for (int example = 0; example < number_of_examples; example++) {
        int thread_id = omp_get_thread_num();

        unsigned int pos = example * step_size;
        
        int max_class_sum = mm_score(mc_mm_thread[thread_id]->merlin_machines[0], &X[pos]);
        int max_class = 0;
        for (int class = 1; class < mc_mm_thread[thread_id]->number_of_classes; class++) {
            int class_sum = mm_score(mc_mm_thread[thread_id]->merlin_machines[class], &X[pos]);
            if (max_class_sum < class_sum) {
                max_class_sum = class_sum;
                max_class = class;
            }
        }

        y[example] = max_class;

    }

    for (int thread = 0; thread < max_threads; thread++) {
        for (int class = 0; class < mc_mm_thread[thread]->number_of_classes; class++) {

            struct MerlinMachine *mm_thread = mc_mm_thread[thread]->merlin_machines[class];

            free(mm_thread->literal_outputs);
			free(mm_thread->feedback_to_la);
			free(mm_thread->feedback_choice);
			free(mm_thread);
        }
    }

    free(mc_mm_thread);

    return;

}

void mc_mm_fit(struct MultiClassMerlinMachine *mc_mm, unsigned int *X, int *y, int number_of_examples, int epochs) {

    unsigned int step_size = mc_mm->number_of_patches * mc_mm->number_of_ta_chunks;

    int max_threads = omp_get_max_threads();
    struct MultiClassMerlinMachine **mc_mm_thread = (void *)malloc(sizeof(struct MultiClassMerlinMachine *) * max_threads);
    struct MerlinMachine *mm = mc_mm->merlin_machines[0];
    
    for (int class = 0; class < mc_mm->number_of_classes; class++) {
        mc_mm->merlin_machines[class]->literal_lock = (omp_lock_t *)malloc(sizeof(omp_lock_t) * mm->number_of_features);
        for (int literal = 0; literal < mm->number_of_features; ++literal) {
            omp_init_lock(&mc_mm->merlin_machines[class]->literal_lock[literal]);
        }
    }

    for (int thread = 0; thread < max_threads; thread++) {
        mc_mm_thread[thread] = CreateMultiClassMerlinMachine(mc_mm->number_of_classes, mm->number_of_features, mm->number_of_patches, mm->number_of_ta_chunks, mm->number_of_state_bits, mm->T, mm->s, mm->s_range);
        for (int class = 0; class < mc_mm->number_of_classes; class++) {
            free(mc_mm_thread[thread]->merlin_machines[class]->ta_state);
            mc_mm_thread[thread]->merlin_machines[class]->ta_state = mc_mm->merlin_machines[class]->ta_state;
            free(mc_mm_thread[thread]->merlin_machines[class]->literal_weights);
            mc_mm_thread[thread]->merlin_machines[class]->literal_weights = mc_mm->merlin_machines[class]->literal_weights;

            mc_mm_thread[thread]->merlin_machines[class]->literal_lock = mc_mm->merlin_machines[class]->literal_lock;
        }
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
		//#pragma omp parallel for
		for (int example = 0; example < number_of_examples; example++) {
			int thread_id = omp_get_thread_num();
			unsigned int pos = example*step_size;

			mc_mm_update(mc_mm_thread[thread_id], &X[pos], y[example]);
		}
	}

	for (int class = 0; class < mc_mm->number_of_classes; class++) {
		for (int literal = 0; literal < mm->number_of_features; ++literal) {
			omp_destroy_lock(&mc_mm->merlin_machines[class]->literal_lock[literal]);
		}
	}

    for (int thread = 0; thread < max_threads; thread++) {
        for (int class = 0; class < mc_mm_thread[thread]->number_of_classes; class++) {

            struct MerlinMachine *mm_thread = mc_mm_thread[thread]->merlin_machines[class];

            free(mm_thread->literal_outputs);
			free(mm_thread->feedback_to_la);
			free(mm_thread->feedback_choice);
			free(mm_thread);
        }
    }

    free(mm->literal_lock);
    free(mc_mm_thread);

}

