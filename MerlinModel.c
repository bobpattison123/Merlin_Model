#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include "fast_rand.h"

#include "MerlinModel.h"


/* Initalizes Merlin Model struct */
struct MerlinMachine *CreateMerlinMachine(int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, double s_range) {
    
    struct MerlinMachine *mm = (void *)malloc(sizeof(struct MerlinMachine));

    mm->number_of_features = number_of_features;
    mm->number_of_patches = number_of_patches;
    mm->number_of_ta_chunks = number_of_ta_chunks;
    mm->number_of_state_bits = number_of_state_bits;
    mm->T = T;
    mm->s = s;

    mm->s_range = s_range;

    mm->ta_state = (unsigned int *)malloc(sizeof(unsigned int) * number_of_ta_chunks * number_of_state_bits);
    mm->literal_outputs = (unsigned int *)malloc(sizeof(unsigned int) * number_of_ta_chunks);
    mm->feedback_to_la = (unsigned int *)malloc(sizeof(unsigned int) * number_of_ta_chunks);
    mm->literal_weights = (unsigned int *)malloc(sizeof(unsigned int) * number_of_features);
    mm->feedback_choice = (int *)malloc(sizeof(int) * number_of_ta_chunks);

    if (((number_of_features) % 32) != 0) {
		mm->filter  = (~(0xffffffff << ((number_of_features) % 32)));
	} else {
		mm->filter = 0xffffffff;
	}

    mm_initialize(mm);

    return mm;
}


/* Intializes all TA states and weights */
void mm_initialize(struct MerlinMachine *mm) {

    // Initalizes TAs
    unsigned int pos_in_arr = 0;
    for(int chunk = 0; chunk < mm->number_of_ta_chunks; ++chunk) {
        for (int bit = 0; bit < mm->number_of_state_bits-1; ++bit) {
            mm->ta_state[pos_in_arr] = ~0;
            pos_in_arr++;
        }
        mm->ta_state[pos_in_arr] = 0;
        pos_in_arr++;
    }

    // Initalizes weights
    for(int literal = 0; literal < mm->number_of_features; ++literal) {
        mm->literal_weights[literal] = 1;
    }
}

static inline void mm_initialize_random_streams(struct MerlinMachine *mm, int feature)
{
	// Initialize all bits to zero	
	memset(mm->feedback_to_la, 0, mm->number_of_ta_chunks*sizeof(unsigned int));

	int n = mm->number_of_features;
	double p = 1.0 / (mm->s + 1.0 * feature * (mm->s_range - mm->s) / mm->number_of_features);

	int active = normal(n * p, n * p * (1 - p));
	active = active >= n ? n : active;
	active = active < 0 ? 0 : active;
	while (active--) {
		int f = fast_rand() % n;
		while (mm->feedback_to_la[f / 32] & (1 << (f % 32))) {
			f = fast_rand() % n;
	    }
		mm->feedback_to_la[f / 32] |= 1 << (f % 32);
	}
}


/* Increments TA state */
static inline void mm_inc(struct MerlinMachine *mm, int chunk, unsigned int active) {
    
    unsigned int carry, carry_next;

    unsigned int *ta_state = &mm->ta_state[chunk*mm->number_of_state_bits];

    /* User: This code essentially just implements a very basic add operation in bit form
            the bit operations of AND and XOR are exactly the same as the gate implementations. Looks
            more confusing than it actually is.
    */

    carry = active;
    for (int bit = 0; bit < mm->number_of_state_bits; ++bit) {

        // If the carry amount is ever zero we can leave as we no longer need to be here
        if(carry == 0)
            break;
        

        // Calculates the next carry value
        carry_next = ta_state[bit] & carry;

        // Updates the current carry value
        ta_state[bit] = ta_state[bit] ^ carry;

        // Moves the carry onto the next iteration
        carry = carry_next;
    }

    // Fills all of the chunk with 1
    if(carry > 0) {
        for(int bit = 0; bit < mm->number_of_state_bits; ++bit) {
            ta_state[bit] |= carry;
        }
    }
}


/* Decrements TA state */
static inline void mm_dec(struct MerlinMachine *mm, int chunk, unsigned int active)
{
	unsigned int carry, carry_next;

	unsigned int *ta_state = &mm->ta_state[chunk*mm->number_of_state_bits];

    /* User: This code works very similar to the previous increment, the major difference being that we negate the
            and causing the bit if zero to borrow from the next bit basically saying we need to decrement the next
            one. The rest works much the same. 
    */

	carry = active;
	for (int bit = 0; bit < mm->number_of_state_bits; ++bit) {
		if (carry == 0)
			break;

		carry_next = (~ta_state[bit]) & carry; // Sets carry bits (overflow) passing on to next bit
		ta_state[bit] = ta_state[bit] ^ carry; // Performs increments with XOR
		carry = carry_next;
	}


    // Fills all the chunk with zero
	if (carry > 0) {
		for (int bit = 0; bit < mm->number_of_state_bits; ++bit) {
			ta_state[bit] &= ~carry;
		}
	} 
}


/* Sum up votes from the literals for the class */
static inline int sum_up_class_votes(struct MerlinMachine *mm) {

    int class_sum = 0;

    // Adds up output from each literal given the literal weight
    for (int literals = 0; literals < mm->number_of_features; literals++) {
        int ta_chunk = literals / 32;
        int ta_pos = literals % 32;

        class_sum += mm->literal_weights[literals] * ((mm->literal_outputs[ta_chunk] & (1 << ta_pos)) > 0);
    }

    class_sum = (class_sum > (mm->T)) ? (mm->T) : class_sum;
	class_sum = (class_sum < -(mm->T)) ? -(mm->T) : class_sum;

    return class_sum; 
}

static inline void mm_calculate_literal_output(struct MerlinMachine *mm, unsigned int *Xi, int predict) {

    unsigned int *ta_state = mm->ta_state;

    for (int patch = 0; patch < mm->number_of_patches; ++patch) {
        for (int chunk = 0; chunk < mm->number_of_ta_chunks-1; ++chunk) {
            unsigned int pos = chunk * mm->number_of_state_bits + mm->number_of_state_bits-1;
            mm->literal_outputs[chunk] = ta_state[pos] & Xi[patch*mm->number_of_patches+chunk];
        }
        unsigned int pos = (mm->number_of_ta_chunks - 1) * mm->number_of_state_bits + mm->number_of_state_bits-1;
        mm->literal_outputs[mm->number_of_ta_chunks-1] = ta_state[pos] & Xi[patch*mm->number_of_patches+(mm->number_of_ta_chunks-1)] & mm->filter;
    }
}

/* WIP */
void mm_update_literals(struct MerlinMachine *mm, unsigned int *Xi, int class_sum, int target) {
    unsigned int *ta_state = mm->ta_state;

    for (int literal = 0; literal < mm->number_of_features; literal++) {
        mm->feedback_choice[literal] = 0;
    }
    
    for (int literal = 0; literal < mm->number_of_features; literal++) {
        mm->feedback_choice[literal] = (((float)fast_rand())/((float)FAST_RAND_MAX) <= (1.0/(mm->T*2))*(mm->T + (1 - 2*target)*class_sum));
    }
    

    for (int literal = 0; literal < mm->number_of_features; literal++) {
        unsigned int ta_chunk = literal / 32;
        unsigned int ta_pos = literal % 32;

        unsigned int lit_mask = 1 << ta_pos;
        
        if (!(mm->feedback_choice[literal])) {
			continue;
		}


        //omp_set_lock(&mm->literal_lock[literal]);

        int sign = mm->literal_weights[literal] > 0 ? 1 : -1;

        if ((2*target-1) * sign == -1) {
            if((mm->literal_outputs[ta_chunk] & (1 << ta_pos)) > 0) {

                mm->literal_weights[literal]--;

                unsigned int pos = ta_chunk * mm->number_of_state_bits + mm->number_of_state_bits-1;
                mm_inc(mm, ta_chunk, (~Xi[ta_chunk]) & (~ta_state[pos]));

            }
        } else if ((2*target-1) * sign == 1) {
            
            mm_initialize_random_streams(mm, literal);

            if((mm->literal_outputs[ta_chunk] & (1 << ta_pos)) > 0) {

                mm->literal_weights[literal]++;

                mm_inc(mm, ta_chunk, Xi[ta_chunk] & lit_mask);

                mm_dec(mm, ta_chunk, (~Xi[ta_chunk]) & mm->feedback_to_la[ta_chunk] & lit_mask);
            }
        } else {
            mm_dec(mm, ta_chunk, mm->feedback_to_la[ta_chunk] & lit_mask);
        }

        //omp_unset_lock(&mm->literal_lock[literal]);
    }
}

// Updates the clauses
void mm_update(struct MerlinMachine *mm, unsigned int *Xi, int target) {

    mm_calculate_literal_output(mm, Xi, UPDATE);

    int class_sum = sum_up_class_votes(mm);

    mm_update_literals(mm, Xi, class_sum, target);
}

// Returns class sums
int mm_score(struct MerlinMachine *mm, unsigned int *Xi) {

    mm_calculate_literal_output(mm, Xi, PREDICT);

    return sum_up_class_votes(mm);
}

/* Given a TA state finds that state */
int mm_ta_state(struct MerlinMachine *mm, int ta) {
    
    if (ta > mm->number_of_features) {
        return -1;
    }
    
    int ta_chunk = ta / 32;
    int ta_chunk_pos = ta % 32;

    unsigned int pos_in_model = mm->number_of_ta_chunks * mm->number_of_state_bits + ta_chunk * mm->number_of_state_bits;
    
    int state = 0;
    for (int bit = 0; bit < mm->number_of_state_bits; ++bit) {
        if (mm->ta_state[pos_in_model + bit] & (1 << ta_chunk_pos)) {
            state |= 1 << bit;
        }
    }

    return state;
}

int mm_ta_action(struct MerlinMachine *mm, int ta)
{
	int ta_chunk = ta / 32;
	int chunk_pos = ta % 32;

	unsigned int pos = ta_chunk * mm->number_of_state_bits + mm->number_of_state_bits-1;

	return (mm->ta_state[pos] & (1 << chunk_pos)) > 0;
}

