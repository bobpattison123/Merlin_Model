#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include "fast_rand.h"

#include "MerlinModel.h"


/* Initalizes Merlin Model struct */
struct MerlinMachine *CreateMerlinMachine(int number_of_features, int number_of_ta_chunks, int number_of_state_bits, int T, double s) {
    
    struct MerlinMachine *mm = (void *)malloc(sizeof(struct MerlinMachine));

    mm->number_of_features = number_of_features;
    mm->number_of_ta_chunks = number_of_ta_chunks;
    mm->number_of_state_bits = number_of_state_bits;
    mm->T = T;
    mm->s = s;


    mm->ta_state = (unsigned int *)malloc(sizeof(unsigned int) * number_of_ta_chunks * number_of_state_bits);
    mm->literal_outputs = (unsigned int *)malloc(sizeof(unsigned int) * number_of_ta_chunks);
    mm->feedback_to_la = (unsigned int *)malloc(sizeof(unsigned int) * number_of_ta_chunks);
    mm->literal_weights = (unsigned int *)malloc(sizeof(unsigned int) * number_of_features);

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



/* Increments TA state */
static inline void mm_inc(struct MerlinMachine *mm, int chunk, unsigned int active) {
    
    unsigned int carry, carry_next;

    unsigned int *ta_state = &mm->ta_state[mm->number_of_ta_chunks*mm->number_of_state_bits + chunk*mm->number_of_state_bits];

    /* Bob: This code essentially just implements a very basic add operation in bit form
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

	unsigned int *ta_state = &mm->ta_state[mm->number_of_ta_chunks*mm->number_of_state_bits + chunk*mm->number_of_state_bits];

    /* Bob: This code works very similar to the previous increment, the major difference being that we negate the
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

    for (int literals = 0; literals < mm->number_of_features; literals++) {
        int ta_chunk = literals / 32;
        int ta_chunk_pos = literals % 32;

        class_sum += mm->literal_weights[literals] * ((mm->literal_outputs[ta_chunk] & (1 << ta_chunk_pos)) > 0);
    }

    class_sum = (class_sum > (mm->T)) ? (mm->T) : class_sum;
	class_sum = (class_sum < -(mm->T)) ? -(mm->T) : class_sum;

    return class_sum;
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