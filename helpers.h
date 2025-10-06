#ifndef HELPERS_H
#define HELPERS_H

uint compute_index(uint elem, uint size, int inc)
{
	int idx = 0;
#if 0
	// broken for now :c
	int is_inc_negative = int(inc < 0);
	int elem_coef = (2*is_inc_negative) -1;
	int new_elem = size - elem_coef*int(elem) - 1*is_inc_negative;
	idx = abs(inc)*new_elem;
#else
	if (inc < 0)
	{
		// get elements in reverse order.
		// elem 0 will be elem size - 1
		elem = uint(size) - (elem) - 1;
		idx = abs(inc)*int(elem);
	}
	else
	{
		idx = (int(elem)*inc);
	}
#endif
	return idx;

}

uint compute_mat_index(uint row_elem, uint column_elem)
{
	return 0;
}

#endif
