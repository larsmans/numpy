#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_rand.h"

extern void
init_by_array(NPY_RandomState *self, unsigned long init_key[],
              npy_intp key_length);
