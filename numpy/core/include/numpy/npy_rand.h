/*
 * NumPy random core library (npyrand).
 *
 * Copyright 2013 Lars Buitinck
 * Copyright 2005 Robert Kern (robert.kern@gmail.com)
 * Copyright (c) 2003-2005, Jean-Sebastien Roy (js@jeannot.org)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 * The implementations of rk_hypergeometric_hyp, rk_hypergeometric_hrua,
 * and rk_triangular were adapted from Ivan Frohne's rv.py which has this
 * license:
 *
 *            Copyright 1998 by Ivan Frohne; Wasilla, Alaska, U.S.A.
 *                            All Rights Reserved
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation for any purpose, free of charge, is granted subject to the
 * following conditions:
 *   The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the software.
 *
 *   THE SOFTWARE AND DOCUMENTATION IS PROVIDED WITHOUT WARRANTY OF ANY KIND,
 *   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO MERCHANTABILITY, FITNESS
 *   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHOR
 *   OR COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM OR DAMAGES IN A CONTRACT
 *   ACTION, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *   SOFTWARE OR ITS DOCUMENTATION.
 */

/*
 * npyrand is based on the randomkit library, version 1.3.
 * @(#) $Jeannot: randomkit.h,v 1.24 2005/07/21 22:14:09 js Exp $
 */

/*
 * Typical use:
 *
 * {
 *  NPY_RandomState state;
 *  unsigned long seed = 1, random_value;
 *
 *  rk_seed(seed, &state); // Initialize the RNG
 *  ...
 *  random_value = rk_random(&state); // Generate random values in [0..RK_MAX]
 * }
 *
 * Instead of rk_seed, you can use rk_randomseed which will get a random seed
 * from /dev/urandom (or the clock, if /dev/urandom is unavailable):
 *
 * {
 *  NPY_RandomState state;
 *  unsigned long random_value;
 *
 *  rk_randomseed(&state); // Initialize the RNG with a random seed
 *  ...
 *  random_value = rk_random(&state); // Generate random values in [0..RK_MAX]
 * }
 */

/*
 * Useful macro:
 *  RK_DEV_RANDOM: the device used for random seeding.
 *                 defaults to "/dev/urandom"
 */

#ifndef NPY_RAND_H_
#define NPY_RAND_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#ifdef HAVE_NPY_CONFIG_H
#include <npy_config.h>
#endif
#include <numpy/npy_common.h>

#define RK_STATE_LEN 624

typedef struct {
    unsigned long key[RK_STATE_LEN];
    int pos;
    int has_gauss; /* !=0: gauss contains a gaussian deviate */
    double gauss;

    /* The NPY_RandomState structure has been extended to store the following
     * information for the binomial generator. If the input values of n or p
     * are different than nsave and psave, then the other parameters will be
     * recomputed. RTK 2005-09-02 */

    int has_binomial; /* !=0: following parameters initialized for
                              binomial */
    double psave;
    long nsave;
    double r;
    double q;
    double fm;
    long m;
    double p1;
    double xm;
    double xl;
    double xr;
    double c;
    double laml;
    double lamr;
    double p2;
    double p3;
    double p4;
} NPY_RandomState;

typedef enum {
    RK_NOERR = 0, /* no error */
    RK_ENODEV = 1, /* no RK_DEV_RANDOM device */
    RK_ERR_MAX = 2
} NPY_RandomError;

/* error strings */
extern const char *rk_strerror[RK_ERR_MAX];

/* Maximum generated random value */
#define RK_MAX 0xFFFFFFFFUL

/*
 * Initialize the RNG state using the given seed.
 */
void rk_seed(NPY_RandomState *state, unsigned long seed);

/*
 * Initialize the RNG state using a random seed.
 * Uses /dev/random or, when unavailable, the clock (see randomkit.c).
 * Returns RK_NOERR when no errors occurs.
 * Returns RK_ENODEV when the use of RK_DEV_RANDOM failed (for example because
 * there is no such device). In this case, the RNG was initialized using the
 * clock.
 */
NPY_RandomError rk_randomseed(NPY_RandomState *state);

/*
 * Returns 32 random bits.
 */
npy_uint32 npy_random_u32(NPY_RandomState *state);

/*
 * Returns a random long between 0 and LONG_MAX inclusive
 */
long rk_long(NPY_RandomState *state);

/*
 * Returns a random unsigned long between 0 and ULONG_MAX inclusive
 */
unsigned long rk_ulong(NPY_RandomState *state);

/*
 * Returns a random unsigned long between 0 and max inclusive.
 */
unsigned long rk_interval(NPY_RandomState *state, unsigned long max);

/*
 * Returns a random double between 0.0 and 1.0, 1.0 excluded.
 */
double rk_double(NPY_RandomState *state);

/*
 * fill the buffer with size random bytes
 */
void rk_fill(NPY_RandomState *state, void *buffer, size_t size);

/*
 * fill the buffer with randombytes from the random device
 * Returns RK_ENODEV if the device is unavailable, or RK_NOERR if it is
 * On Unix, if strong is defined, RK_DEV_RANDOM is used. If not, RK_DEV_URANDOM
 * is used instead. This parameter has no effect on Windows.
 * Warning: on most unixes RK_DEV_RANDOM will wait for enough entropy to answer
 * which can take a very long time on quiet systems.
 */
NPY_RandomError rk_devfill(void *buffer, size_t size, int strong);

/*
 * fill the buffer using rk_devfill if the random device is available and using
 * rk_fill if is is not
 * parameters have the same meaning as rk_fill and rk_devfill
 * Returns RK_ENODEV if the device is unavailable, or RK_NOERR if it is
 */
/*
NPY_RandomError rk_altfill(NPY_RandomState *state,
                           void *buffer, size_t size, int strong);
*/

/*
 * return a random gaussian deviate with variance unity and zero mean.
 */
double rk_gauss(NPY_RandomState *state);



/* References:
 *
 * Devroye, Luc. _Non-Uniform Random Variate Generation_.
 *  Springer-Verlag, New York, 1986.
 *  http://cgm.cs.mcgill.ca/~luc/rnbookindex.html
 *
 * Kachitvichyanukul, V. and Schmeiser, B. W. Binomial Random Variate
 *  Generation. Communications of the ACM, 31, 2 (February, 1988) 216.
 *
 * Hoermann, W. The Transformed Rejection Method for Generating Poisson Random
 *  Variables. Insurance: Mathematics and Economics, (to appear)
 *  http://citeseer.csail.mit.edu/151115.html
 *
 * Marsaglia, G. and Tsang, W. W. A Simple Method for Generating Gamma
 * Variables. ACM Transactions on Mathematical Software, Vol. 26, No. 3,
 * September 2000, Pages 363–372.
 */

/* Normal distribution with mean=loc and standard deviation=scale. */
double rk_normal(NPY_RandomState *state, double loc, double scale);

/* Standard exponential distribution (mean=1) computed by inversion of the
 * CDF. */
double rk_standard_exponential(NPY_RandomState *state);

/* Exponential distribution with mean=scale. */
double rk_exponential(NPY_RandomState *state, double scale);

/* Uniform distribution on interval [loc, loc+scale). */
double rk_uniform(NPY_RandomState *state, double loc, double scale);

/* Standard gamma distribution with shape parameter.
 * When shape < 1, the algorithm given by (Devroye p. 304) is used.
 * When shape == 1, a Exponential variate is generated.
 * When shape > 1, the small and fast method of (Marsaglia and Tsang 2000)
 * is used.
 */
double rk_standard_gamma(NPY_RandomState *state, double shape);

/* Gamma distribution with shape and scale. */
double rk_gamma(NPY_RandomState *state, double shape, double scale);

/* Beta distribution computed by combining two gamma variates (Devroye p. 432).
 */
double rk_beta(NPY_RandomState *state, double a, double b);

/* Chi^2 distribution computed by transforming a gamma variate (it being a
 * special case Gamma(df/2, 2)). */
double rk_chisquare(NPY_RandomState *state, double df);

/* Noncentral Chi^2 distribution computed by modifying a Chi^2 variate. */
double rk_noncentral_chisquare(NPY_RandomState *state, double df, double nonc);

/* F distribution computed by taking the ratio of two Chi^2 variates. */
double rk_f(NPY_RandomState *state, double dfnum, double dfden);

/* Noncentral F distribution computed by taking the ratio of a noncentral Chi^2
 * and a Chi^2 variate. */
double rk_noncentral_f(NPY_RandomState *state, double dfnum, double dfden, double nonc);

/* Binomial distribution with n Bernoulli trials with success probability p.
 * When n*p <= 30, the "Second waiting time method" given by (Devroye p. 525) is
 * used. Otherwise, the BTPE algorithm of (Kachitvichyanukul and Schmeiser 1988)
 * is used. */
long rk_binomial(NPY_RandomState *state, long n, double p);

/* Binomial distribution using BTPE. */
long rk_binomial_btpe(NPY_RandomState *state, long n, double p);

/* Binomial distribution using inversion and chop-down */
long rk_binomial_inversion(NPY_RandomState *state, long n, double p);

/* Negative binomial distribution computed by generating a Gamma(n, (1-p)/p)
 * variate Y and returning a Poisson(Y) variate (Devroye p. 543). */
long rk_negative_binomial(NPY_RandomState *state, double n, double p);

/* Poisson distribution with mean=lam.
 * When lam < 10, a basic algorithm using repeated multiplications of uniform
 * variates is used (Devroye p. 504).
 * When lam >= 10, algorithm PTRS from (Hoermann 1992) is used.
 */
long rk_poisson(NPY_RandomState *state, double lam);

/* Poisson distribution computed by repeated multiplication of uniform variates.
 */
long rk_poisson_mult(NPY_RandomState *state, double lam);

/* Poisson distribution computer by the PTRS algorithm. */
long rk_poisson_ptrs(NPY_RandomState *state, double lam);

/* Standard Cauchy distribution computed by dividing standard gaussians
 * (Devroye p. 451). */
double rk_standard_cauchy(NPY_RandomState *state);

/* Standard t-distribution with df degrees of freedom (Devroye p. 445 as
 * corrected in the Errata). */
double rk_standard_t(NPY_RandomState *state, double df);

/* von Mises circular distribution with center mu and shape kappa on [-pi,pi]
 * (Devroye p. 476 as corrected in the Errata). */
double rk_vonmises(NPY_RandomState *state, double mu, double kappa);

/* Pareto distribution via inversion (Devroye p. 262) */
double rk_pareto(NPY_RandomState *state, double a);

/* Weibull distribution via inversion (Devroye p. 262) */
double rk_weibull(NPY_RandomState *state, double a);

/* Power distribution via inversion (Devroye p. 262) */
double rk_power(NPY_RandomState *state, double a);

/* Laplace distribution */
double rk_laplace(NPY_RandomState *state, double loc, double scale);

/* Gumbel distribution */
double rk_gumbel(NPY_RandomState *state, double loc, double scale);

/* Logistic distribution */
double rk_logistic(NPY_RandomState *state, double loc, double scale);

/* Log-normal distribution */
double rk_lognormal(NPY_RandomState *state, double mean, double sigma);

/* Rayleigh distribution */
double rk_rayleigh(NPY_RandomState *state, double mode);

/* Wald distribution */
double rk_wald(NPY_RandomState *state, double mean, double scale);

/* Zipf distribution */
long rk_zipf(NPY_RandomState *state, double a);

/* Geometric distribution */
long rk_geometric(NPY_RandomState *state, double p);
long rk_geometric_search(NPY_RandomState *state, double p);
long rk_geometric_inversion(NPY_RandomState *state, double p);

/* Hypergeometric distribution */
long rk_hypergeometric(NPY_RandomState *state, long good, long bad, long sample);
long rk_hypergeometric_hyp(NPY_RandomState *state, long good, long bad, long sample);
long rk_hypergeometric_hrua(NPY_RandomState *state, long good, long bad, long sample);

/* Triangular distribution */
double rk_triangular(NPY_RandomState *state, double left, double mode, double right);

/* Logarithmic series distribution */
long rk_logseries(NPY_RandomState *state, double p);


#ifdef __cplusplus
}
#endif

#endif /* NPY_RAND_H_ */
