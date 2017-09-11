extern crate num;
use num::{One, Zero, BigInt, BigUint, pow};

use std::cmp::Ord;
use std::fmt;
use std::ops::{Mul, Rem, ShrAssign, BitAnd};
use std::iter::{Iterator};
use super::iterators::ForeverRange;
use prelude::cannot_happen;
use itertools;

pub enum PossiblePrimesState {
    Two, Three, TailOne, TailFive
}

pub struct PossiblePrimes {
    state: PossiblePrimesState,
    offset: BigInt
}

impl PossiblePrimes {
}

impl Iterator for PossiblePrimes {
    type Item = BigInt;

    fn next(&mut self) -> Option<BigInt> {
        let (next_state, output) = match self.state {
            PossiblePrimesState::Two =>
                (PossiblePrimesState::Three, BigInt::from(2)),

            PossiblePrimesState::Three =>
                (PossiblePrimesState::TailFive, BigInt::from(3)),

            PossiblePrimesState::TailOne =>
                (PossiblePrimesState::TailFive, &self.offset + &BigInt::from(1)),

            PossiblePrimesState::TailFive => {
                let ret = (PossiblePrimesState::TailOne, &self.offset + &BigInt::from(5));
                self.offset = &self.offset + &BigInt::from(6);
                ret
            }
        };

        self.state = next_state;
        Some(output)
    }
}

pub struct RootConvergentIter {
    base_iter: RootContFracIter,
    iteration: usize,
    p_prev: BigInt,
    p_prev_prev: BigInt,
    q_prev: BigInt,
    q_prev_prev: BigInt,
}

impl RootConvergentIter {
    pub fn new(d: &BigInt) -> RootConvergentIter {
        use num::ToPrimitive;

        let out = RootConvergentIter {
            base_iter: RootContFracIter::new(d.to_i64().unwrap()),
            iteration: 0,
            p_prev: BigInt::from(0),
            p_prev_prev: BigInt::from(0),
            q_prev: BigInt::from(0),
            q_prev_prev: BigInt::from(0),
        };

        out
    }
}

impl Iterator for RootConvergentIter {
    type Item = (BigInt, BigInt);

    fn next(&mut self) -> Option<Self::Item> {
        use std::mem::replace;

        self.iteration += 1;

        let a = self.base_iter.next().unwrap();

        let (p, q) = match self.iteration {
            1 => { (a, BigInt::from(1)) },
            2 => { (&a * &self.p_prev + BigInt::from(1), a) },
            _ => { (&(&a * &self.p_prev) + &self.p_prev_prev, &(&a * &self.q_prev) + &self.q_prev_prev)}
        };

        self.p_prev_prev = replace(&mut self.p_prev, p.clone());
        self.q_prev_prev = replace(&mut self.q_prev, q.clone());
        self.q_prev = q.clone();

        Some((p, q))
    }
}

pub struct PrimePowerFactors {
    rem: BigInt,
    last_p: PossiblePrimes,
}

impl PrimePowerFactors {
    ///
    /// Iterator through the (maximal) prime power factors of the specified number.
    ///
    /// # Examples
    /// ```
    /// extern crate num;
    /// use num::BigInt;
    ///
    /// # extern crate euler_lib; use euler_lib::numerics::PrimePowerFactors;
    /// # fn main() {
    /// let actual:   Vec<_> = PrimePowerFactors::from(&BigInt::from(36)).collect();
    /// let expected: Vec<_> = vec![BigInt::from(4), BigInt::from(9)];
    ///
    /// assert_eq!(actual, expected);
    /// # }
    /// ```
    ///
    pub fn from(n: &BigInt) -> PrimePowerFactors {
        PrimePowerFactors {
            rem: n.clone(),
            last_p: possible_primes(),
        }
    }
}

impl Iterator for PrimePowerFactors {
    type Item = BigInt;

    fn next(&mut self) -> Option<BigInt> {
        if self.rem == BigInt::from(1) {
            None
        } else {
            let zero = BigInt::from(0);
            let mut next_p;
            loop {
                next_p = self.last_p.next().unwrap(); // PossiblePrimes never returns None
                if &self.rem % &next_p == zero {
                    break;
                } else if &next_p * &next_p > self.rem {
                    let out = self.rem.clone();
                    self.rem = BigInt::from(1);
                    return Some(out)
                }
            }

            let mut out = next_p.clone();
            self.rem = &self.rem / &next_p;

            while &self.rem % &next_p == zero {
                out = &out * &next_p;
                self.rem = &self.rem / &next_p;
            }

            Some(out)
        }
    }
}

pub struct RootContFracIter {
    partial: Partial, // state so far
}

impl RootContFracIter {
    pub fn new(d: i64) -> RootContFracIter {
        let mut out = RootContFracIter {
            partial: Partial { d: d, a: 0, b: 1 },
        };
        out.next();
        out
    }
}

impl Iterator for RootContFracIter {
    type Item = BigInt;

    fn next(&mut self) -> Option<Self::Item> {
        let (r, next) = self.partial.next();
        self.partial = next;
        Some(BigInt::from(r))
    }
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Copy, Clone)]
pub struct Partial { // TODO: should this be BigInt?
    pub d: i64,
    pub a: i64,
    pub b: i64,
}

impl fmt::Display for Partial {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(sqrt {} + {}) / {}", self.d, self.a, self.b)
    }
}

impl Partial {
    pub fn next(&self) -> (i64, Partial) {
        if self.is_dead() {
            return (0, *self);
        }

        if (self.d - self.a * self.a) % self.b != 0 {
            panic!("I assumed there was no need for T!");
        }

        let mut next_b = (self.d - self.a * self.a) / self.b;
        let mut next_a = -self.a;

        if next_b < 0 {
            next_b = -next_b;
            next_a = -next_a;
        }

        let mut next_r = 0;

        loop {
            let quant = -next_a + next_r * next_b;
            if quant > 0 && quant*quant > self.d {
                next_r -= 1;
            } else {
                break;
            }
        }

        loop {
            let quant = next_b * (next_r + 1) - next_a;
            if quant <= 0 || quant*quant <= self.d {
                next_r += 1;
            } else {
                break;
            }
        }

        next_a -= next_b * next_r;

        if !(next_a >= 0 || next_a * next_a <= self.d) {
            panic!(">= 0 not satisfied!");
        }

        if !(next_b - next_a > 0 && pow(next_b - next_a, 2) > self.d) {
            panic!(">= 1 not satisfied!");
        }

        let out = Partial { d: self.d, b: next_b, a: next_a };

        (next_r, out)
    }

    pub fn is_dead(&self) -> bool {
        self.d == self.a * self.a
    }
}

///
/// Goes through the natural numbers in order, skipping some composite numbers but hitting every
/// prime number. This is an infinite iterator.
///
/// Specifically, this yields 2 and 3, then every odd number which is not divisible by three.
/// This always returns BigInts.
///
/// # Examples
///
/// ```
/// extern crate num;
/// use num::BigInt;
///
/// # extern crate euler_lib; use euler_lib::numerics::possible_primes;
/// # fn main() {
/// let actual:   Vec<_> = possible_primes().take(8).collect();
/// let expected: Vec<_> = vec![2, 3, 5, 7, 11, 13, 17, 19].iter()
///                        .map(|n: &i64| BigInt::from(*n)).collect::<Vec<BigInt>>();
/// assert_eq!(actual, expected);
/// # }
/// ```
///
pub fn possible_primes() -> PossiblePrimes {
    /*
    let a: Vec<i64> = vec![1, 2, 3, 4];
    let b: Vec<BigInt> = vec![1, 2, 3, 4].iter().map(|n: &i64| BigInt::from(*n)).collect::<Vec<BigInt>>();
    */

    PossiblePrimes { state: PossiblePrimesState::Two, offset: BigInt::from(0) }
}

/// Raises a value to the power of exp, modulo the specified modulus. This uses exponentiation by
/// squaring and is more or less ripped from the num crate.
///
/// This has potential for internal overflow, even if the final result is representable in your
/// type, so it is recommended to use only with BigInt. However it is safe for smaller types,
/// assuming both of the following are verified:
///     (a) base ^ 2 is below the maximum value of your type, and
///     (b) modulo ^ 2 is below the maximum value of your type
///
/// # Example
///
/// ```rust
/// use euler_lib::numerics::powmod;
///
/// assert_eq!(powmod(15, 2, 500_u64), 225);
/// assert_eq!(powmod(2, 7, 127_u64), 1);
/// ```
#[inline]
pub fn powmod<T, E>(mut base: T, mut exp: E, modulo: T) -> T
    where T: Clone + One + Mul<T, Output = T> + Rem<T, Output = T> + MulMod,
          E: Copy + One + Zero + Eq + ShrAssign<E> + Ord + BitAnd<E, Output=E>
{
    if exp == E::zero() { return T::one() }

    while exp & E::one() == E::zero() {
        base = mul_mod_ptr(&base, &base, &modulo);
        exp >>= E::one();
    }
    if exp == E::one() { return base }

    let mut acc = base.clone();
    while exp > E::one() {
        exp >>= E::one();
        base = mul_mod_ptr(&base, &base, &modulo);
        if exp & E::one() == E::one() {
            acc = mul_mod_ptr(&acc, &base, &modulo);
        }
    }
    acc
}

///
/// Computes the floor of the square root of the argument.
///
/// # Examples
///
/// ```
/// # extern crate euler_lib; use euler_lib::numerics::sqrt_floor;
/// # fn main() {
/// assert_eq!(sqrt_floor(12), 3);
/// assert_eq!(sqrt_floor(16), 4);
/// assert_eq!(sqrt_floor(169), 13);
/// assert_eq!(sqrt_floor(168), 12);
/// # }
/// ```
pub fn sqrt_floor(n: u64) -> u64 {
    let mut low = 1;
    let mut high = n;

    while low + 1 < high {
        let mid = (low + high) / 2;
        if mid * mid <= n {
            low = mid;
        } else {
            high = mid;
        }
    }

    if high * high <= n {
        high
    } else {
        low
    }
}

///
/// Computes the gcd of the specified arguments.
///
/// # Examples
///
/// ```
/// # extern crate euler_lib; use euler_lib::numerics::gcd;
/// # fn main() {
/// assert_eq!(gcd(12, 15), 3);
/// assert_eq!(gcd(21, 25), 1);
/// assert_eq!(gcd(100, -45), 5);
/// assert_eq!(gcd(0, 0), 0);
/// assert_eq!(gcd(47, 1), 1);
/// # }
/// ```
pub fn gcd(mut x: i64, mut y: i64) -> i64 {
    if x == 0 {
        y
    } else {
        if x < 0 {
            x = -x;
        }
        if y < 0 {
            y = -y;
        }

        while y != 0 {
            let m = x % y;
            x = y;
            y = m;
        }

        x
    }
}

///
/// Returns the largest prime factor of the specified BigInt.
///
/// This runs in O(sqrt(n) log n) time, which is O(sqrt(n)) arithmetic expressions on BigInts of
/// commensurate size to n, which therefore have approximately O(log n) running time. But in most
/// cases the sqrt(n) is unrealistically large (e.g. when `n` has lots of prime factors).
///
/// # Examples
///
/// ```
/// extern crate num;
/// use num::bigint::BigInt;
///
/// # extern crate euler_lib; use euler_lib::numerics::largest_prime_factor;
/// # fn main() {
/// let actual = largest_prime_factor(&BigInt::from(32));
/// let expected = BigInt::from(2);
///
/// assert_eq!(actual, expected);
///
/// let actual = largest_prime_factor(&BigInt::from(145));
/// let expected = BigInt::from(29);
///
/// assert_eq!(actual, expected);
///
/// let actual = largest_prime_factor(&BigInt::from(37));
/// let expected = BigInt::from(37);
///
/// assert_eq!(actual, expected);
///
/// let actual = largest_prime_factor(&BigInt::from(-32));
/// let expected = BigInt::from(2);
///
/// assert_eq!(actual, expected);
///
/// let actual = largest_prime_factor(&BigInt::from(0));
/// let expected = BigInt::from(1);
///
/// assert_eq!(actual, expected);
/// # }
/// ```
///
pub fn largest_prime_factor(n: &BigInt) -> BigInt {
    let zero = BigInt::from(0);
    let one = BigInt::from(1);

    let mut n = n.clone();

    if n < zero {
        n = -n;
    }

    if n <= one {
        return one;
    }

    for p in possible_primes() {
        if &p * &p > n {
            return n;
        }

        while &n % &p == zero {
            n = &n / &p;
        }

        if n == one {
            return p;
        }
    }

    cannot_happen();
}

///
/// Determines if the incoming number is prime.
///
/// This is an extremely simple method, just testing for divisibility by possible primes up to
/// sqrt(n), skipping even numbers (past 2) and multiples of three (past 3).
///
/// Runs in O(sqrt(n) log(n)) time, worst case is when n is prime or has no small prime factors.
///
/// # Examples
///
/// ```
/// extern crate num;
/// use num::bigint::BigInt;
///
/// # extern crate euler_lib; use euler_lib::numerics::is_prime;
/// # fn main() {
/// assert!(is_prime(&BigInt::from(37)));
/// assert!(!is_prime(&BigInt::from(117)));
/// # }
/// ```
///
pub fn is_prime(n: &BigInt) -> bool {
    if *n < BigInt::from(2) {
        return false;
    }

    let zero = BigInt::from(0);

    for p in possible_primes() {
        if &p * &p > *n {
            return true;
        } else if n % &p == zero {
            return false;
        }
    }

    cannot_happen();
}

///
/// Gives a Vector of all primes up to (but not including) the given cap.
///
/// Keeps things as i64 since running this with a BigInt cap is unrealistic. Running time is a bit
/// involved to compute but is approximately O(n^2 / log n). This uses a sieve method.
///
/// # Examples
///
/// ```
/// # extern crate euler_lib; use euler_lib::numerics::all_primes;
/// # fn main() {
/// assert_eq!(all_primes(10), vec![2, 3, 5, 7]);
/// assert_eq!(all_primes(2), vec![]);
/// # }
/// ```
///
pub fn all_primes(cap: usize) -> Vec<usize> {
    let mut bools = Vec::with_capacity(cap);

    bools.push(false);
    bools.push(false);

    for _ in 2 .. cap {
        bools.push(true);
    }

    for p in 2 .. cap { // TODO -- use possible primes here (requires a usize possible_primes)
        if bools[p] {
            for k in ForeverRange::new(p*p, |n| Some(n + p)).take_while(|&n| n < cap) {
                bools[k] = false;
            }
        }
        if p*p >= cap {
            break;
        }
    }

    bools.iter()
        .enumerate()
        .filter(|val| *val.1)
        .map(|val| val.0)
        .collect()
}

///
/// Computes the number of distinct prime divisors of n.
///
/// # Examples
/// ```
/// extern crate num;
/// use num::BigInt;
///
/// # extern crate euler_lib; use euler_lib::numerics::num_prime_divisors;
/// # pub fn main() {
/// assert_eq!(num_prime_divisors(BigInt::from(1)), BigInt::from(0));
/// assert_eq!(num_prime_divisors(BigInt::from(10)), BigInt::from(2));
/// assert_eq!(num_prime_divisors(BigInt::from(37)), BigInt::from(1));
/// # }
/// ```
pub fn num_prime_divisors(mut n: BigInt) -> BigInt {
    let zero = BigInt::from(0);
    let one = BigInt::from(1);
    let two = BigInt::from(2);

    if n < two {
        return zero;
    }

    let mut num_divisors = BigInt::from(0);

    for p in possible_primes() {
        if &n % &p == zero {
            num_divisors = &num_divisors + &one;
            n = &n / &p;
            while &n % &p == zero {
                n = &n / &p;
            }
        }

        if &p * &p > n {
            if n > one {
                num_divisors = &num_divisors + &one;
            }

            break;
        }
    }

    num_divisors
}

///
/// Given n, computes the number of distinct prime divisors for all integers
/// from 0 to n-1. Considers 0 and 1 to have no prime divisors (0 is by convention).
///
/// ```
/// extern crate num;
/// use num::BigInt;
///
/// # extern crate euler_lib; use euler_lib::numerics::num_prime_div_sieve;
/// # pub fn main() {
/// assert_eq!(num_prime_div_sieve(11), vec![0, 0, 1, 1, 1, 1, 2, 1, 1, 1, 2]);
/// # }
/// ```
pub fn num_prime_div_sieve(cap: usize) -> Vec<u64> {
    let mut out = vec![0; cap];

    // TODO R3 -- make a usize version of possible_primes. Do it in a cute way, too???
    for p in 2 .. cap {
        if *out.get(p).unwrap() == 0 {
            // then p is prime
            for k in itertools::unfold(0, |state| { *state += p; Some(*state) }).take_while(|n| *n < cap) {
                *out.get_mut(k).unwrap() += 1;
            }
        }
    }

    out
}

///
/// Computes the number of divisors of n, including 1 and n.
///
/// # Examples
/// ```
/// extern crate num;
/// use num::BigInt;
///
/// # extern crate euler_lib; use euler_lib::numerics::num_divisors;
/// # pub fn main() {
/// assert_eq!(num_divisors(BigInt::from(1)), BigInt::from(1));
/// assert_eq!(num_divisors(BigInt::from(10)), BigInt::from(4));
/// assert_eq!(num_divisors(BigInt::from(37)), BigInt::from(2));
/// # }
/// ```
pub fn num_divisors(mut n: BigInt) -> BigInt {
    let zero = BigInt::from(0);
    let one = BigInt::from(1);
    let two = BigInt::from(2);

    if n < two {
        return one;
    }

    let mut num_divisors = BigInt::from(1);

    for p in possible_primes() {
        if &n % &p == zero {
            let mut count = 2; // 1 for "start", plus one for the actual division
            n = &n / &p;
            while &n % &p == zero {
                count += 1;
                n = &n / &p;
            }
            num_divisors = num_divisors * BigInt::from(count);
        }

        if &p * &p > n {
            if n > one {
                num_divisors = &num_divisors * &two;
            }

            break;
        }
    }

    num_divisors
}

///
/// Computes the sum of divisors of n, including 1 and n.
///
/// # Examples
/// ```
/// extern crate num;
/// use num::BigInt;
///
/// # extern crate euler_lib; use euler_lib::numerics::sum_divisors;
/// # pub fn main() {
/// assert_eq!(sum_divisors(BigInt::from(1)), BigInt::from(1));
/// assert_eq!(sum_divisors(BigInt::from(10)), BigInt::from(1+2+5+10));
/// assert_eq!(sum_divisors(BigInt::from(37)), BigInt::from(1+37));
/// # }
/// ```
pub fn sum_divisors(mut n: BigInt) -> BigInt {
    let zero = BigInt::from(0);
    let one = BigInt::from(1);
    let two = BigInt::from(2);

    if n < two {
        return one;
    }

    let mut sum_divisors = BigInt::from(1);

    for p in possible_primes() {
        if &n % &p == zero {
            let mut count = &p + &one; // 1 for "start", plus p for the actual division
            n = &n / &p;
            while &n % &p == zero {
                count = &count * &p + &one;
                n = &n / &p;
            }
            sum_divisors = sum_divisors * count;
        }

        if &p * &p > n {
            if n > one {
                sum_divisors = &sum_divisors * &(&one + &n);
            }

            break;
        }
    }

    sum_divisors
}

///
/// Computes the sum of divisors of n, including 1 and n.
///
/// This uses the multiplicativity of the sum_divisors function, and runs in better than n log n
/// time for large enough n -- actually the log n factor is the expected number of primes in the
/// factorization of n, which is bounded above by log_2 n.
///
/// This is overflow-safe when 2*cap fits in usize.
///
/// # Examples
/// ```
/// extern crate num;
/// use num::BigInt;
///
/// # extern crate euler_lib; use euler_lib::numerics::all_sum_divisors;
/// # pub fn main() {
/// let actual = all_sum_divisors(10);
/// let expected = vec![0, 1, 3, 4, 7, 6, 12, 8, 15, 13];
/// assert_eq!(actual, expected);
/// # }
/// ```
pub fn all_sum_divisors(cap: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(cap);

    out.push(0);

    for _ in 1 .. cap {
        out.push(1);
    }

    for p in 2 .. cap {
        if out[p] == 1 {
            // then it's a prime!

            let mut k = p;
            while k < cap {
                let mut mult = 1;
                let mut n = k;

                while n % p == 0 {
                    n /= p;
                    mult = p * mult + 1;
                }

                out[k] *= mult;

                k += p;
            }
        }
    }

    out
}

///
/// Computes Euler's totient for n.
///
/// That is, the number of integers m where 1 <= m <= n and gcd(m, n) == 1.
/// Uses multiplicativity of the totient.
///
/// By convention, totient(0) == 0
///
/// # Examples
/// ```
/// extern crate euler_lib;
/// use euler_lib::numerics::totient;
///
/// # fn main() {
/// assert_eq!(totient(1), 1);
/// assert_eq!(totient(2), 1);
/// assert_eq!(totient(3), 2);
/// assert_eq!(totient(4), 2);
/// assert_eq!(totient(5), 4);
/// assert_eq!(totient(6), 2);
/// assert_eq!(totient(7), 6);
/// assert_eq!(totient(8), 4);
/// assert_eq!(totient(9), 6);
/// assert_eq!(totient(10), 4);
/// # }
/// ```
pub fn totient(mut n: u64) -> u64 {
    if n == 0 {
        return 0;
    }

    let mut totient = 1;
    if n % 2 == 0 {
        n /= 2;
        let mut prod = 1;
        while n % 2 == 0 {
            n /= 2;
            prod *= 2;
        }
        totient *= prod;
    }

    for p in itertools::unfold(1, |state| { *state += 2; Some(*state) }) {
        if n % p == 0 {
            n /= p;
            let mut prod = p - 1;
            while n % p == 0 {
                n /= p;
                prod *= p;
            }
            totient *= prod;
        } else if p*p > n {
            if n > 1 {
                totient *= n - 1;
            }
            break;
        }
    }

    totient
}

///
/// Gives a Vec of the totient values, from 0 to cap (not including cap).
///
/// Uses multiplicativity and a sieve algorithm.
///
/// # Examples
/// ```
/// extern crate euler_lib;
/// use euler_lib::numerics::all_totient;
///
/// # fn main() {
/// let actual = all_totient(10);
/// let expected = vec![0, 1, 1, 2, 2, 4, 2, 6, 4, 6];
/// assert_eq!(actual, expected);
/// # }
/// ```
pub fn all_totient(cap: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(cap);

    out.push(0); // phi(0) == 0

    for _ in 1 .. cap {
        out.push(1);
    }

    for p in 2 .. cap {
        // then it's prime
        if *out.get(p).unwrap() == 1 {
            for pmult in itertools::unfold(0, |state| { *state += p; Some(*state) }).take_while(|&k| k < cap) {
                *out.get_mut(pmult).unwrap() *= p-1;
            }

            for ppow in itertools::unfold(p, |m| { *m *= p; Some(*m) }).take_while(|&k| k < cap) {
                for ppowmult in itertools::unfold(0, |m| { *m += ppow; Some(*m) }).take_while(|&k| k < cap) {
                    *out.get_mut(ppowmult).unwrap() *= p;
                }
            }
        }
    }

    out
}

pub trait MulMod {
    fn mul_mod(&self, other: &Self, modulus: &Self) -> Self;
}

impl MulMod for BigInt {
    fn mul_mod(&self, other: &Self, modulus: &Self) -> Self {
        // TODO: there is a better way to do this
        &(self * other) % modulus
    }
}

impl MulMod for BigUint {
    fn mul_mod(&self, other: &Self, modulus: &Self) -> Self {
        // TODO: there is a better way to do this
        &(self * other) % modulus
    }
}

impl MulMod for u64 {
    fn mul_mod(&self, other: &u64, modulus: &u64) -> u64 {
        let mut x = *self;
        let mut y = *other;
        let m = *modulus;

        let msb = 0x8000_0000_0000_0000;
        let mut d = 0;
        let mp2 = m >> 1;
        x %= m;
        y %= m;

        if m & msb == 0 {
            for _ in 0..64 {
                d = if d > mp2 {
                    (d << 1) - m
                } else {
                    d << 1
                };
                if x & msb != 0 {
                    d += y;
                }
                if d >= m {
                    d -= m;
                }
                x <<= 1;
            }
            d
        } else {
            for _ in 0..64 {
                d = if d > mp2 {
                    d.wrapping_shl(1).wrapping_sub(m)
                } else {
                    // the case d == m && x == 0 is taken care of
                    // after the end of the loop
                    d << 1
                };
                if x & msb != 0 {
                    let (mut d1, overflow) = d.overflowing_add(y);
                    if overflow {
                        d1 = d1.wrapping_sub(m);
                    }
                    d = if d1 >= m { d1 - m } else { d1 };
                }
                x <<= 1;
            }
            if d >= m { d - m } else { d }
        }
    }
}

impl MulMod for u32 {
    fn mul_mod(&self, other: &u32, modulus: &u32) -> u32 {
        let mut x = *self;
        let mut y = *other;
        let m = *modulus;

        let msb = 0x8000_0000;
        let mut d = 0;
        let mp2 = m >> 1;
        x %= m;
        y %= m;

        if m & msb == 0 {
            for _ in 0..32 {
                d = if d > mp2 {
                    (d << 1) - m
                } else {
                    d << 1
                };
                if x & msb != 0 {
                    d += y;
                }
                if d >= m {
                    d -= m;
                }
                x <<= 1;
            }
            d
        } else {
            for _ in 0..32 {
                d = if d > mp2 {
                    d.wrapping_shl(1).wrapping_sub(m)
                } else {
                    // the case d == m && x == 0 is taken care of
                    // after the end of the loop
                    d << 1
                };
                if x & msb != 0 {
                    let (mut d1, overflow) = d.overflowing_add(y);
                    if overflow {
                        d1 = d1.wrapping_sub(m);
                    }
                    d = if d1 >= m { d1 - m } else { d1 };
                }
                x <<= 1;
            }
            if d >= m { d - m } else { d }
        }
    }
}

pub fn mul_mod<T: MulMod + Copy>(x: T, y: T, m: T) -> T {
    x.mul_mod(&y, &m)
}

pub fn mul_mod_ptr<T: MulMod>(x: &T, y: &T, m: &T) -> T {
    x.mul_mod(y, m)
}

pub fn mul_mod_u32(x: u32, y: u32, m: u32) -> u32 {
    mul_mod(x, y, m)
}

pub fn mul_mod_u64(x: u64, y: u64, m: u64) -> u64 {
    mul_mod(x, y, m)
}

impl IsPseudoPrime for u32 {
    fn is_pseudo_prime(&self, b: u32) -> bool {
        let n = *self;
        // know self > 2 and is odd, and b % self != 0
        let mut d = n - 1;
        let mut s = 1;

        while d % 2 == 0 {
            d /= 2;
            s += 1;
        }

        let mut bd = powmod(b, d, n);
        if bd == 1 {
            return true;
        }
        for _ in 0 .. s {
            if bd + 1 == n {
                return true;
            } else {
                bd = mul_mod(bd, bd, n);
            }
        }

        false
    }
}

impl IsPseudoPrime for u64 {
    fn is_pseudo_prime(&self, b: u64) -> bool {
        // know self > 2 and is odd, and b % self != 0
        let n = *self;
        let mut d = n - 1;
        let mut s = 1;

        while d % 2 == 0 {
            d /= 2;
            s += 1;
        }

        let mut bd = powmod(b, d, n);
        if bd == 1 {
            return true;
        }
        for _ in 0 .. s {
            if bd + 1 == n {
                return true;
            } else {
                bd = mul_mod(bd, bd, n);
            }
        }

        false
    }
}

pub trait IsPseudoPrime {
    // PRE: self > 2, self % 2 == 1
    fn is_pseudo_prime(&self, b: Self) -> bool;
}

pub trait IsPrime {
    fn is_prime(&self) -> bool;
}

impl IsPrime for u32 {
    fn is_prime(&self) -> bool {
        if *self < 2 {
            false
        } else if *self % 2 == 0 {
            *self == 2
        } else if *self % 3 == 0 {
            *self == 3
        } else if *self % 5 == 0 {
            *self == 5
        } else if *self % 7 == 0 {
            *self == 7
        } else {
            // Theorem [Jaeschke, Sinclair]: This works
            [2, 7, 61].into_iter()
                .all(|&b| b % *self == 0 || self.is_pseudo_prime(b))
        }
    }
}

impl IsPrime for u64 {
    fn is_prime(&self) -> bool {
        if *self < 2 {
            false
        } else if *self % 2 == 0 {
            *self == 2
        } else if *self % 3 == 0 {
            *self == 3
        } else if *self % 5 == 0 {
            *self == 5
        } else if *self % 7 == 0 {
            *self == 7
        } else {
            // Theorem [Jaeschke, Sinclair]: This works
            [2, 325, 9375, 28178, 450775, 9780504, 1795265022].into_iter()
                .all(|&b| b % *self == 0 || self.is_pseudo_prime(b))
        }
    }
}

///
/// Returns (x-y) % m in a safe manner
///
pub fn mod_sub(x: u64, y: u64, m: u64) -> u64 {
    if x >= y {
        x - y
    } else {
        m - ((y - x) % m)
    }
}

pub fn mod_inv(x: u64, m: u64) -> u64 {
    // Uses the extended Euclidean algorithm; if ax+bm = 1 then a=inv_m(x)
    let mut r_last = x;
    let mut r_curr = m;

    let mut s_last = 1;
    let mut s_curr = 0;

    while r_curr != 0 {
        let q = r_last / r_curr;
        let r_next = r_last - q * r_curr;
        let s_next = mod_sub(s_last, q * s_curr, m);

        r_last = r_curr;
        r_curr = r_next;

        s_last = s_curr;
        s_curr = s_next;
    }

    s_last
}

#[cfg(test)]
mod tests {
    use std;
    use super::*;

    #[test]
    fn mod_sub_test() {
        assert_eq!(mod_sub(1, 2, 5), 4);
        assert_eq!(mod_sub(12, 9, 41), 3);
        assert_eq!(mod_sub(14213, 51331, 135315), 98197);
    }

    #[test]
    fn mod_inv_test() {
        for m in vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 61, 173] {
            for x in 1 .. m {
                let x_inv = mod_inv(x, m);
                assert_eq!(mul_mod(x, x_inv, m), 1);
            }
        }
    }

    fn to_big_int<'a, T: Iterator<Item=&'a i64>>(it: T) -> Vec<BigInt> {
        it.map(|n: &i64| BigInt::from(*n)).collect()
    }

    #[test]
    fn num_div_test() {
        let test_cases = vec![
            (1, 1), (2, 2), (3, 2), (4, 3), (5, 2),
            (6, 4), (7, 2), (8, 4), (9, 3), (10, 4),
            (37 * 37, 3)
        ];

        for (n, nd) in test_cases {
            let n = BigInt::from(n);
            let nd = BigInt::from(nd);

            assert_eq!(num_divisors(n), nd);
        }
    }

    #[test]
    fn pprimes5() {
        let actual: Vec<_> = possible_primes().take(5).collect();
        let expected: Vec<_> = to_big_int(vec![2, 3, 5, 7, 11].iter());

        assert_eq!(actual, expected, "Expected the first 5 possible primes");
    }

    #[test]
    fn pprimes10() {
        let actual: Vec<_> = possible_primes().take(10).collect();
        let expected: Vec<_> = to_big_int(vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 25].iter());

        assert_eq!(actual, expected, "Expected the first 10 possible primes");
    }

    #[test]
    fn biggest_factor_test() {
        let actual = largest_prime_factor(&BigInt::from(32));
        let expected = BigInt::from(2);

        assert_eq!(actual, expected);

        let actual = largest_prime_factor(&BigInt::from(145));
        let expected = BigInt::from(29);

        assert_eq!(actual, expected);

        let actual = largest_prime_factor(&BigInt::from(37));
        let expected = BigInt::from(37);

        assert_eq!(actual, expected);

        let actual = largest_prime_factor(&BigInt::from(-32));
        let expected = BigInt::from(2);

        assert_eq!(actual, expected);

        let actual = largest_prime_factor(&BigInt::from(0));
        let expected = BigInt::from(1);

        assert_eq!(actual, expected);
    }

    #[test]
    fn is_prime_test() {
        let checker = |n: i64, val: bool| assert_eq!(is_prime(&BigInt::from(n)), val);

        checker(-1, false);
        checker(0, false);
        checker(1, false);
        checker(2, true);
        checker(3, true);
        checker(4, false);
        checker(5, true);

        checker(31, true);
        checker(39, false);
    }

    #[test]
    fn is_prime_u32_test() {
        let cap = 65536;
        let expected = all_primes(cap);
        let actual = (0 .. cap as u32)
            .filter(|&n| n.is_prime())
            .map(|n| n as usize) // for comparison
            .collect::<Vec<usize>>();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_mul_mod_u32() {
        let half = 1 << 16;
        let max = std::u32::MAX;

        assert_eq!(mul_mod_u32(0, 0, 2), 0);
        assert_eq!(mul_mod_u32(1, 0, 2), 0);
        assert_eq!(mul_mod_u32(0, 1, 2), 0);
        assert_eq!(mul_mod_u32(1, 1, 2), 1);
        assert_eq!(mul_mod_u32(42, 1, 2), 0);
        assert_eq!(mul_mod_u32(1, 42, 2), 0);
        assert_eq!(mul_mod_u32(42, 42, 2), 0);
        assert_eq!(mul_mod_u32(42, 42, 42), 0);
        assert_eq!(mul_mod_u32(42, 42, 41), 1);
        assert_eq!(mul_mod_u32(1239876, 2948635, 234897), 163320);

        assert_eq!(mul_mod_u32(1239876, 2948635, half), 18476);
        assert_eq!(mul_mod_u32(half, half, half), 0);
        assert_eq!(mul_mod_u32(half+1, half+1, half), 1);

        assert_eq!(mul_mod_u32(max, max, max), 0);
        assert_eq!(mul_mod_u32(1239876, 2948635, max), 924601215);
        assert_eq!(mul_mod_u32(1239876, max, max), 0);
        assert_eq!(mul_mod_u32(1239876, max-1, max), max-1239876);
        assert_eq!(mul_mod_u32(max, 2948635, max), 0);
        assert_eq!(mul_mod_u32(max-1, 2948635, max), max-2948635);
        assert_eq!(mul_mod_u32(max-1, max-1, max), 1);
        assert_eq!(mul_mod_u32(2, max/2, max-1), 0);
    }

    #[test]
    fn test_mul_mod_u64() {
        let half = 1 << 16;
        let max = std::u64::MAX;

        assert_eq!(mul_mod_u64(0, 0, 2), 0);
        assert_eq!(mul_mod_u64(1, 0, 2), 0);
        assert_eq!(mul_mod_u64(0, 1, 2), 0);
        assert_eq!(mul_mod_u64(1, 1, 2), 1);
        assert_eq!(mul_mod_u64(42, 1, 2), 0);
        assert_eq!(mul_mod_u64(1, 42, 2), 0);
        assert_eq!(mul_mod_u64(42, 42, 2), 0);
        assert_eq!(mul_mod_u64(42, 42, 42), 0);
        assert_eq!(mul_mod_u64(42, 42, 41), 1);
        assert_eq!(mul_mod_u64(1239876, 2948635, 234897), 163320);

        assert_eq!(mul_mod_u64(1239876, 2948635, half), 18476);
        assert_eq!(mul_mod_u64(half, half, half), 0);
        assert_eq!(mul_mod_u64(half+1, half+1, half), 1);

        assert_eq!(mul_mod_u64(max, max, max), 0);
        assert_eq!(mul_mod_u64(1239876, 2948635, max), 3655941769260);
        assert_eq!(mul_mod_u64(1239876, max, max), 0);
        assert_eq!(mul_mod_u64(1239876, max-1, max), max-1239876);
        assert_eq!(mul_mod_u64(max, 2948635, max), 0);
        assert_eq!(mul_mod_u64(max-1, 2948635, max), max-2948635);
        assert_eq!(mul_mod_u64(max-1, max-1, max), 1);
        assert_eq!(mul_mod_u64(2, max/2, max-1), 0);
    }
}
