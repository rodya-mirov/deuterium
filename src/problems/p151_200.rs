use std::collections::{HashMap};

use euler_lib::numerics;
use euler_lib::numerics::{powmod};

use num::bigint::{BigUint, BigInt};
use num::{pow, Zero, One};



pub fn p164() -> String {
    fn count_sum(digits_remaining: u8, left_two: u8, left_one: u8, cache: &mut HashMap<(u8, u8, u8), BigUint>) -> BigUint {
        let leading_sum = left_two + left_one;

        if leading_sum > 9 {
            return BigUint::zero();
        } else if digits_remaining == 0 {
            return BigUint::one();
        }

        let key = (digits_remaining, left_two, left_one);
        if let Some(val) = cache.get(&key) {
            return val.clone();
        }

        let mut total = BigUint::zero();

        for allowed_digit in 0 .. 10 - leading_sum {
            total += count_sum(digits_remaining - 1, left_one, allowed_digit, cache);
        }

        cache.insert(key, total.clone());
        total
    }

    let mut cache = HashMap::new();

    let full_digits = 20;

    let inclusive = count_sum(full_digits, 0, 0, &mut cache);
    let less = count_sum(full_digits-1, 0, 0, &mut cache); // eliminates "leading zeroes" from 'inclusive'

    // known: with 3 digits, get 220, 165
    // known: with 2 digits, get  55,  45

    (inclusive - less).to_string()
}



pub fn p169() -> String {

    fn num_sum_powers(n: &BigUint, greatest_pow: &BigUint, cache: &mut HashMap<(BigUint, BigUint), BigUint>) -> BigUint {

        let key = (n.clone(), greatest_pow.clone());

        if let Some(val) = cache.get(&key) {
            return val.clone();
        }

        let l_shift = greatest_pow << 1;
        let r_shift = greatest_pow >> 1;
        let max_bound = &l_shift << 1;

        if n.is_zero() {
            return BigUint::one();
        } else if greatest_pow > n {
            return num_sum_powers(n, &r_shift, cache);
        } else if n >= &max_bound {
            return BigUint::zero();
        }

        let mut val = num_sum_powers(n, &r_shift, cache) + num_sum_powers(&(n - greatest_pow), &r_shift, cache);
        if n >= &l_shift {
            val += num_sum_powers(&(n - l_shift), &r_shift, cache);
        }

        cache.insert(key, val.clone());
        return val;
    }

    let n = pow(BigUint::from(10_u32), 25);
    let two_pow = pow(BigUint::from(2_u32), 100);
    let mut cache = HashMap::new();

    return num_sum_powers(&n, &two_pow, &mut cache)
        .to_string();
}



pub fn p173() -> String {
    // tbh i'm a bit disappointed this worked (1.5 ms with --release)
    // there are a lot more optimizations that could/should be made, like
    // figuring out COUNT directly from outer_width (this is technically quadratic)
    let max_squares: u64 = 1_000_000;

    let mut count = 0;
    let mut outer_width = 3;
    
    loop {
        let mut hole_width = outer_width - 2;
        let start_area = outer_width * outer_width - hole_width * hole_width;

        if start_area > max_squares {
            break;
        }

        count += 1;
        hole_width -= 2;

        while hole_width > 0 {
            let area = outer_width * outer_width - hole_width * hole_width;
            if area > max_squares {
                break;
            }

            count += 1;
            hole_width -= 2;
        }

        outer_width += 1;
    }

    count.to_string()
}

pub fn p174() -> String {
    let max_squares: i64 = 1_000_000;
    let mut ways = HashMap::new();

    let mut outer_width = 3;
    
    loop {
        let mut hole_width = outer_width - 2;
        let start_area = outer_width * outer_width - hole_width * hole_width;

        if start_area > max_squares {
            break;
        }

        *ways.entry(start_area).or_insert(0) += 1;
        hole_width -= 2;

        while hole_width > 0 {
            let area = outer_width * outer_width - hole_width * hole_width;
            if area > max_squares {
                break;
            }

            *ways.entry(area).or_insert(0) += 1;
            hole_width -= 2;
        }

        outer_width += 1;
    }

    let mut count = 0;
    for (ref _area, ref area_count) in ways {
        if *area_count <= 10 {
            count += 1;
        }
    }
    
    count.to_string()
}



pub fn p181() -> String {
    use std;

    #[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
    struct State {
        num_black: u8,
        num_white: u8
    }
    
    impl State {
        pub fn minned_with(&self, other: &State) -> State {
            // lexicographic ordering; black bigger than white
            if self.num_black > other.num_black {
                State {
                    num_black: other.num_black,
                    num_white: other.num_white
                }
            } else if self.num_white > other.num_white {
                State {
                    num_black: self.num_black,
                    num_white: other.num_white
                }
            } else {
                State {
                    num_black: self.num_black,
                    num_white: self.num_white
                }
            }
        }

        pub fn stepped_down(&self) -> State {
            if self.num_white > 0 {
                State {
                    num_black: self.num_black,
                    num_white: self.num_white - 1
                }
            } else if self.num_black > 0 {
                State {
                    num_black: self.num_black - 1,
                    num_white: std::u8::MAX
                }
            } else {
                panic!();
            }
        }

        pub fn reduced_by(&self, other: &State) -> State {
            State {
                num_black: self.num_black - other.num_black,
                num_white: self.num_white - other.num_white
            }
        }

        pub fn is_zero(&self) -> bool {
            self.num_black == 0 && self.num_white == 0
        }
    }

    type Cache = HashMap<(State, State), BigUint>;

    fn num_perms(state: State, max_step: State, cache: &mut Cache) -> BigUint {
        let max_step = max_step.minned_with(&state);

        if state.is_zero() {
            return BigUint::one();
        } else if max_step.is_zero() {
            return BigUint::zero();
        }

        let key = (state, max_step);
        if let Some(cached) = cache.get(&key) {
            return cached.clone();
        }

        let step_down = num_perms(state, max_step.stepped_down(), cache);
        let reduced = num_perms(state.reduced_by(&max_step), max_step, cache);

        let val = step_down + reduced;
        cache.insert(key, val.clone());

        val
    }

    let num_black = 60;
    let num_white = 40;

    let state = State { num_black, num_white };
    let mut cache = HashMap::new();

    num_perms(state, state, &mut cache).to_string()
}



pub fn p188() -> String {
    let a: u64 = 1777;
    let mut b: u64 = 1855;
    let modulus: u64 = 100_000_000; // 10 ** 8

    let mut running = a;

    while b > 1 {
        running = powmod(a, running, modulus);
        b -= 1;
    }

    running.to_string()
}



pub fn p193() -> String {
    let cap: usize = 1 << 20; //1 << 50;
    let root_cap: usize = 1 << 11; //1 << 25; // sqrt of cap

    // fact: 2 * 3 * 5 * ... * 19 is the largest primorial whose square is below cap (found by hand)
    // this is 8 primes
    let max_length = 9;

    let primes = numerics::all_primes(root_cap);

    // total size of all intersections of sets of the form:
    //      1) "things from 1 to cap, inclusive", where
    //      2) the "thing" is divisible by a^2b^2...k^2, where
    //      3) (a, b, c, ..., k) is an ascending subsequence of asc of length "length"
    fn sum_intersections(length: usize, asc: &[usize], cap: usize) -> BigInt {
        fn total_helper(primes: &[usize], stuff_so_far: &mut Vec<usize>, desired_length: usize, next_index: usize, running_sq_product: usize, cap: usize, running_total: &mut BigInt) {
            if stuff_so_far.len() == desired_length {
                *running_total = &*running_total + &BigInt::from(cap / running_sq_product);
                return;
            } else if next_index >= primes.len() {
                return;
            }

            for actual_next_index in next_index .. primes.len() {
                let next_prime = primes[actual_next_index];
                let next_running_sq_product = running_sq_product * next_prime * next_prime;

                if next_running_sq_product > cap {
                    break;
                }

                // add all the things that involve the next prime
                stuff_so_far.push(next_prime);
                total_helper(primes, stuff_so_far, desired_length, actual_next_index+1, next_running_sq_product, cap, running_total);
                stuff_so_far.pop();
            }
        }

        let mut total = BigInt::zero();
        total_helper(asc, &mut Vec::new(), length, 0, 1, cap, &mut total);

        total
    }

    let mut num_square_free = BigInt::from(cap); // 1 .. cap

    let mut parity: i64 = 1;
    for length in 1 .. max_length+1 {
        parity *= -1;
        let curr = sum_intersections(length, &primes, cap);

        if parity < 0 {
            num_square_free = num_square_free - curr;
        } else {
            num_square_free = num_square_free + curr;
        }
    }

    num_square_free.to_string()
}