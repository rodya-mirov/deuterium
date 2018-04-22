use std::collections::{HashMap};
use euler_lib::numerics::{powmod};
use num::bigint::{BigUint};
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