use num::pow::{pow};

use euler_lib::numerics::{all_primes, powmod, MulMod};

pub fn p381() -> String {

    #[inline]
    fn sum_fact(p: usize) -> usize {
        let two_inv = (p+1)/2;
        let two_inv_cubed = powmod(two_inv, 3, p);
        return p - (3.mul_mod(&two_inv_cubed, &p));
    }

    let cap = pow(10, 8); // <cap

    let primes = all_primes(cap);

    let mut sum = 4; // sum_fact(5) is 4

    for p in primes.into_iter().filter(|&p| p > 5) {
        let sf = sum_fact(p);
        //println!("For {} got {}", p, sf);
        sum += sf;
    }

    sum.to_string()
}