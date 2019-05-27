use std::collections::HashSet;

use euler_lib::numerics::gcd;

pub fn p504() -> String {
    let m: i64 = 100;

    let mut last_nsq = 0;
    let mut last_n = 0;
    let mut squares = HashSet::new();
    squares.insert(last_nsq);

    let mut is_square = |v: i64| {
        while v > last_nsq {
            last_n += 1;
            last_nsq = last_n * last_n;
            squares.insert(last_nsq);
        }

        squares.contains(&v)
    };

    let mut counter = 0;
    for a in 1..m + 1 {
        for b in 1..m + 1 {
            let gcd_ab = gcd(a, b);

            for c in 1..m + 1 {
                let gcd_bc = gcd(b, c);

                for d in 1..m + 1 {
                    let area = (a * b + b * c + c * d + d * a) / 2;
                    let num_bdry = gcd_ab + gcd_bc + gcd(c, d) + gcd(d, a);
                    let num_int = area + 1 - (num_bdry / 2);

                    if is_square(num_int) {
                        counter += 1;
                    }
                }
            }
        }
    }

    return counter.to_string();
}

pub fn p523() -> String {
    let n = 30;

    fn exp(n: u64) -> f64 {
        if n <= 1 {
            return 0.0;
        }

        let swap_back = (1 << (n - 1)) - 1;

        (swap_back as f64) / (n as f64) + exp(n - 1)
    }

    exp(n).to_string()
}
