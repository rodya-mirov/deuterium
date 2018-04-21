use std::collections::{HashMap};
use euler_lib::numerics::{powmod};

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