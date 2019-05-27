use std::collections::{HashMap, HashSet};

use euler_lib::numerics::powmod;

use num::bigint::BigUint;
use num::rational::Ratio;
use num::{pow, One, Zero};

pub fn p164() -> String {
    fn count_sum(
        digits_remaining: u8,
        left_two: u8,
        left_one: u8,
        cache: &mut HashMap<(u8, u8, u8), BigUint>,
    ) -> BigUint {
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

        for allowed_digit in 0..10 - leading_sum {
            total += count_sum(digits_remaining - 1, left_one, allowed_digit, cache);
        }

        cache.insert(key, total.clone());
        total
    }

    let mut cache = HashMap::new();

    let full_digits = 20;

    let inclusive = count_sum(full_digits, 0, 0, &mut cache);
    let less = count_sum(full_digits - 1, 0, 0, &mut cache); // eliminates "leading zeroes" from 'inclusive'

    // known: with 3 digits, get 220, 165
    // known: with 2 digits, get  55,  45

    (inclusive - less).to_string()
}

pub fn p165() -> String {
    use std::fmt;
    use std::fmt::Display;

    type Rational = Ratio<i64>;

    #[derive(Hash, Eq, PartialEq, Copy, Clone, Debug)]
    struct RPoint {
        x: Rational,
        y: Rational,
    }

    #[derive(Hash, Eq, PartialEq, Copy, Clone, Debug)]
    struct LineSegment {
        x1: Rational,
        y1: Rational,
        x2: Rational,
        y2: Rational,
    }

    impl Display for LineSegment {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "LineSegment: ({}, {}) to ({}, {})",
                self.x1, self.y1, self.x2, self.y2
            )
        }
    }

    fn between(a: &Rational, b: &Rational, mid: &Rational) -> bool {
        if a < b {
            a < mid && mid < b
        } else if a > b {
            b < mid && mid < a
        } else {
            mid == a
        }
    }

    impl LineSegment {
        fn contains(&self, p: &RPoint) -> bool {
            between(&self.x1, &self.x2, &p.x) && between(&self.y1, &self.y2, &p.y)
        }

        fn slope(&self) -> Rational {
            (self.y2 - self.y1) / (self.x2 - self.x1)
        }
    }

    fn true_intersection(a: &LineSegment, b: &LineSegment) -> Option<RPoint> {
        if a.x1 == a.x2 {
            let x = a.x1;
            if b.x1 == b.x2 {
                return None;
            }

            let mb = b.slope();
            let y = mb * (x - b.x1) + b.y1;
            let p = RPoint { x, y };

            if a.contains(&p) && b.contains(&p) {
                return Some(p);
            } else {
                return None;
            }
        } else if b.x1 == b.x2 {
            return true_intersection(b, a);
        }

        let ma = a.slope();
        let mb = b.slope();

        if ma == mb {
            return None;
        }

        let x_numer = b.y1 - a.y1 + (ma * a.x1) - (mb * b.x1);
        let x_denom = ma - mb;

        let x = x_numer / x_denom;
        let y = ma * (x - a.x1) + a.y1;
        let also_y = mb * (x - b.x1) + b.y1;

        if y != also_y {
            panic!()
        }

        let p = RPoint { x, y };

        if a.contains(&p) && b.contains(&p) {
            Some(p)
        } else {
            None
        }
    }

    struct BBS {
        s: u64,
    }

    impl BBS {
        fn next(&mut self) -> u64 {
            self.s = (self.s * self.s) % 50515093;
            self.s % 500
        }

        fn next_rational(&mut self) -> Rational {
            Rational::from_integer(self.next() as i64)
        }

        fn next_seg(&mut self) -> LineSegment {
            LineSegment {
                x1: self.next_rational(),
                y1: self.next_rational(),
                x2: self.next_rational(),
                y2: self.next_rational(),
            }
        }
    }

    let mut bbs = BBS { s: 290797 };

    let mut lines = Vec::new();
    let mut found = HashSet::new();
    let mut count = 0;

    for i in 0..5000 {
        let next = bbs.next_seg();
        // println!("Line {}: {:?}", i+1, next);

        for j in 0..i {
            let prev = lines[j];
            if let Some(p) = true_intersection(&prev, &next) {
                found.insert(p);
                count += 1;
            }
        }

        lines.push(next);
    }

    println!("Raw: {}, deduped: {}", count, found.len());

    found.len().to_string()
}

pub fn p169() -> String {
    fn num_sum_powers(
        n: &BigUint,
        greatest_pow: &BigUint,
        cache: &mut HashMap<(BigUint, BigUint), BigUint>,
    ) -> BigUint {
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

        let mut val = num_sum_powers(n, &r_shift, cache)
            + num_sum_powers(&(n - greatest_pow), &r_shift, cache);
        if n >= &l_shift {
            val += num_sum_powers(&(n - l_shift), &r_shift, cache);
        }

        cache.insert(key, val.clone());
        return val;
    }

    let n = pow(BigUint::from(10_u32), 25);
    let two_pow = pow(BigUint::from(2_u32), 100);
    let mut cache = HashMap::new();

    return num_sum_powers(&n, &two_pow, &mut cache).to_string();
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
    #[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
    struct State {
        num_black: u8,
        num_white: u8,
    }

    impl State {
        pub fn minned_with(&self, other: &State) -> State {
            // lexicographic ordering; black bigger than white
            if self.num_black > other.num_black {
                State {
                    num_black: other.num_black,
                    num_white: other.num_white,
                }
            } else if self.num_white > other.num_white {
                State {
                    num_black: self.num_black,
                    num_white: other.num_white,
                }
            } else {
                State {
                    num_black: self.num_black,
                    num_white: self.num_white,
                }
            }
        }

        pub fn stepped_down(&self) -> State {
            if self.num_white > 0 {
                State {
                    num_black: self.num_black,
                    num_white: self.num_white - 1,
                }
            } else if self.num_black > 0 {
                State {
                    num_black: self.num_black - 1,
                    num_white: std::u8::MAX,
                }
            } else {
                panic!();
            }
        }

        pub fn reduced_by(&self, other: &State) -> State {
            State {
                num_black: self.num_black - other.num_black,
                num_white: self.num_white - other.num_white,
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

    let state = State {
        num_black,
        num_white,
    };
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

pub fn p191() -> String {
    #[derive(Copy, Clone, Hash, Eq, PartialEq)]
    enum Day {
        O,
        A,
        L,
    }

    #[derive(Copy, Clone, Hash, Eq, PartialEq)]
    struct Status {
        days_remaining: usize,
        late_so_far: usize,
        prev: Day,
        prev_prev: Day,
    }

    impl Status {
        fn next(&self, day: Day) -> Status {
            Status {
                days_remaining: self.days_remaining - 1,
                late_so_far: self.late_so_far + (if day == Day::L { 1 } else { 0 }),
                prev: day,
                prev_prev: self.prev,
            }
        }
    }

    type Cache = HashMap<Status, BigUint>;

    fn num_strings(status: Status, cache: &mut Cache) -> BigUint {
        if status.days_remaining == 0 {
            return BigUint::one();
        }

        if let Some(val) = cache.get(&status) {
            return val.clone();
        }

        let mut total = num_strings(status.next(Day::O), cache);
        if status.late_so_far == 0 {
            total = total + num_strings(status.next(Day::L), cache);
        }
        if status.prev != Day::A || status.prev_prev != Day::A {
            total = total + num_strings(status.next(Day::A), cache);
        }

        cache.insert(status, total.clone());
        total
    }

    let status = Status {
        days_remaining: 30,
        late_so_far: 0,
        prev: Day::O,
        prev_prev: Day::O,
    };

    num_strings(status, &mut Cache::new()).to_string()
}
