use std::cmp::{min, Ordering};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::Read;

use num::bigint::{BigInt, BigUint};
use num::integer::{gcd, lcm};
use num::pow::pow;

use euler_lib::data::RectVec;
use euler_lib::numerics::{mod_inv, powmod, IsPrime};

use im::ordset::OrdSet;

pub fn p102() -> String {
    // read the file somehow
    struct Pt {
        x: i32,
        y: i32,
    };

    struct Triangle {
        a: Pt,
        b: Pt,
        c: Pt,
    };

    let origin = Pt { x: 0, y: 0 };

    fn same_side(a: &Pt, b: &Pt, c: &Pt, p: &Pt) -> bool {
        let n_1 = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        if n_1 == 0 {
            return true;
        }

        let n_2 = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if n_1 >= 0 {
            n_2 >= 0
        } else {
            n_2 <= 0
        }
    }

    fn contains(t: &Triangle, p: &Pt) -> bool {
        same_side(&t.a, &t.b, &t.c, p)
            && same_side(&t.a, &t.c, &t.b, p)
            && same_side(&t.b, &t.c, &t.a, p)
    }

    let mut text = String::new();

    File::open("resources/p102.txt")
        .expect("IO Error?")
        .read_to_string(&mut text)
        .expect("IO Error?");

    let mut good_triangles: u32 = 0;

    println!("Num lines: {}", text.lines().count());

    for line in text.lines() {
        let tokens = line
            .split(",")
            .flat_map(|x| x.parse::<i32>().into_iter())
            .collect::<Vec<i32>>();

        if tokens.len() != 6 {
            panic!();
        }

        let tri = Triangle {
            a: Pt {
                x: tokens[0],
                y: tokens[1],
            },
            b: Pt {
                x: tokens[2],
                y: tokens[3],
            },
            c: Pt {
                x: tokens[4],
                y: tokens[5],
            },
        };

        if contains(&tri, &origin) {
            good_triangles += 1;
        }
    }

    good_triangles.to_string()
}

pub fn p103() -> String {
    let mut upper_bound: i64 = 350; // optimum is strictly below this; modify as improvements are found
    let mut set_str = "This didn't work".to_string();

    fn set_distinct(a1: i64, a2: i64, a3: i64, a4: i64, a5: i64, a6: i64, a7: i64) -> bool {
        let flag_sum = |flag: u8| {
            let mut out = 0;
            if flag & (1 << 0) != 0 {
                out += a1;
            }
            if flag & (1 << 1) != 0 {
                out += a2;
            }
            if flag & (1 << 2) != 0 {
                out += a3;
            }
            if flag & (1 << 3) != 0 {
                out += a4;
            }
            if flag & (1 << 4) != 0 {
                out += a5;
            }
            if flag & (1 << 5) != 0 {
                out += a6;
            }
            if flag & (1 << 6) != 0 {
                out += a7;
            }
            out
        };

        let mut seen = HashSet::new();

        for flag in 0..(1 << 7) {
            if !seen.insert(flag_sum(flag)) {
                return false;
            }
        }

        true
    }

    for a1 in 1..upper_bound / 7 + 1 {
        for a2 in a1 + 1..upper_bound / 6 + 1 {
            // a7 < a1+a2
            for a7 in a2 + 5..min(upper_bound + 1, a1 + a2) {
                // a3 <= a7-4
                for a3 in a2 + 1..min(a7 - 3, upper_bound / 5 + 1) {
                    // a6 < a1+a2+a3-a7
                    for a6 in a3 + 3..min(a7, a1 + a2 + a3 - a7) {
                        // a4 < a6-2
                        for a4 in a3 + 1..min(a6 - 1, upper_bound / 3 + 1) {
                            // a5 < a1+a2+a3+a4-a7-a6
                            for a5 in a4 + 1..min(a6 - 1, a1 + a2 + a3 + a4 - a7 - a6) {
                                let total = a1 + a2 + a3 + a4 + a5 + a6 + a7;
                                if total >= upper_bound {
                                    break;
                                }
                                if set_distinct(a1, a2, a3, a4, a5, a6, a7) {
                                    upper_bound = total;
                                    set_str = format!("{}{}{}{}{}{}{}", a1, a2, a3, a4, a5, a6, a7);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    set_str
}

pub fn p104() -> String {
    use num::{Float, ToPrimitive};

    /// Determines if all the digits 1-9 appear in n
    fn is_pan(mut n: u64) -> bool {
        let mut seen = [false; 10];

        while n > 0 {
            seen[(n % 10) as usize] = true;
            n /= 10;
        }

        seen.into_iter().skip(1).all(|&n| n)
    }

    let ten_pow = pow(10, 9);
    let ten_pow_float = pow(10.0, 9);

    let phi = (1.0 + (5.0).sqrt()) / 2.0;
    let mut approx = phi / ((5.0).sqrt());

    let mut k = 1;
    let mut curr = 1;
    let mut prev = 0;

    loop {
        k += 1;

        let next = (prev + curr) % ten_pow;
        prev = curr;
        curr = next;

        approx *= phi;
        if approx > ten_pow_float {
            approx /= 10.0;
        }

        if is_pan(next) {
            if is_pan(approx.to_u64().unwrap()) {
                return k.to_string();
            }
        }
    }
}

pub fn p105() -> String {
    fn check_monotone(v: &Vec<u64>) -> bool {
        // PRE: v is sorted
        let n = v.len();
        // compare sum(first k+1 elts) > sum(last k elts)
        let k = if n % 2 == 1 { n / 2 } else { n / 2 - 1 };

        let leading = (0..k + 1).map(|i| v[i]).sum::<u64>();
        let trailing = ((n - k)..n).map(|i| v[i]).sum::<u64>();
        leading > trailing
    }

    // running time is k 2^k where k=len(v); good thing len(v) <= 12
    fn check_distinct(v: &Vec<u64>) -> bool {
        let mut seen_totals = HashSet::new();

        for flag in 0..(1 << v.len()) {
            let mut total = 0;
            for bit in 0..v.len() {
                if flag & (1 << bit) != 0 {
                    total += v[bit];
                }
            }
            if !seen_totals.insert(total) {
                return false;
            }
        }

        true
    }

    let mut text = String::new();
    File::open("resources/p105.txt")
        .expect("IO error?")
        .read_to_string(&mut text)
        .expect("IO error?");

    let mut monotone_count = 0;
    let mut distinct_count = 0;

    let sets: Vec<Vec<u64>> = text
        .lines()
        .map(|line| {
            line.split(",")
                .map(|token| token.parse::<u64>().unwrap())
                .collect::<Vec<u64>>()
        })
        .map(|mut v| {
            v.sort();
            v
        })
        .filter(|v| check_monotone(v))
        .map(|v| {
            monotone_count += 1;
            v
        })
        .filter(|v| check_distinct(v))
        .map(|v| {
            distinct_count += 1;
            v
        })
        .collect();

    println!("Got {} through the monotone filter", monotone_count);
    println!("Got {} through the distinct filter", distinct_count);

    sets.into_iter()
        .map(|v| v.into_iter().sum::<u64>())
        .sum::<u64>()
        .to_string()
}

pub fn p106() -> String {
    let n = 12;

    fn bits_of(flag: u64) -> Vec<u8> {
        let mut out = Vec::new();

        let mut bit = 0;
        while (1 << bit) <= flag {
            if (1 << bit) & flag != 0 {
                out.push(bit);
            }
            bit += 1;
        }
        out
    }

    // returns true iff the first nonzero bit of flag1 is < the first nonzero bit of flag2
    // and likewise with the second, etc.
    // PRE: flag1.count_ones() == flag2.count_ones()
    fn bit_less(flag1: u64, flag2: u64) -> bool {
        let bits1 = bits_of(flag1);
        let bits2 = bits_of(flag2);

        for i in 0..bits1.len() {
            if bits1[i] >= bits2[i] {
                return false;
            }
        }
        true
    }

    fn num_distinct_subsets(n: u64) -> u64 {
        let mut count = 0;
        for flag1 in 1_u64..(1 << n) {
            for flag2 in 1_u64..flag1 {
                // check for disjointness of the subsets
                if flag1 & flag2 != 0 {
                    continue;
                }
                // check for same size of the subsets
                if flag1.count_ones() != flag2.count_ones() {
                    continue;
                }

                // ignore empty sets
                if flag1.count_ones() == 0 {
                    continue;
                }

                // make sure they're not strictly in order
                if bit_less(flag1, flag2) || bit_less(flag2, flag1) {
                    continue;
                }

                count += 1;
            }
        }
        count
    }

    num_distinct_subsets(n).to_string()
}

pub fn p107() -> String {
    #[derive(Eq, PartialEq)]
    struct Edge {
        start: usize,
        end: usize,
        weight: u64,
    }

    impl Ord for Edge {
        // reversed order, to turn the max heap into a min heap
        fn cmp(&self, other: &Edge) -> Ordering {
            other.weight.cmp(&self.weight)
        }
    }

    impl PartialOrd for Edge {
        fn partial_cmp(&self, other: &Edge) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let (mut edges, original_total_weight) = {
        let mut text = String::new();

        File::open("resources/p107.txt")
            .expect("IO Error?")
            .read_to_string(&mut text)
            .expect("IO Error?");

        let mut rows = Vec::with_capacity(40);

        for line in text.lines() {
            let mut row = Vec::with_capacity(40);

            for token in line.split(",") {
                let maybe_weight = match token.parse::<u64>() {
                    Ok(w) => Some(w),
                    Err(_) => None,
                };
                row.push(maybe_weight);
            }

            rows.push(row);
        }

        let mut total = 0;
        let mut edges = BinaryHeap::new();
        for i in 0..40 {
            for j in 0..i {
                if let Some(weight) = rows[i][j] {
                    edges.push(Edge {
                        start: i,
                        end: j,
                        weight,
                    });
                    total += weight;
                }
            }
        }

        (edges, total)
    };

    // then we repeatedly add the smallest edge available that improves connectivity
    // note it can be "final" because of interior mutability (ugh)
    let mut components = (0..40)
        .map(|i| {
            let mut set = HashSet::new();
            set.insert(i);
            (i, set)
        })
        .collect::<HashMap<usize, HashSet<usize>>>();

    let mut first_member = (0..40).map(|i| (i, i)).collect::<HashMap<usize, usize>>();

    let mut total_weight = 0;

    // edges is a minheap so this is greedy -- always add the smallest edge that improves connectivity
    while !edges.is_empty() {
        let edge = edges.pop().unwrap();
        let i = first_member[&edge.start];
        let j = first_member[&edge.end];

        if components[&i].contains(&j) {
            continue;
        }

        // otherwise it's an improvement!
        let i_comp = components.remove(&i).unwrap();
        for &k in i_comp.iter() {
            first_member.insert(k, j);
            components.get_mut(&j).unwrap().insert(k);
        }

        total_weight += edge.weight;
    }

    (original_total_weight - total_weight).to_string()
}

pub fn p108() -> String {
    struct PossPrimes {
        next: u64,
        step: u64,
    }

    impl PossPrimes {
        fn new() -> PossPrimes {
            PossPrimes { next: 5, step: 2 }
        }
    }

    impl Iterator for PossPrimes {
        type Item = u64;

        fn next(&mut self) -> Option<Self::Item> {
            let out = self.next;
            self.next += self.step;
            self.step = 6 - self.step;
            Some(out)
        }
    }

    fn num_sol(mut n: u64) -> u64 {
        let mut num_sol = 1;

        for p in vec![2, 3].into_iter().chain(PossPrimes::new()) {
            let mut ppow = 0;

            while n % p == 0 {
                n /= p;
                ppow += 1;
            }

            num_sol *= 2 * ppow + 1;

            if p * p > n {
                if n > 1 {
                    // then the rest is prime, so ...
                    num_sol *= 3;
                }
                break;
            }
        }

        (num_sol + 1) / 2
    }

    let mut n = 1;

    loop {
        let ns = num_sol(n);

        if ns > 1_000 {
            return n.to_string();
        }

        n += 1;
    }
}

pub fn p109() -> String {
    #[derive(Copy, Clone)]
    enum Kind {
        Single,
        Double,
        Triple,
    }

    #[derive(Copy, Clone)]
    struct NormalShot {
        kind: Kind,
        pnts: u64,
    }

    impl NormalShot {
        fn score(&self) -> u64 {
            (match self.kind {
                Kind::Single => 1,
                Kind::Double => 2,
                Kind::Triple => 3,
            }) * self.pnts
        }
    }

    #[derive(Copy, Clone)]
    enum Shot {
        Normal(NormalShot),
        SingleBull,
        DoubleBull,
        Miss,
    }

    impl Shot {
        fn score(&self) -> u64 {
            match self {
                &Shot::Normal(ref normal) => normal.score(),
                &Shot::Miss => 0,
                &Shot::DoubleBull => 50,
                &Shot::SingleBull => 25,
            }
        }
    }

    #[derive(Copy, Clone)]
    struct Checkout(Shot, Shot, Shot);

    impl Checkout {
        fn score(&self) -> u64 {
            let &Checkout(ref a, ref b, ref c) = self;
            a.score() + b.score() + c.score()
        }
    }

    let shots = {
        let mut shots = Vec::with_capacity(63);
        shots.push(Shot::Miss);

        for pnts in 1..21 {
            for kind in vec![Kind::Single, Kind::Double, Kind::Triple] {
                shots.push(Shot::Normal(NormalShot { kind, pnts }));
            }
        }

        shots.push(Shot::SingleBull);
        shots.push(Shot::DoubleBull);
        shots
    };

    let doubles = {
        let mut doubles = Vec::with_capacity(21);
        for pnts in 1..21 {
            doubles.push(Shot::Normal(NormalShot {
                kind: Kind::Double,
                pnts,
            }));
        }

        doubles.push(Shot::DoubleBull);
        doubles
    };

    let checkouts = {
        let mut checkouts = Vec::new();
        for i in 0..shots.len() {
            for j in i..shots.len() {
                for &last in doubles.iter() {
                    checkouts.push(Checkout(shots[i], shots[j], last));
                }
            }
        }

        checkouts
    };

    checkouts
        .into_iter()
        .filter(|&checkout| checkout.score() < 100)
        .count()
        .to_string()
}

pub fn p110() -> String {
    use std::iter::Chain;
    use std::vec::IntoIter;

    struct UniqueHeap {
        seen: HashSet<Partial>,
        heap: BinaryHeap<Partial>,
    }

    impl UniqueHeap {
        fn push(&mut self, partial: Partial) {
            if self.seen.contains(&partial) {
                return;
            }

            self.seen.insert(partial.clone());
            self.heap.push(partial);
        }

        fn pop(&mut self) -> Option<Partial> {
            self.heap.pop()
        }

        fn new() -> UniqueHeap {
            UniqueHeap {
                seen: HashSet::new(),
                heap: BinaryHeap::new(),
            }
        }
    }

    fn poss_primes() -> Chain<IntoIter<u64>, PossPrimes> {
        vec![2, 3].into_iter().chain(PossPrimes::new())
    }

    struct PossPrimes {
        next: u64,
        step: u64,
    }

    impl PossPrimes {
        fn new() -> PossPrimes {
            PossPrimes { next: 5, step: 2 }
        }
    }

    impl Iterator for PossPrimes {
        type Item = u64;

        fn next(&mut self) -> Option<Self::Item> {
            let out = self.next;
            self.next += self.step;
            self.step = 6 - self.step;
            Some(out)
        }
    }

    #[derive(Eq, PartialEq, Clone, Hash)]
    struct Partial {
        value: BigUint,
        num_sol: u64,
        powers: Vec<(u64, u64)>,
    }

    impl Partial {
        fn add_to(&self, heap: &mut UniqueHeap) {
            // increment 2
            let mut out_powers = self.powers.clone();
            out_powers[0].1 += 1;
            heap.push(Partial::from(out_powers));

            // increment powers (unless this is suboptimal)
            for i in 1..self.powers.len() {
                if self.powers[i].1 < self.powers[i - 1].1 {
                    let mut out_powers = self.powers.clone();
                    out_powers[i].1 += 1;
                    heap.push(Partial::from(out_powers));
                }
            }

            // add a prime on the end
            let last = self.powers[self.powers.len() - 1].0;
            let next_prime = poss_primes()
                .filter(|&n| n > last)
                .filter(|&n| n.is_prime())
                .nth(0)
                .unwrap();

            let mut out_powers = self.powers.clone();
            out_powers.push((next_prime, 1));
            heap.push(Partial::from(out_powers));
        }

        fn from(powers: Vec<(u64, u64)>) -> Partial {
            let num_sol = powers
                .iter()
                .map(|&(_, power)| 2 * power + 1)
                .fold(1, |acc, power| acc * power);

            let num_sol = (num_sol + 1) / 2;

            let value = powers
                .iter()
                .map(|&(base, power)| pow(BigUint::from(base), power as usize))
                .fold(BigUint::from(1_u64), |acc, power| &acc * &power);

            Partial {
                value,
                num_sol,
                powers,
            }
        }
    }

    impl Ord for Partial {
        fn cmp(&self, other: &Partial) -> Ordering {
            self.value.cmp(&other.value).reverse()
        }
    }

    impl PartialOrd for Partial {
        fn partial_cmp(&self, other: &Partial) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut to_process = UniqueHeap::new();
    to_process.push(Partial::from(vec![(2, 1)]));

    loop {
        let next = to_process.pop().unwrap();

        //println!("Processing [ value: {}, ns: {}, powers: {:?} ]", next.value, next.num_sol, next.powers);

        if next.num_sol > 4_000_000 {
            return next.value.to_string();
        } else {
            next.add_to(&mut to_process);
        }
    }
}

pub fn p112() -> String {
    fn not_bouncy(mut n: i64) -> bool {
        let mut direction = 0;
        let mut last_digit = n % 10;
        n /= 10;

        while direction == 0 && n > 0 {
            let next_digit = n % 10;
            direction = next_digit - last_digit;
            last_digit = next_digit;

            n /= 10;
        }

        let direction = direction; // frozen

        while n > 0 {
            let next_digit = n % 10;
            let next_direction = next_digit - last_digit;

            if next_direction * direction < 0 {
                return false;
            }

            n /= 10;
            last_digit = next_digit;
        }

        true
    }

    // just assume 1 is processed (not bouncy)
    let mut n = 2;
    let mut num_bouncy = 0;
    let mut num_seen = 1;

    while num_bouncy * 100 < num_seen * 99 {
        if !not_bouncy(n) {
            num_bouncy += 1;
        }
        num_seen += 1;

        n += 1;
    }

    num_seen.to_string()
}

pub fn p113() -> String {
    fn num_increasing(num_digits: u32) -> BigInt {
        let mut total = BigInt::from(0);
        let mut cache = HashMap::new();

        for first_digit in 1..10 {
            total = &total + num_inc_helper(num_digits - 1, first_digit, &mut cache);
        }

        total
    }

    fn num_inc_helper(
        num_digits: u32,
        first_digit: u32,
        cache: &mut HashMap<(u32, u32), BigInt>,
    ) -> &BigInt {
        let key = (num_digits, first_digit);
        if cache.contains_key(&key) {
            cache.get(&key).unwrap()
        } else {
            let total = if num_digits == 0 {
                BigInt::from(1)
            } else if num_digits == 1 {
                BigInt::from(10 - first_digit) // remaining choices; f, f+1, ..., 9
            } else {
                let mut total = BigInt::from(0);
                for next_digit in first_digit..10 {
                    total = &total + num_inc_helper(num_digits - 1, next_digit, cache);
                }
                total
            };

            cache.insert(key, total);
            cache.get(&key).unwrap()
        }
    }

    fn num_decreasing(num_digits: u32) -> BigInt {
        let mut total = BigInt::from(0);
        let mut cache = HashMap::new();

        for first_digit in 1..10 {
            total = &total + num_dec_helper(num_digits - 1, first_digit, &mut cache);
        }

        total
    }

    fn num_dec_helper(
        num_digits: u32,
        first_digit: u32,
        cache: &mut HashMap<(u32, u32), BigInt>,
    ) -> &BigInt {
        let key = (num_digits, first_digit);
        if cache.contains_key(&key) {
            cache.get(&key).unwrap()
        } else {
            let total = if num_digits == 0 {
                BigInt::from(1)
            } else if num_digits == 1 {
                BigInt::from(first_digit + 1) // remaining choices; 0, 1, 2, ..., first_digit
            } else {
                let mut total = BigInt::from(0);
                for next_digit in 0..first_digit + 1 {
                    total = &total + num_dec_helper(num_digits - 1, next_digit, cache);
                }
                total
            };

            cache.insert(key, total);
            cache.get(&key).unwrap()
        }
    }

    fn num_non_bouncy(num_digits: u32) -> BigInt {
        num_increasing(num_digits) + num_decreasing(num_digits) - BigInt::from(9) // remove the double counting
    }

    let mut total = BigInt::from(0);
    for num_digits in 1..100 + 1 {
        total = &total + &num_non_bouncy(num_digits);
    }
    total.to_string()
}

pub fn p114() -> String {
    fn combos(m: usize, n: usize, cache: &mut HashMap<usize, BigInt>) -> &BigInt {
        if cache.contains_key(&n) {
            return cache.get(&n).unwrap();
        }

        let out = {
            if n < m {
                BigInt::from(1)
            } else {
                let mut count = BigInt::from(1); // all red
                let mut red_len = m + 1;

                // strictly speaking -- we're allocating an entire red (of specified len) then a black
                while red_len <= n {
                    count = &count + combos(m, n - red_len, cache);
                    red_len += 1;
                }

                // then maybe start with a black
                count = &count + combos(m, n - 1, cache);
                count
            }
        };

        cache.insert(n, out);
        cache.get(&n).unwrap()
    }

    let mut cache = HashMap::new();
    let m = 3;

    combos(m, 50, &mut cache).to_string()
}

pub fn p115() -> String {
    fn combos(m: usize, n: usize, cache: &mut HashMap<usize, BigInt>) -> &BigInt {
        if cache.contains_key(&n) {
            return cache.get(&n).unwrap();
        }

        let out = {
            if n < m {
                BigInt::from(1)
            } else {
                let mut count = BigInt::from(1); // all red
                let mut red_len = m + 1;

                // strictly speaking -- we're allocating an entire red (of specified len) then a black
                while red_len <= n {
                    count = &count + combos(m, n - red_len, cache);
                    red_len += 1;
                }

                // then maybe start with a black
                count = &count + combos(m, n - 1, cache);
                count
            }
        };

        cache.insert(n, out);
        cache.get(&n).unwrap()
    }

    let goal = &BigInt::from(1_000_000);

    let mut cache = HashMap::new();
    let m = 50;

    let mut n = 1;
    loop {
        if combos(m, n, &mut cache) > goal {
            return n.to_string();
        } else {
            n += 1;
        }
    }
}

pub fn p116() -> String {
    fn fixed_count(block_len: usize, n: usize, cache: &mut HashMap<usize, BigInt>) -> &BigInt {
        if cache.contains_key(&n) {
            return cache.get(&n).unwrap();
        }

        let out = {
            if n < block_len {
                BigInt::from(1)
            } else {
                let black = fixed_count(block_len, n - 1, cache).clone();
                &black + fixed_count(block_len, n - block_len, cache)
            }
        };

        cache.insert(n, out);
        cache.get(&n).unwrap()
    }

    let n = 50;
    let mut out = BigInt::from(-3); // removing the "all black" solutions
    out = &out + fixed_count(2, n, &mut HashMap::new());
    out = &out + fixed_count(3, n, &mut HashMap::new());
    out = &out + fixed_count(4, n, &mut HashMap::new());

    out.to_string()
}

pub fn p117() -> String {
    fn fixed_count<'a>(
        block_lens: &Vec<usize>,
        n: usize,
        cache: &'a mut HashMap<usize, BigInt>,
    ) -> &'a BigInt {
        if cache.contains_key(&n) {
            return cache.get(&n).unwrap();
        }

        let mut total;
        if n == 0 {
            total = BigInt::from(1);
        } else {
            total = BigInt::from(0);
            for &s in block_lens.iter() {
                if s <= n {
                    total = &total + fixed_count(block_lens, n - s, cache);
                }
            }
        }

        cache.insert(n, total);
        cache.get(&n).unwrap()
    }

    let n = 50;
    fixed_count(&vec![1, 2, 3, 4], n, &mut HashMap::new()).to_string()
}

pub fn p118() -> String {
    use euler_lib::toys::nth_permutation;

    struct PossPrimes {
        next: u32,
        step: u32,
    }

    impl PossPrimes {
        fn new() -> PossPrimes {
            PossPrimes { next: 5, step: 2 }
        }
    }

    impl Iterator for PossPrimes {
        type Item = u32;

        fn next(&mut self) -> Option<Self::Item> {
            let out = self.next;
            self.next += self.step;
            self.step = 6 - self.step;
            Some(out)
        }
    }

    fn zero_free(mut n: u32) -> bool {
        while n > 0 {
            if n % 10 == 0 {
                return false;
            }
            n /= 10;
        }
        true
    }

    fn repeat_free(nums: &[u32]) -> bool {
        let mut found = [false; 10];

        for &n in nums {
            let mut n = n;
            while n > 0 {
                let digit = (n % 10) as usize;
                if found[digit] {
                    return false;
                }
                found[digit] = true;
                n /= 10;
            }
        }

        true
    }

    fn digit_count_vec(nums: &[u32]) -> usize {
        nums.iter().map(|&n| digit_count(n)).sum()
    }

    fn digit_count(mut n: u32) -> usize {
        let mut out: usize = 0;
        while n > 0 {
            out += 1;
            n /= 10;
        }
        out
    }

    fn vecs(so_far: &mut Vec<u32>, draw_from: &[u32], count: &mut usize) {
        let old_len = digit_count_vec(so_far);

        for i in 0..draw_from.len() {
            let next = draw_from[i];
            let next_len = digit_count(next);
            let full_len = old_len + digit_count(next);

            if full_len > 9 {
                break;
            }

            so_far.push(next);

            if !repeat_free(so_far) {
                so_far.pop();
                continue;
            }

            if full_len == 9 {
                *count += 1;
            } else if full_len + next_len <= 9 {
                vecs(so_far, &draw_from[i + 1..], count);
            }

            so_far.pop();
        }
    }

    // we could be smarter but why
    let primes: Vec<_> = PossPrimes::new()
        .take_while(|&n| n < 10_000_000)
        .filter(|&n| zero_free(n))
        .filter(|&p| p.is_prime())
        .fold(vec![2, 3], |mut acc, p| {
            acc.push(p);
            acc
        });

    let mut count: usize = 0;

    // this gives all the combos without 8 digit primes
    vecs(&mut vec![], &primes, &mut count);

    // but we can also have 8 digit primes and a 1 digit prime
    for one_digit in vec![2, 3, 5, 7] {
        let rem = (1..10).filter(|&d| d != one_digit).collect::<Vec<u32>>();

        for p_num in 0..40320 {
            // 0 to 8!
            let perm = nth_permutation(&rem, p_num);

            let p = perm.into_iter().fold(0, |acc, &d| acc * 10 + d);
            if (p as u32).is_prime() {
                count += 1;
            }
        }
    }

    // there are no 9 digit pandigital primes (9 digit pandigitals are all divisible by 3)

    count.to_string()
}

pub fn p119() -> String {
    #[derive(Eq, PartialEq, Debug)]
    struct NextPow {
        result: BigUint,
        base: u64,
        power: usize,
    }

    impl Ord for NextPow {
        fn cmp(&self, other: &NextPow) -> Ordering {
            self.result
                .cmp(&other.result)
                .reverse()
                .then_with(|| self.base.cmp(&other.base))
                .then_with(|| self.power.cmp(&other.power))
        }
    }

    impl PartialOrd for NextPow {
        fn partial_cmp(&self, other: &NextPow) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl NextPow {
        fn add_next(&self, heap: &mut BinaryHeap<NextPow>) {
            if self.base == 2 {
                heap.push(NextPow::from(2, self.power + 1));
            }

            heap.push(NextPow::from(self.base + 1, self.power));
        }

        fn from(base: u64, power: usize) -> NextPow {
            NextPow {
                result: pow(BigUint::from(base), power),
                base: base,
                power: power,
            }
        }

        fn is_good(&self) -> bool {
            let digit_sum = self
                .result
                .to_str_radix(10)
                .chars()
                .map(|c| c.to_digit(10).unwrap() as u64)
                .sum::<u64>();

            digit_sum == self.base
        }
    }

    let mut to_process = BinaryHeap::new();
    to_process.push(NextPow::from(2, 2));

    let goal_num = 30;
    let mut k = 0;

    loop {
        let next = to_process.pop().unwrap();
        next.add_next(&mut to_process);

        if next.is_good() {
            k += 1;

            if k >= goal_num {
                return next.result.to_string();
            }
        }
    }
}

pub fn p120() -> String {
    use std::cmp::max;

    // it's not hard to get a concrete formula but this still runs so fast it's offensive
    fn r_max(a: u64) -> u64 {
        let mut best = 2;
        for n in 0..2 * a {
            if n % 2 == 0 {
                continue;
            }

            best = max(best, (2 * n * a) % (a * a));
        }
        best
    }

    (3..1_001).map(|a| r_max(a)).sum::<u64>().to_string()
}

pub fn p121() -> String {
    use std::ops::{Add, Mul};

    #[derive(Clone)]
    struct Rational {
        num: BigInt,
        den: BigInt,
    }

    impl Rational {
        fn from(num: u64, den: u64) -> Rational {
            let g = gcd(num, den);
            Rational {
                num: BigInt::from(num / g),
                den: BigInt::from(den / g),
            }
        }
    }

    impl<'a> Mul<&'a Rational> for &'a Rational {
        type Output = Rational;

        fn mul(self, other: &Rational) -> Rational {
            let num = &self.num * &other.num;
            let den = &self.den * &other.den;

            let g = gcd(num.clone(), den.clone());

            Rational {
                num: &num / &g,
                den: &den / &g,
            }
        }
    }

    impl<'a> Add<&'a Rational> for &'a Rational {
        type Output = Rational;

        fn add(self, other: &Rational) -> Rational {
            let num = &self.num * &other.den + &self.den * &other.num;
            let den = &self.den * &other.den;

            let g = gcd(num.clone(), den.clone());

            Rational {
                num: &num / &g,
                den: &den / &g,
            }
        }
    }

    let turns = 15;

    // state[i] is probability of exactly i blues
    let mut state = vec![Rational::from(1, 1)];

    for turn in 1..(turns + 1) {
        let mut next: Vec<Rational> = Vec::with_capacity(turn + 1);

        let num_blues = 1 as u64;
        let num_reds = turn as u64;
        let num_discs = num_blues + num_reds;

        let blue_chance = Rational::from(num_blues, num_discs);
        let red_chance = Rational::from(num_reds, num_discs);

        {
            // first pull is special; blue impossible
            let next_red = &state[0] * &red_chance;
            next.push(next_red);
        }

        for i in 1..turn {
            let next_blue = &state[i - 1] * &blue_chance;
            let next_red = &state[i] * &red_chance;

            next.push(&next_blue + &next_red);
        }

        {
            // last pull is special; red impossible
            let next_blue = &state[turn - 1] * &blue_chance;
            next.push(next_blue);
        }

        state = next;
    }

    let win_prob = {
        let mut total = Rational::from(0, 1);
        let min_blues = turns / 2 + 1; // if turns=2k, this gives k+1; if turns=2k+1, this gives k+1

        for num_blues in min_blues..(turns + 1) {
            total = &total + &state[num_blues];
        }
        total
    };

    let payout = win_prob.den / win_prob.num;
    payout.to_string()
}

pub fn p122() -> String {
    #[allow(non_upper_case_globals)]
    const k_cap: usize = 200;

    struct VecSet {
        elts: [bool; k_cap + 1],
        size: usize,
    }

    impl VecSet {
        #[inline(always)]
        fn contains(&self, n: usize) -> bool {
            n <= k_cap && self.elts[n]
        }

        #[inline(always)]
        fn insert(&mut self, n: usize) -> bool {
            if !self.contains(n) {
                self.elts[n] = true;
                self.size += 1;
                true
            } else {
                false
            }
        }

        #[inline(always)]
        fn len(&self) -> usize {
            self.size
        }

        fn new() -> VecSet {
            VecSet {
                elts: [false; k_cap + 1],
                size: 0,
            }
        }
    }

    let mut seen = VecSet::new();
    seen.insert(1); // we can get it with 0 multiplications

    let mut total_cost: u32 = 0;

    let mut to_process = HashSet::new();
    to_process.insert(OrdSet::singleton(1));

    let mut num_mults = 0;

    // essentially a BFS for hitting everything from 2 to k_cap
    'main: loop {
        num_mults += 1;
        let mut next_to_process = HashSet::new();
        let mut done = false;

        'level: for possibility_set in to_process {
            if done {
                break;
            }

            for a in &possibility_set {
                for b in &possibility_set {
                    let total = *a + *b;
                    if b > a || total > k_cap || possibility_set.contains(&total) {
                        break 'level;
                    }

                    if seen.insert(total) {
                        //println!("m({})={}; found {} of {} so far", total, num_mults, seen.len(), k_cap);
                        total_cost += num_mults;
                        if seen.len() >= k_cap {
                            done = true;
                        }
                    }

                    next_to_process.insert(possibility_set.insert(total));
                }
            }
        }

        to_process = next_to_process;
        if done {
            break;
        }
    }

    total_cost.to_string()
}

pub fn p123() -> String {
    // ( (k-1)^n + (k+1)^n ) % k^2  =  2nk  if n is odd, or 0 if n is even

    let mut maybe_p: u64 = 3;
    let mut n: u64 = 1; // 1 prime has gone by (2)

    let goal = pow(10, 10);

    loop {
        if maybe_p.is_prime() {
            n += 1;

            if n % 2 == 1 {
                let rem = (2 * n * maybe_p) % (maybe_p * maybe_p);
                if rem > goal {
                    return n.to_string();
                }
            }
        }
        // ...

        maybe_p += 2;
    }
}

pub fn p124() -> String {
    let cap = 100000;

    let one: BigUint = BigUint::from(1 as u32);

    let mut e = (0..cap + 1)
        .map(|_ignored| BigUint::from(1 as u32))
        .collect::<Vec<BigUint>>();

    for i in 2..cap + 1 {
        // TODO: skip evens
        if &e[i] > &one {
            continue;
        }

        let mult = BigUint::from(i);

        // if we got here, we're prime
        let mut j = i;
        while j <= cap {
            e[j] = &e[j] * &mult;

            j += i;
        }
    }

    let mut e_sorted = (0..cap + 1).collect::<Vec<usize>>();
    e_sorted.sort_by_key(|x| &e[*x]);

    e_sorted[10000].to_string()
}

pub fn p125() -> String {
    let cap: u64 = 100_000_000; // 100_000_000

    let mut sum_sq = HashSet::new();

    let mut start = 1;
    while 2 * start * start <= cap {
        let mut end = start + 1;
        let mut total = start * start + end * end;

        while total <= cap {
            sum_sq.insert(total);

            end += 1;
            total += end * end;
        }

        start += 1;
    }

    fn is_palindrome(x: &u64) -> bool {
        let digits = x
            .to_string()
            .chars()
            .map(|d| d.to_digit(10).unwrap())
            .collect::<Vec<u32>>();

        let num_digits = digits.len();

        for i in 0..(num_digits / 2) {
            if digits[i] != digits[num_digits - i - 1] {
                return false;
            }
        }

        true
    }

    sum_sq
        .into_iter()
        .filter(|x| is_palindrome(x))
        .sum::<u64>()
        .to_string()
}

pub fn p129() -> String {
    struct NextP {
        next: u64,
        step: u64,
    }

    impl NextP {
        fn start() -> NextP {
            NextP { next: 5, step: 2 }
        }
    }

    impl Iterator for NextP {
        type Item = u64;

        fn next(&mut self) -> Option<u64> {
            let out = self.next;

            self.next += self.step;
            self.step = 6 - self.step; // 2 <-> 4

            Some(out)
        }
    }

    fn raw_ord(e: u64, base: u64, cache: &mut HashMap<(u64, u64), u64>) -> u64 {
        *cache.entry((e, base)).or_insert_with(|| {
            let mut n = 1;
            let mut e_pow = e % base;

            while e_pow > 1 {
                n += 1;
                e_pow = (e_pow * e) % base;
            }

            // now e_pow is either 1 or 0
            if e_pow == 0 {
                0
            } else {
                n
            }
        })
    }

    fn ord(e: u64, mut base: u64, cache: &mut HashMap<(u64, u64), u64>) -> u64 {
        let mut three_pow = 1;
        while base % 3 == 0 {
            base /= 3;
            three_pow *= 3;
        }

        let mut out = raw_ord(e, three_pow, cache);

        if base == 1 {
            return out;
        }

        for p in NextP::start() {
            if base % p == 0 {
                let mut ppow = p;
                base /= p;
                while base % p == 0 {
                    base /= p;
                    ppow *= p;
                }

                out = lcm(out, raw_ord(e, ppow, cache));

                if base == 1 {
                    break;
                }
            } else if p * p > base {
                // then `base` is a prime, work it in and stop
                out = lcm(out, raw_ord(e, base, cache));
                break;
            }
        }

        out
    }

    // NB: return value <= k-1 so no risk of overflow
    fn a(k: u64, cache: &mut HashMap<(u64, u64), u64>) -> u64 {
        if k % 2 == 0 || k % 5 == 0 {
            return 0;
        }

        ord(10, 9 * k, cache)
    }

    let mut cache = HashMap::new();
    let cap = 1_000_000;
    let mut k = cap; // A(k) < k so ...

    loop {
        let ak = a(k, &mut cache);
        if ak > cap {
            return k.to_string();
        }

        k += 1;
    }
}

pub fn p130() -> String {
    struct NextP {
        next: u64,
        step: u64,
    }

    impl NextP {
        fn start() -> NextP {
            NextP { next: 5, step: 2 }
        }
    }

    impl Iterator for NextP {
        type Item = u64;

        fn next(&mut self) -> Option<u64> {
            let out = self.next;

            self.next += self.step;
            self.step = 6 - self.step; // 2 <-> 4

            Some(out)
        }
    }

    fn raw_ord(e: u64, base: u64, cache: &mut HashMap<(u64, u64), u64>) -> u64 {
        *cache.entry((e, base)).or_insert_with(|| {
            let mut n = 1;
            let mut e_pow = e % base;

            while e_pow > 1 {
                n += 1;
                e_pow = (e_pow * e) % base;
            }

            // now e_pow is either 1 or 0
            if e_pow == 0 {
                0
            } else {
                n
            }
        })
    }

    fn ord(e: u64, mut base: u64, cache: &mut HashMap<(u64, u64), u64>) -> u64 {
        let mut three_pow = 1;
        while base % 3 == 0 {
            base /= 3;
            three_pow *= 3;
        }

        let mut out = raw_ord(e, three_pow, cache);

        if base == 1 {
            return out;
        }

        for p in NextP::start() {
            if base % p == 0 {
                let mut ppow = p;
                base /= p;
                while base % p == 0 {
                    base /= p;
                    ppow *= p;
                }

                out = lcm(out, raw_ord(e, ppow, cache));

                if base == 1 {
                    break;
                }
            } else if p * p > base {
                // then `base` is a prime, work it in and stop
                out = lcm(out, raw_ord(e, base, cache));
                break;
            }
        }

        out
    }

    // NB: return value <= k-1 so no risk of overflow
    fn a(k: u64, cache: &mut HashMap<(u64, u64), u64>) -> u64 {
        if k % 2 == 0 || k % 5 == 0 {
            return 0;
        }

        ord(10, 9 * k, cache)
    }

    fn is_prime(maybe: u64, seen: &Vec<u64>) -> bool {
        if maybe <= 2 {
            return maybe == 2;
        }

        for &p in seen {
            if maybe % p == 0 {
                return false;
            } else if p * p > maybe {
                return true;
            }
        }

        panic!("Insufficient primes vector!");
    }

    let mut primes = vec![2, 3, 5];

    let mut cache = HashMap::new();

    let mut n = 2;

    let mut solution_total = 0;
    let mut num_solutions = 0;

    loop {
        if n % 2 != 0 && n % 5 != 0 {
            if is_prime(n, &primes) {
                primes.push(n);
            } else {
                if (n - 1) % a(n, &mut cache) == 0 {
                    solution_total += n;
                    num_solutions += 1;

                    if num_solutions >= 25 {
                        break;
                    }
                }
            }
        }

        n += 1;
    }

    solution_total.to_string()
}

pub fn p132() -> String {
    struct PossPrimes {
        next: u64,
        step: u64,
    }

    impl Iterator for PossPrimes {
        type Item = u64;

        fn next(&mut self) -> Option<u64> {
            let out = self.next;
            self.next += self.step;
            self.step = 6 - self.step; // 2 <-> 4

            Some(out)
        }
    }

    fn poss_primes() -> PossPrimes {
        PossPrimes { next: 5, step: 2 }
    }

    // NB: (9*(10^9)) fits very happily in u64
    fn is_div(r_base: u64, divisor: u64) -> bool {
        // want divisor | R(r_base)
        // R(r_base) = (10^{r_base} - 1) / (9)
        // so equiv: 9 * divisor | 10^{r_base} - 1
        // equiv: powmod(10, r_base, 9*divisor) == 1
        powmod(10, r_base, 9 * divisor) == 1
    }

    let r_base = pow(10, 10);

    let mut primes_seen = Vec::new();

    // sloppy prime check method which only works IN THIS EXACT CONTEXT DO NOT COPY
    fn check_is_prime(k: u64, seen: &Vec<u64>) -> bool {
        for &p in seen {
            if k % p == 0 {
                return false;
            }
        }

        return true;
    }

    if is_div(r_base, 3) {
        primes_seen.push(3);
    }

    for p in poss_primes() {
        if check_is_prime(p, &primes_seen) && is_div(r_base, p) {
            primes_seen.push(p);

            if primes_seen.len() >= 40 {
                break;
            }
        }
    }

    primes_seen.into_iter().sum::<u64>().to_string()
}

pub fn p133() -> String {
    fn is_good(p: u64) -> bool {
        let mut seen = HashSet::new();

        let mut power = 10 % p;

        loop {
            if power == 1 {
                return false;
            } else if seen.contains(&power) {
                return true;
            } else {
                seen.insert(power);
                power = powmod(power, 10, p);
            }
        }
    }

    struct PossPrimes {
        next: u64,
        step: u64,
    }

    impl Iterator for PossPrimes {
        type Item = u64;

        fn next(&mut self) -> Option<u64> {
            let out = self.next;

            self.next += self.step;
            self.step = 6 - self.step;

            Some(out)
        }
    }

    fn is_prime(maybe: u64, primes: &Vec<u64>) -> bool {
        // PRE: maybe >= 2
        for p in primes {
            if maybe % p == 0 {
                return false;
            } else if p * p > maybe {
                return true;
            }
        }
        true
    }

    let mut primes = vec![2, 3, 5];

    let cap = 100_000;

    let mut total = 2 + 3 + 5;
    for maybe in (PossPrimes { next: 7, step: 4 }).take_while(|&p| p < cap) {
        if is_prime(maybe, &primes) {
            primes.push(maybe);
            if is_good(maybe) {
                total += maybe;
            }
        }
    }

    total.to_string()
}

pub fn p134() -> String {
    struct PossPrime {
        next: u64,
        step: u64,
    }

    impl PossPrime {
        fn new() -> PossPrime {
            PossPrime { next: 5, step: 2 }
        }
    }

    impl Iterator for PossPrime {
        type Item = u64;

        fn next(&mut self) -> Option<u64> {
            let out = self.next;
            self.next += self.step;
            self.step = 6 - self.step;
            Some(out)
        }
    }

    struct Pairs<T: Iterator> {
        iter: T,
        last: Option<T::Item>,
    }

    impl<T: Iterator> Pairs<T> {
        fn from(it: T) -> Pairs<T> {
            Pairs {
                iter: it,
                last: None,
            }
        }
    }

    impl<T> Iterator for Pairs<T>
    where
        T: Iterator,
        T::Item: Copy,
    {
        type Item = (T::Item, T::Item);

        fn next(&mut self) -> Option<Self::Item> {
            let next = self.iter.next();
            if next.is_none() {
                return None;
            }
            let next_item = next.unwrap();

            if let Some(prev_item) = self.last {
                let out = Some((prev_item, next_item));
                self.last = next;
                out
            } else {
                self.last = next;
                self.next()
            }
        }
    }

    fn ten_pow(n: u64) -> u64 {
        let mut out = 1;
        while out <= n {
            out *= 10;
        }
        out
    }

    fn find_s(p1: u64, p2: u64) -> u64 {
        let inc = ten_pow(p1);

        // want to minimize k where (inc * k + p1) % p2 == 0
        // iff      inc * k == -p1         (mod p2)
        // iff      k == -p1 * (inc^{-1})  (mod p2)
        // where inc^{-1} is computed mod p2

        let k = ((p2 - p1) * mod_inv(inc, p2)) % p2;
        let n = k * inc + p1;

        n
    }

    let primes = PossPrime::new().filter(|&n| n.is_prime());

    Pairs::from(primes)
        .take_while(|&(p1, _)| p1 <= 1_000_000)
        .map(|(p1, p2)| find_s(p1, p2))
        .fold(BigUint::from(0_u64), |acc, next| {
            &acc + &BigUint::from(next)
        })
        .to_string()
}

pub fn p137() -> String {
    // FACTS ABOUT FIBONACCI GOLDEN NUGGETS
    //
    // A_F(x) = \sum_{n=1}^\infty x^n F_n
    //        = x / (1-x-x^2)
    //
    // x_F(a) = [ -(a+1) + sqrt( (a+1)^2 + (2a)^2 ) ] / 2a

    // So a is a golden nugget
    //    iff x_F(a) is rational
    //    iff sqrt( (a+1)^2 + (2a)^2 ) is rational  [since a>0 is an integer]
    //    iff (a+1)^2 + (2a)^2 is a perfect square
    //    iff for some k, (a+1, 2a, k) is a pythagorean triple
    //
    // Don't actually care what x is -- just want positive integers a where \exists k where (a+1, 2a, k) is a PT
    //
    // After Euclid's formula we discover that either of:
    //        A = m^2 - n^2 - 1, 	where m = 1/2 * (n + sqrt(5n^2 + 4))
    //              Note:	if n is odd then so is 5n^2+4, so (assuming the square root is an integer), n + sqrt(5n^2+4) is even
    //                      if n is even then 4 | 5n^2+4,  so (assuming the square root is an integer), n + sqrt(5n^2+4) is even
    //                      so we don't need to worry about this half being well-defined
    //
    //        A = 4mn - 1, 		where m = 2n + sqrt(3n^2 - 1)
    //              It turns out that the Pell equation u^2 - 3n^2 = -1 has no solutions, so there are no solutions of the second kind.
    //
    // The first solution to u^2 - 5v^2 = 4 is (3, 1)
    // Therefore the recurrence for solutions here is:
    //        u_{n+1}  =  (1/2) (3u_n + 5v_n)
    //        v_{n+1}  =  (1/2) ( u_n + 3v_n)
    //
    // So we iterate through solutions (u,v) to u^2 - 5v^2 = 4
    // Turn these into pairs (m,n) = ( (u+v)/2, v )
    // Then turn these into elements a = m^2 - n^2 - 1
    use std::mem;

    #[derive(Clone, Debug)]
    struct Pell {
        u: BigInt,
        v: BigInt,
    }

    impl Pell {
        fn start() -> Pell {
            Pell {
                u: BigInt::from(3),
                v: BigInt::from(1),
            }
        }

        fn next(&self) -> Pell {
            let two = BigInt::from(2);
            let three = BigInt::from(3);

            let three_u = &three * &self.u;
            let five_v = &BigInt::from(5) * &self.v;
            let three_v = &three * &self.v;

            Pell {
                u: &(&three_u + &five_v) / &two,
                v: &(&self.u + &three_v) / &two,
            }
        }
    }

    struct PellIter {
        next: Pell,
    }

    impl PellIter {
        fn start() -> PellIter {
            PellIter {
                next: Pell::start(),
            }
        }
    }

    impl Iterator for PellIter {
        type Item = Pell;

        fn next(&mut self) -> Option<Pell> {
            let next = self.next.next();
            Some(mem::replace(&mut self.next, next))
        }
    }

    let one = BigInt::from(1);
    let two = BigInt::from(2);

    let pell = PellIter::start().nth(14).unwrap();

    let m = &(&pell.v + &pell.u) / &two;
    let n = pell.v;

    let a = &(&m * &m) - &(&n * &n) - &one;
    a.to_string()
}

pub fn p139() -> String {
    // Looking for PTs (a, b, c) where c % (a-b) == 0
    // If (a, b, c) works then so does (ka, kb, kc) and vice-versa
    //
    // Perimeter is a+b+c; by Euclid a = m^2-n^2, b = 2mn, c = m^2 + n^2 (skipping k)
    // so perimeter is 2m^2 + 2mn = 2m(m+n)

    let cap = pow(10, 8);

    let mut count = 0;

    let mut m: i64 = 1; // i64::max is well above the numbers we'll need

    loop {
        if 2 * m * m >= cap {
            break;
        }

        let mut n = 1 + (m % 2); // opposite parity
        while n < m {
            let peri = 2 * m * (m + n);
            if 2 * m * (m + n) >= cap {
                break;
            } else if gcd(m, n) == 1 {
                let a = m * m - n * n;
                let b = 2 * m * n;
                let c = m * m + n * n;

                if c % (a - b) == 0 {
                    count += (cap - 1) / peri; // the number of multiples of this triangle that fit
                }
            }

            n += 2;
        }

        m += 1;
    }

    count.to_string()
}

pub fn p140() -> String {
    // FACTS ABOUT MODIFIED FIBONACCI GOLDEN NUGGETS
    //
    // G_n = phi^n ( 4phi + 1 ) / (phi^2 sqrt(5))  +  phi^n ( -4psi - 1 ) / (psi^2 sqrt(5))
    //
    // So A_G(x) = xG_1 + x^2G_2 + ...
    //           = ...
    //           = x ( 1 - 3x ) / ( 1 - x - x^2 )
    //
    // Inverting, x = ( A + 1 \pm \sqrt{ (A+1)^2 + 4A(A+3) }) / (-6-2A)
    // Only need x rational (nnec integer) so only need (A+1)^2 + 4A(A+3) a perfect square
    // After completing the square we see we need  5( A + 7/5 )^2  -  44/5  =  v^2  for some v
    // Let u = 5A+7; then we need  u^2 - 5v^2  =  44    AND    u === 2 (mod 5)
    // (This is equivalent! Given (u,v) we have a = (u-7)/5; given (A,v) we have u=5A+7)
    //
    // So need solutions to  u^2 - 5v^2  =  44
    //
    // The minimal solution is (7, 1)   [found by hand]
    // But since N = 44 is not \pm 1 or \pm 4, it's a little more complicated than before.
    //
    // The minimal solution to the associated equation x^2 - 5y^2 = 1 is (9, 4)     [found by hand]
    //
    // So given ANY solution (u, v), the next equivalent solution is (9u+20v, 4u+9v)
    // The equivalence classes are "dense in each other" so we can find representatives of each
    // class between the first and its successor.
    //
    // The successor of (7, 1) is (9*7 + 20*1, 4*7+9*1) = (83, 37)
    // The solutions to u^2 - 5v^2 = 44 with 7 <= u < 83 are:
    //      ( 7,  1)
    //      ( 8,  2)
    //      (13,  5)
    //      (17,  7)
    //      (32, 14)
    //      (43, 19)
    //
    // For a total of 6 classes. Additionally the order doesn't change (so if i < j, then the
    // nth iterate of class i always appears before the nth iterate of class j)
    //
    // So we need only iterate through all of them in a round robin fashion, filter out those (u, v)
    // where u !=== 2 (mod 5), transform the remaining u into A, take the first 30, and add them up!

    use std::mem::replace;

    struct PellIter {
        u: BigInt,
        v: BigInt,
    }

    impl Iterator for PellIter {
        type Item = (BigInt, BigInt);

        fn next(&mut self) -> Option<Self::Item> {
            let four = BigInt::from(4);
            let nine = BigInt::from(9);
            let twenty = BigInt::from(20);

            let next_u = &(&nine * &self.u) + &(&twenty * &self.v);
            let next_v = &(&four * &self.u) + &(&nine * &self.v);

            let u = replace(&mut self.u, next_u);
            let v = replace(&mut self.v, next_v);

            Some((u, v))
        }
    }

    let mut fundamentals: Vec<PellIter> =
        vec![(7, 1), (8, 2), (13, 5), (17, 7), (32, 14), (43, 19)]
            .into_iter()
            .map(|(u, v)| PellIter {
                u: BigInt::from(u),
                v: BigInt::from(v),
            })
            .collect();

    let mut i = 0;
    let mut count = 0;

    let mut total = BigInt::from(0);

    let two = BigInt::from(2);
    let five = BigInt::from(5);
    let seven = BigInt::from(7);

    while count < 30 {
        let (u, _) = fundamentals[i].next().unwrap();

        if u > seven && &u % &five == two {
            let a = &(&u - &seven) / &five;
            count += 1;
            total = &total + &a;
            //println!("Nugget {}: {}", count, a);
        }

        i = (i + 1) % 6;
    }

    total.to_string()
}

pub fn p145() -> String {
    fn reverse(mut n: u64) -> u64 {
        let mut out = 0;
        while n > 0 {
            out = 10 * out + (n % 10);
            n /= 10;
        }
        out
    }

    fn is_good(n: u64) -> bool {
        let mut total = n + reverse(n);
        while total > 0 {
            if (total % 10) % 2 == 0 {
                return false;
            }
            total /= 10;
        }
        true
    }

    let num_digits = 9; // max num digits, really
    let cap = pow(10, num_digits); // n < cap

    let mut total = 0;
    for n in 1..cap {
        // check for leading zeroes
        if n % 10 == 0 {
            continue;
        }
        if is_good(n) {
            total += 1;
        }
    }

    total.to_string()
}

pub fn p146() -> String {
    fn lcm_vec(v: &[u64]) -> u64 {
        let mut out = v[0];
        for i in 1..v.len() {
            out = lcm(out, v[i]);
        }
        out
    }

    fn is_good(n: u64) -> bool {
        let nsq = n * n;

        [27, 13, 9, 7, 3, 1]
            .into_iter()
            .all(|add| (nsq + add).is_prime())
    }

    let mut total: u64 = 0;
    let cap = 150_000_000; //150_000_000;

    // the actual numbers will be n plus some residual (below)
    let mut n = 0;

    // Fast primality check gets us in the ball park, then we filter by various remainders of n*n
    let step = lcm_vec(&vec![2, 3, 5, 7, 11, 13]);
    let res = (0..step)
        .filter(|&k| (k * k) % 2 == 0)
        .filter(|&k| (k * k) % 3 == 1)
        .filter(|&k| (k * k) % 5 == 0)
        .filter(|&k| (k * k) % 7 == 2)
        .filter(|&k| vec![0, 1, 5, 3].contains(&((k * k) % 11)))
        .filter(|&k| vec![1, 3, 9, 10].contains(&((k * k) % 13)))
        .collect::<Vec<u64>>();

    while n < cap {
        for x in res.iter().map(|r| r + n) {
            if is_good(x) && !(x * x + 21).is_prime() {
                if n < cap {
                    total += x;
                } else {
                    break;
                }
            }
        }

        n += step;
    }

    total.to_string()
}

pub fn p147() -> String {
    fn count_straight(xmax: u64, ymax: u64) -> u64 {
        let mut total = 0;

        // (x, y) can be any point in the grid, but there is no point
        // in considering things on the right or bottom edge
        for x in 0..xmax {
            for y in 0..ymax {
                // count the number of straight rects in the grid with
                // UL corner (x, y)
                total = total + (xmax - x) * (ymax - y);
            }
        }

        total
    }

    fn count_tilted(xmax: u64, ymax: u64) -> u64 {
        let mut total = 0;

        let effective_xmax = xmax * 2;
        let effective_ymax = ymax * 2;

        for x in 0..effective_xmax {
            for y in 1..effective_ymax {
                if x % 2 != y % 2 {
                    continue;
                }

                // count the number of tilted rectangles in the grid with
                // LEFT corner (x, y)
                let max_up = y;
                let max_down = effective_ymax - y;

                let max_right = effective_xmax - x;

                // the number of TOTAL moves (up-right and down-right) we're making for this rectangle
                for right_moves in 2..max_right + 1 {
                    for up_right_moves in 1..min(right_moves, max_up + 1) {
                        let down_right_moves = right_moves - up_right_moves;
                        if down_right_moves <= max_down {
                            total += 1;
                        }
                    }
                }
            }
        }

        total
    }

    let xmax_max = 47;
    let ymax_max = 43;

    let mut total = 0;

    for xmax in 1..xmax_max + 1 {
        for ymax in 1..ymax_max + 1 {
            total = total + (count_straight(xmax, ymax) + count_tilted(xmax, ymax));
        }
    }

    total.to_string()
}

pub fn p149() -> String {
    use std::cmp::max;

    trait MaxSubiter<I: Iterator<Item = i64>> {
        fn max_subiter(self) -> MaxSubarrayIter<I>;
    }

    impl<I: Iterator<Item = i64>> MaxSubiter<I> for I {
        fn max_subiter(self) -> MaxSubarrayIter<I> {
            MaxSubarrayIter {
                iter: self,
                best: None,
            }
        }
    }

    struct MaxSubarrayIter<I: Iterator<Item = i64>> {
        iter: I,
        best: Option<i64>,
    }

    impl<I: Iterator<Item = i64>> Iterator for MaxSubarrayIter<I> {
        type Item = i64;

        fn next(&mut self) -> Option<i64> {
            if let Some(next) = self.iter.next() {
                if self.best.is_none() {
                    self.best = Some(next);
                } else {
                    self.best = Some(max(self.best.unwrap() + next, next));
                }

                self.best
            } else {
                None
            }
        }
    }

    let grid = {
        let cap = 4_000_000;
        let mut lagged: Vec<i64> = Vec::with_capacity(cap);

        for k in 0..55 {
            let next =
                ((100003 - (200003 * (k + 1)) + (300007 * pow(k + 1, 3))) % 1000000) - 500000;
            lagged.push(next);
        }

        for k in 55..cap {
            let next = ((lagged[k - 24] + lagged[k - 55] + 1000000) % 1000000) - 500000;
            lagged.push(next);
        }

        RectVec::from(lagged, 2000, 2000).unwrap()
    };

    let mut best = 0;

    // sweep right
    for y in 0..2_000 {
        let max_right = (0..2_000)
            .map(|x| *grid.get(x, y).unwrap())
            .max_subiter()
            .max()
            .unwrap();
        best = max_right;
    }

    // sweep down
    for x in 0..2_000 {
        let max_down = (0..2_000)
            .map(|y| *grid.get(x, y).unwrap())
            .max_subiter()
            .max()
            .unwrap();
        best = max(best, max_down);
    }

    // sweep down-right; can start from left wall or top wall
    for start_y in 0..2_000 {
        // left wall
        let left_dr_max = (start_y..2_000) // our end will be when y is too big
            .map(|y| *grid.get(y - start_y, y).unwrap())
            .max_subiter()
            .max()
            .unwrap();

        best = max(best, left_dr_max);
    }

    for start_x in 0..2_000 {
        // top wall
        let top_dr_max = (start_x..2_000) // end when x is too big
            .map(|x| *grid.get(x, x - start_x).unwrap())
            .max_subiter()
            .max()
            .unwrap();

        best = max(best, top_dr_max);
    }

    // now sweep up-right; can start from left wall or bottom wall
    use num::iter::range_step_inclusive;

    for start_y in 0..2_000_i64 {
        // left wall
        let left_ur_max =
            range_step_inclusive(start_y, 0, -1) // end when y breaks the top
                .map(|y| *grid.get((start_y - y) as usize, y as usize).unwrap())
                .max_subiter()
                .max()
                .unwrap();

        best = max(best, left_ur_max);
    }

    for start_x in 0..2_000 {
        // bottom wall
        let bott_ur_max = (start_x..2_000)
            .map(|x| *grid.get(x, 1_999 - (x - start_x)).unwrap())
            .max_subiter()
            .max()
            .unwrap();

        best = max(best, bott_ur_max);
    }

    best.to_string()
}
