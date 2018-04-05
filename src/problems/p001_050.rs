use std::cmp::{max};
use std::fs::File;
use std::io::prelude::*;
use std::collections::{HashMap, HashSet, BinaryHeap};

use num;
use num::Integer;
use num::bigint::BigInt;

use std::str::FromStr;

use itertools;

use euler_lib::prelude::*;
use euler_lib::{toys, numerics};

pub fn problem(problem_number: i32) -> String {
    match problem_number {
        1 => p001(),
        2 => p002(),
        3 => p003(),
        4 => p004(),
        5 => p005(),
        6 => p006(),
        7 => p007(),
        8 => p008(),
        9 => p009(),
        10 => p010(),
        11 => p011(),
        12 => p012(),
        13 => p013(),
        14 => p014(),
        15 => p015(),
        16 => p016(),
        17 => p017(),
        18 => p018(),
        19 => p019(),
        20 => p020(),
        21 => p021(),
        22 => p022(),
        23 => p023(),
        24 => p024(),
        25 => p025(),
        26 => p026(),
        27 => p027(),
        28 => p028(),
        29 => p029(),
        30 => p030(),
        31 => p031(),
        32 => p032(),
        33 => p033(),
        34 => p034(),
        35 => p035(),
        36 => p036(),
        37 => p037(),
        38 => p038(),
        39 => p039(),
        40 => p040(),
        41 => p041(),
        42 => p042(),
        43 => p043(),
        44 => p044(),
        45 => p045(),
        46 => p046(),
        47 => p047(),
        48 => p048(),
        49 => p049(),
        50 => p050(),

        _ => {
            panic!("Problem {} should not be passed to this module!", problem_number);
        },
    }
}

pub fn p001() -> String {
    let mut total = BigInt::from(0);
    let mut i = BigInt::from(0);

    let cap = BigInt::from(1000);
    let one = BigInt::from(1);
    let three = BigInt::from(3);
    let five = BigInt::from(5);
    let zero = BigInt::from(0);

    while i < cap {
        if &i % &three == zero || &i % &five == zero {
            total = &total + &i;
        }
        i = &i + &one;
    }

    total.to_str_radix(10)
}

pub fn p002() -> String {
    let mut a = BigInt::from(1);
    let mut b = BigInt::from(2);

    let mut total = BigInt::from(0);
    let cap = BigInt::from_str("4000000").unwrap();

    while &b <= &cap {
        if (&b).is_even() {
            total = &total + &b;
        }

        let c = &a + &b;
        a = b;
        b = c;
    }

    total.to_str_radix(10)
}

pub fn p003() -> String {
    let n = BigInt::from_str("600851475143").unwrap();
    numerics::largest_prime_factor(&n).to_str_radix(10)
}

pub fn p004() -> String {
    let max = BigInt::from(999);
    let min = BigInt::from(100);

    let one = BigInt::from(1);

    let mut best = BigInt::from(-1);

    let mut a = max.clone();
    while a >= min && (&a * &max > best) {
        let mut b = max.clone();
        while b >= a && (&b * &a > best) {
            let prod = &a * &b;
            if toys::is_palindrome(&prod) {
                best = prod;
            }

            b = &b - &one;
        }

        a = &a - &one;
    }

    best.to_str_radix(10)
}

pub fn p005() -> String {
    let mut out = BigInt::from(1);
    let mut i = BigInt::from(1);

    let one = BigInt::from(1);
    let cap = BigInt::from(20);

    while i <= cap {
        out = num::Integer::lcm(&out, &i);
        i = &i + &one;
    }

    out.to_str_radix(10)
}

pub fn p006() -> String {
    let mut sq_sum = BigInt::from(0);
    let mut sum_sq = BigInt::from(0);

    let mut i = BigInt::from(1);

    let one = BigInt::from(1);
    let cap = BigInt::from(100);

    while i <= cap {
        sq_sum = &sq_sum + &i;
        sum_sq = &sum_sq + &(&i * &i);

        i = &i + &one;
    }

    sq_sum = &sq_sum * &sq_sum;

    (&sq_sum - &sum_sq).to_str_radix(10)
}

pub fn p007() -> String {
    let mut count = 0 as i32;
    let cap = 10_001;

    for p in numerics::possible_primes() {
        if numerics::is_prime(&p) {
            count += 1;
            if count == cap {
                return p.to_str_radix(10)
            }
        }
    }

    cannot_happen();
}

pub fn p008() -> String {
    use std::fs::File;
    use std::io::prelude::*;

    let filename = "resources/p008.txt";

    let mut f = File::open(filename)
        .expect(&format!("File '{}' not found!", filename));

    let mut contents = String::new();

    f.read_to_string(&mut contents)
        .expect("IO issue of some kind");

    let contents: Vec<i64> = contents.chars()
        .filter(|c: &char| c.is_digit(10))
        .map(|c: char| c.to_digit(10).unwrap() as i64) // 9^13 < i64.max_value
        .collect();

    let prod_length = 13;
    let mut max_prod = 0;

    for i in prod_length .. (contents.len()+1) {
        let prod: i64 = contents[i-prod_length .. i].iter().product();
        max_prod = max(max_prod, prod);
    }

    max_prod.to_string()
}

pub fn p009() -> String {
    let max_sum = 1000;

    for a in 1 .. max_sum / 3 {
        for b in a+1 .. (max_sum - a) / 2 {
            let c = 1000 - a - b;

            if a*a + b*b == c*c {
                return (a*b*c).to_string();
            }
        }
    }

    cannot_happen();
}

pub fn p010() -> String {
    let mut sum = BigInt::from(0);

    for p in numerics::all_primes(2_000_000) {
        sum = sum + BigInt::from(p);
    }

    sum.to_str_radix(10)
}

pub fn p011() -> String  {
    let mut text = String::new();

    File::open("resources/p011.txt").expect("Error reading file!")
        .read_to_string(&mut text).expect("Error reading file!");

    let grid: Vec<Vec<i64>> = text.trim()
        .lines()
        .map(|line: &str| {
            line.split_whitespace().map(|token| {
                token.parse::<i64>().expect("Couldn't parse string?")
            }).collect::<Vec<i64>>()
        }).collect();

    let size = 20; // height and width

    let mut max_prod = -1;
    for y in 0 .. size {
        for x in 0 .. size - 4 {
            let horizontal: i64 = (0..4).map(|i: i64| grid.get(y).unwrap().get(x+i as usize).unwrap()).product();
            max_prod = max(max_prod, horizontal);
        }
    }

    for x in 0 .. size {
        for y in 0 .. size - 4 {
            let vertical: i64 = (0..4).map(|i: i64| grid.get(y+i as usize).unwrap().get(x).unwrap()).product();
            max_prod = max(max_prod, vertical);
        }
    }

    for x in 0 .. size - 4 {
        for y in 0 .. size - 4 {
            let dr_diag: i64 = (0..4).map(|i: i64| grid.get(y+i as usize).unwrap().get(x+i as usize).unwrap()).product();
            max_prod = max(max_prod, dr_diag);
        }
    }

    for x in 3 .. size {
        for y in 0 .. size - 4 {
            let dl_diag = (0..4).map(|i: i64| grid.get(y+i as usize).unwrap().get(x-i as usize).unwrap()).product();
            max_prod = max(max_prod, dl_diag);
        }
    }

    max_prod.to_string()
}

pub fn p012() -> String {
    use self::numerics::num_divisors;

    let goal = BigInt::from(500);

    let mut n = BigInt::from(1);
    let mut tri = BigInt::from(1);

    let one = BigInt::from(1);
    let two = BigInt::from(2);

    loop {
        n = &n + &one;
        tri = &tri + &n;

        // tri = (n * (n+1)) / 2; these are coprime so can num_divisors them separately
        let nd = if n.is_even() {
            num_divisors(&n / &two) * num_divisors(&n + &one)
        } else {
            num_divisors(n.clone()) * num_divisors(&(&n + &one) / &two)
        };

        if nd > goal {
            break;
        }
    }

    tri.to_str_radix(10)
}

pub fn p013() -> String {
    let mut text = String::new();

    File::open("resources/p013.txt").expect("Error reading file!")
        .read_to_string(&mut text).expect("Error reading file!");

    let total = text.trim()
        .lines()
        .map(|line: &str| BigInt::from_str(line.trim()).expect("Error parsing BigInt"))
        .fold(BigInt::from(0), |acc, val| acc + val);

    format!("{}", total).chars().take(10).collect()
}

pub fn p014() -> String {
    fn collatz_count(n: i64, cache: &mut HashMap<i64, i64>) -> i64 {
        if n == 1 {
            1
        } else if cache.contains_key(&n) {
            *cache.get(&n).unwrap()
        } else {
            let next = if n % 2 == 0 {
                n / 2
            } else {
                3 * n + 1
            };

            let val = collatz_count(next, cache) + 1;
            cache.insert(n, val);
            val
        }
    }

    let mut cache = HashMap::new();

    let (best_i, _) = (1 .. 1_000_000)
        .map(|i| (i, collatz_count(i, &mut cache)))
        .fold((0, 0), |(acc_i, acc_ct), (next_i, next_ct)| {
            if acc_ct >= next_ct {
                (acc_i, acc_ct)
            } else {
                (next_i, next_ct)
            }
        });

    best_i.to_string()
}

pub fn p015() -> String {
    fn count(x: u32, y: u32, cache: &mut HashMap<(u32, u32), BigInt>) -> BigInt {
        if x == 0 || y == 0 {
            BigInt::from(1)
        } else if cache.contains_key(&(x, y)) {
            cache.get(&(x, y)).unwrap().clone()
        } else {
            let ways = count(x-1, y, cache) + count(x, y-1, cache);
            cache.insert((x, y), ways.clone());
            ways
        }
    }

    let size = 20;

    count(size, size, &mut HashMap::new()).to_str_radix(10)
}

pub fn p016() -> String {
    use num::pow;

    let digit_sum: BigInt = pow(BigInt::from(2), 1000)
        .to_str_radix(10).chars().filter_map(|c: char| c.to_digit(10))
        .fold(BigInt::from(0), |acc: BigInt, digit: u32| &acc + &BigInt::from(digit));

    digit_sum.to_str_radix(10)
}

pub fn p017() -> String {
    fn char_len(s: &str) -> u64 {
        s.chars().count() as u64
    }

    fn letter_count(n: u64) -> u64 {
        if n < 1 || n > 1_000 {
            panic!("Unsupported argument!");
        } else if n == 1_000 {
            char_len("onethousand")
        } else if n % 100 == 0 {
            char_len("hundred") + letter_count(n / 100)
        } else if n > 100 {
            letter_count(n - (n%100)) + letter_count(n%100) + 3
        } else if n < 20 {
            let s = match n {
                1 => "one",
                2 => "two",
                3 => "three",
                4 => "four",
                5 => "five",
                6 => "six",
                7 => "seven",
                8 => "eight",
                9 => "nine",
                10 => "ten",
                11 => "eleven",
                12 => "twelve",
                13 => "thirteen",
                14 => "fourteen",
                15 => "fifteen",
                16 => "sixteen",
                17 => "seventeen",
                18 => "eighteen",
                19 => "nineteen",

                _ => panic!("Fuck!"),
            };
            char_len(s)
        } else if n % 10 == 0 {
            let s = match n {
                20 => "twenty",
                30 => "thirty",
                40 => "forty",
                50 => "fifty",
                60 => "sixty",
                70 => "seventy",
                80 => "eighty",
                90 => "ninety",

                _ => panic!("Fuck!"),
            };
            char_len(s)
        } else {
            letter_count(n - (n % 10)) + letter_count(n % 10)
        }
    }

    (1 .. 1_000+1).map(letter_count).sum::<u64>().to_string()
}

pub fn p018() -> String {
    let mut text = String::new();

    File::open("resources/p018.txt").expect("Error reading file!")
        .read_to_string(&mut text).expect("Error reading file!");

    let grid: Vec<Vec<i64>> = text.trim()
        .lines()
        .map(|line: &str| {
            line.split_whitespace().map(|token| {
                token.parse::<i64>().expect("Couldn't parse string?")
            }).collect::<Vec<i64>>()
        }).collect();

    fn best_path(x: usize, y: usize, grid: &Vec<Vec<i64>>, cache: &mut HashMap<(usize, usize), i64>) -> i64 {
        let curr = *grid.get(y).unwrap().get(x).unwrap();
        if y == grid.len() - 1 {
            curr
        } else {
            let left = best_path(x, y+1, grid, cache);
            let right = best_path(x+1, y+1, grid, cache);
            let val = curr + max(left, right);
            cache.insert((x, y), val);
            val
        }
    }

    best_path(0, 0, &grid, &mut HashMap::new()).to_string()
}

pub fn p019() -> String {
    struct Date {
        month: u32, // 1 is January ... 12 is December (no 0, no 13)
        date: u32,  // 1 to self.end_of_month()
        year: u32,  // 1901 to 2001
        day_of_week: u32, // 1 is Sunday, 2 is Monday ... 7 is Saturday
    }

    // I might have gone overboard on this but date math is actually kind of annoying
    impl Date {
        fn is_leap_year(&self) -> bool {
            return self.year % 4 == 0 && (self.year % 100 != 0 || self.year % 400 == 0)
        }

        fn end_of_month(&self) -> u32 {
            if vec![1, 3, 5, 7, 8, 10, 12].contains(&self.month) {
                31
            } else if vec![4, 6, 9, 11].contains(&self.month) {
                30
            } else if self.month == 2 {
                if self.is_leap_year() { 29 } else { 28 }
            } else {
                panic!("Unrecognized month {}; expected 1, 2, 3, ..., 12", self.month);
            }
        }

        fn is_win(&self) -> bool {
            self.day_of_week == 1 && self.date == 1
        }

        fn stop(&self) -> bool {
            self.year > 2000
        }

        fn start(&self) -> bool {
            self.year >= 1901
        }

        fn next_month(&self) -> u32 {
            if self.month == 12 {
                1
            } else {
                self.month + 1
            }
        }

        fn next(self) -> Date {
            let next_date =
                if self.end_of_month() == self.date {
                    1
                } else {
                    self.date + 1
                };

            let next_month =
                if next_date == 1 {
                    self.next_month()
                } else {
                    self.month
                };

            let next_year =
                if next_date == 1 && next_month == 1 {
                    self.year + 1
                } else {
                    self.year
                };

            let next_day_of_week =
                if self.day_of_week == 7 {
                    1
                } else {
                    self.day_of_week + 1
                };

            Date {
                date: next_date,
                month: next_month,
                year: next_year,
                day_of_week: next_day_of_week
            }
        }
    }

    let mut current_date = Date { day_of_week: 2, date: 1, month: 1, year: 1900 };

    let mut count = 0;

    while ! current_date.stop() {
        if current_date.start() && current_date.is_win() {
            count += 1;
        }

        current_date = current_date.next();
    }

    count.to_string()
}

pub fn p020() -> String {
    fn fact(mut n: u32) -> BigInt {
        let mut out = BigInt::from(1);
        while n > 1 {
            out = &out * &BigInt::from(n);
            n -= 1;
        }
        out
    }

    fact(100)
        .to_str_radix(10).chars()
        .filter_map(|c: char| c.to_digit(10))
        .fold(BigInt::from(0), |acc, digit| acc + BigInt::from(digit))
        .to_str_radix(10)
}

pub fn p021() -> String {
    let prop_div = |n: &BigInt| &numerics::sum_divisors(n.clone()) - n;
    let mut cache: HashMap<BigInt, BigInt> = HashMap::new();

    // cached proper divisors
    let mut cpd = |n: &BigInt| {
        if cache.contains_key(n) {
            cache.get(n).unwrap().clone()
        } else {
            let val = prop_div(n);
            cache.insert(n.clone(), val.clone());
            val
        }
    };

    (2 .. 10000)
        .map(|n: u32| BigInt::from(n))
        .filter(|n: &BigInt| {
            let sd = cpd(n);
            &sd != n && &cpd(&sd) == n
        }).fold(BigInt::from(0), |acc, next| &acc + &next)
        .to_str_radix(10)
}

pub fn p022() -> String {
    let letter_score: HashMap<char, u64> = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        .chars().enumerate()
        .fold(HashMap::new(), |mut map, tup| {
            map.insert(tup.1, tup.0 as u64 + 1);
            map
        });

    let score = |name: &str| -> u64 {
        name.chars().map(|c| *letter_score.get(&c).unwrap()).sum()
    };

    let mut text = String::new();

    File::open("resources/p022.txt").expect("Error reading file!")
        .read_to_string(&mut text).expect("Error reading file!");

    let mut names: Vec<String> = text.split(",")
        .map(|token| token.chars().filter(|c| *c != '"').collect())
        .collect();

    names.sort();

    let total: u64 = names.iter().enumerate()
        .map(|tup| (tup.0 as u64 + 1) * score(tup.1))
        .sum();

    total.to_string()
}

pub fn p023() -> String {
    let cap = 28124;

    let abundant: Vec<u64> = (1 .. cap)
        .filter(|n| {
            let bi = BigInt::from(*n);
            &numerics::sum_divisors(bi.clone()) - &bi > bi
        }).collect();

    let mut sums: HashSet<u64> = (1 .. cap).collect();

    for i in 0 .. abundant.len() {
        let a = abundant.get(i).unwrap();
        for j in 0 .. i+1 {
            let b = abundant.get(j).unwrap();

            if a + b >= cap {
                break;
            }

            sums.remove(&(a+b));
        }
    }

    let sum: u64 = sums.iter().sum();
    sum.to_string()
}

pub fn p024() -> String {
    let digits = (0 .. 10).collect::<Vec<i64>>();
    toys::nth_permutation(&digits, 1_000_000 -1) // -1 from their counting from 1
        .into_iter().map(|n| *n)
        .fold(String::new(), |mut s, d| { s.push_str(&d.to_string()); s })
}

pub fn p025() -> String {
    let cap = num::pow(BigInt::from(10), 999); // >= this means 1000 digits

    let mut last = BigInt::from(1);
    let mut curr = BigInt::from(1);

    let mut index = 2;

    while curr < cap {
        index += 1;
        let temp = curr;
        curr = &temp + &last;
        last = temp;
    }

    index.to_string()
}

pub fn p026() -> String {
    fn num_repeating_digits(den: i64) -> i64 {
        let mut seen = Vec::new();

        let mut rem = 1;
        loop {
            rem = (rem * 10) % den;
            if rem == 0 {
                return 0;
            } else if seen.contains(&rem) {
                return (seen.len() - seen.iter().position(|n| *n == rem).unwrap()) as i64
            } else {
                seen.push(rem);
            }
        }
    }

    let mut max_rep = -1;
    let mut max_den = -1;

    for d in 1 .. 1_000 {
        let rep = num_repeating_digits(d);
        if rep > max_rep {
            max_rep = rep;
            max_den = d;
        }
    }

    max_den.to_string()
}

pub fn p027() -> String {
    let cap = 5_000_000 as i64; // picked abitrarily; panics if it was too low
    let primes: HashSet<i64> = numerics::all_primes(cap as usize).into_iter()
        .map(|p| p as i64).collect();

    let is_prime = |n: i64| {
        if n >= cap {
            panic!("Cannot determine primality of {} because cap {} is too low", n, cap);
        } else {
            primes.contains(&n)
        }
    };

    let mut best_count = -1;
    let mut best_prod = -1;

    for a in -999 .. 1000 {
        for b in -999 .. 1000 {
            let poly_fn = |n| Some(n*n + a*n + b);
            let count = itertools::unfold(0 as i64, |state|
                {
                    let out = poly_fn(*state);
                    *state += 1;
                    out
                })
                .take_while(|poly| is_prime(*poly))
                .count() as i64;

            if count > best_count {
                best_count = count;
                best_prod = a*b;
            }
        }
    }

    best_prod.to_string()
}

pub fn p028() -> String {
    let mut total = 1 as u64;

    let mut row = 1;
    let mut last = 1;

    let max_diam = 1_001; // diameter of the spiral
    // NB: you could easily get a closed form for this but this is already <1ms
    while 2 * row + 1 <= max_diam {
        for _ in 0 .. 4 {
            last += 2 * row;
            total += last;
        }

        row += 1;
    }

    total.to_string()
}

pub fn p029() -> String {
    let mut distinct = HashSet::new();

    for a in 2 .. 101 {
        for b in 2 .. 101 {
            distinct.insert(num::pow(BigInt::from(a), b));
        }
    }

    distinct.len().to_string()
}

pub fn p030() -> String {
    // a 7 digit number has digit-power-sum at most 9^5 * 7 < 1,000,000
    // so all solutions are <= 6 digits

    (10 .. 1_000_000 as i64)
        .filter(|n| {
            let mut total = 0;
            let mut n_copy = *n;
            while n_copy > 0 {
                let digit = n_copy % 10;
                total += digit * digit * digit * digit * digit;
                n_copy /= 10;
            }
            total == *n
        }).sum::<i64>()
        .to_string()
}

pub fn p031() -> String {
    // NB: depending on arguments, this could cause stack overflow. Would have to loopify then
    fn coin_count(cents: u32, coins: &[u32]) -> BigInt {
        if coins.len() == 0 {
            panic!();
        } else if cents == 0 || coins.len() == 1 {
            BigInt::from(1)
        } else if coins[0] > cents {
            coin_count(cents, &coins[1..])
        } else {
            coin_count(cents - coins[0], coins) + coin_count(cents, &coins[1..])
        }
    }

    coin_count(200, &vec![200, 100, 50, 20, 10, 5, 2, 1]).to_str_radix(10)
}

pub fn p032() -> String {
    let digits = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

    let mut products_seen: HashSet<u64> = HashSet::new();

    let piece = |perm: &[&u64], start, fin| {
        perm[start..fin].iter().fold(0, |acc, d| acc*10 + *d)
    };

    // NB: 362880 is 9!, so this is all permutations
    for perm in (0 .. 362880).map(|n| toys::nth_permutation(&digits, n)) {
        // 99 * 99 = 9801 < 10000 so 2/2/5 doesn't work; WLoG can assume first is 3+ digit
        for fac_end in 3 .. 9 {
            let fac_a = piece(&perm, 0, fac_end);
            for prod_start in fac_end+1 .. 9 {
                let fac_b = piece(&perm, fac_end, prod_start);
                let prod = piece(&perm, prod_start, 9);

                if fac_a * fac_b == prod {
                    products_seen.insert(prod);
                }
            }
        }
    }

    products_seen.into_iter().sum::<u64>().to_string()
}

pub fn p033() -> String {
    let mut top = 1;
    let mut bot = 1;
    for n1 in 1 .. 10 {
        for n2 in 1..10 {
            for d1 in 1 .. 10 {
                for d2 in 1 .. 10 {
                    // orig: 10n1 + n2 / 10d1 + d2
                    // new: n1 / d2
                    if n1 == n2 && d1 == d2 {
                        continue;
                    } else if n2 != d1 {
                        continue; // NB: could build this into the loop but it's already <1 ms
                    } else if d2 * (10*n1 + n2) == n1 * (10*d1 + d2) {
                        //println!("{}{}/{}{} == {}/{}", n1, n2, d1, d2, n1, d2);
                        top *= 10 * n1 + n2;
                        bot *= 10 * d1 + d2;
                    }
                }
            }
        }
    }

    (bot / num::Integer::gcd(&top, &bot)).to_string()
}

pub fn p034() -> String {
    let facts: HashMap<u32, u32> = {
        fn fact(n: u32) -> u32 {
            if n <= 1 {
                1
            } else {
                n * fact(n-1)
            }
        }

        let mut out = HashMap::new();
        for i in 0..10 {
            out.insert(i, fact(i));
        }
        out
    };

    fn digit_fact_sum(mut n: u32, facts: &HashMap<u32, u32>) -> u32 {
        let mut sum = 0;
        while n > 0 {
            sum += *facts.get(&(n % 10)).unwrap();
            n /= 10;
        }
        sum
    }

    // 9! = 362880; if n < 10^{k+1} then dfs(n) < k * 9!
    // 10^7 > 8 * 9!, so if n has 8+ digits then this can't work
    // additionally all <=7 digit numbers have dfs <= 7 * 9! = 2540160
    // all numbers below 2540160 have dfs <= 2! + 6*9! = 2177282

    // we could do a lot more optimizations, e.g.:
    // - for <6 digit numbers can't contain 9s
    // - for 7 digit numbers, need at least 3 9s
    // but maybe 250ms is fast enough

    let sum: i64 = (10 .. 2_177_282 + 1)
        .filter(|n| digit_fact_sum(*n, &facts) == *n)
        //.map(|n| { println!("{}", n); n })
        .map(|n| n as i64) // cast to i64 because I don't know why u32 is big enough (it is though)
        .sum();

    sum.to_string()
}

pub fn p035() -> String {
    let cap = 1_000_000;

    let rotations = |n| {
        let mut pow = 1;
        let mut count = 1;
        while pow * 10 < n {
            pow *= 10;
            count += 1;
        }

        itertools::unfold(n, move |state| { *state = *state / pow + 10 * (*state % pow); Some(*state) })
            .take(count)
    };

    let prime_vec: Vec<_> = numerics::all_primes(cap).iter().map(|n| *n as u64).collect();
    let prime_set: HashSet<_> = prime_vec.iter().map(|n| *n).collect();

    let mut count = 0;
    for p in prime_vec {
        if rotations(p).all(|rot| prime_set.contains(&rot)) {
            count += 1;
        }
    }

    count.to_string()
}

pub fn p036() -> String {
    use euler_lib::toys::is_symmetric;

    fn digits(mut n: u64, base: u64) -> Vec<u64> {
        if base == 0 {
            panic!("The fuck you want me to do with this?");
        } else if n == 0 {
            return vec![0];
        }

        let mut out = Vec::new();

        while n > 0 {
            out.push(n % base);
            n /= base;
        }

        out
    }

    let bin_pal = |n| is_symmetric(&digits(n, 2));
    let dec_pal = |n| is_symmetric(&digits(n, 10));

    let cap = 1_000_000; // given by problem

    let mut total = 0;

    for n in 1 .. cap {
        if n % 2 == 0 || n % 10 == 0 { // no leading zeros -- optimizes!
            continue; // NB: could build this into an iterator to feel more fancy but it's not
        } else if dec_pal(n) && bin_pal(n) {
            total += n;
        }
    }

    total.to_string()
}

pub fn p037() -> String {
    let cap = 1_000_000; // picked out of thin air; panics if it was too low
    let desired = 11; // we're given that there are exactly 11; I didn't work out why

    let primes: HashSet<usize> = numerics::all_primes(cap)
        .into_iter().collect();

    let truncatable: Vec<usize> = primes.iter()
        .filter(|p| **p >= 10)
        .filter(|p_ref| { // L-trunc
            let mut p = **p_ref;
            while p >= 10 {
                p = p.to_string().chars().skip(1)
                    .fold(String::new(), |mut s, c| { s.push(c); s })
                    .parse().unwrap();

                if !primes.contains(&p) {
                    return false;
                }
            }
            true
        })
        .filter(|p_ref| { // R-trunc
            let mut p = **p_ref;
            while p > 0 {
                if primes.contains(&p) {
                    p /= 10;
                } else {
                    return false;
                }
            }
            true
        })
        .take(11)
        .map(|p| *p).collect();

    if truncatable.len() != desired {
        panic!("Missed some! Found {} but needed {}. Cap of {} was too low.",
               truncatable.len(), desired, cap);
    }

    truncatable.iter().sum::<usize>().to_string()
}

pub fn p038() -> String {
    let is_pandigital = |v: &Vec<u64>| {
        let set: HashSet<_> = v.iter().collect();
        set.len() == 9 && !set.contains(&0)
    };

    let push_digits = |mut n: u64, to_push: &mut Vec<u64>| {
        if n <= 0 {
            panic!("Not defined for zero");
        }
        let mut stack = Vec::new();
        while n > 0 {
            stack.push(n % 10);
            n /= 10;
        }
        while stack.len() > 0 {
            to_push.push(stack.pop().unwrap());
        }
    };

    let cap = 1_000_000_000 as u64; // all numbers must be above this
    let mut best = 0;

    for n in 2 .. 10 as u64 {
        for i in itertools::unfold(0, |state| { *state += 1; Some(*state) }) {
            let mut digits = Vec::new();

            for j in 1 .. n+1 {
                push_digits(i * j, &mut digits);
            }

            let val = digits.iter().fold(0, |acc, d| acc * 10 + d);
            if val > cap {
                break;
            } else if val > best && is_pandigital(&digits) {
                best = val;
            }
        }
    }

    best.to_string()
}

pub fn p039() -> String {
    let cap = 1_000;

    let mut counts = HashMap::new();

    for m in itertools::unfold(0_i64, |state| { *state += 1; Some(*state) }) {
        for n in (1 .. m).filter(|&n| (m*n) % 2 == 0 && numerics::gcd(m, n) == 1) {

            let a = m*m - n*n;
            let b = 2*m*n;
            let c = m*m + n*n;

            let base_p = a + b + c;
            if base_p > cap { // as n increases, so does the perimeter
                break;
            }

            for k in itertools::unfold(0, |state| { *state += 1; Some(*state) })
                .take_while(|&k| k*base_p <= cap)
            {
                let perimeter = k * base_p;
                // no need to check for redundancies!
                if counts.contains_key(&perimeter) {
                    *counts.get_mut(&perimeter).unwrap() += 1;
                } else {
                    counts.insert(perimeter, 1);
                }
            }
        }

        // 2m^2 + 2*m is the lowest perimeter of any triangle using m
        if 2*m*(m+1) > cap {
            break;
        }
    }

    let mut best_count = -1;
    let mut best_perim = -1;

    for (perim, count) in counts {
        if count > best_count {
            best_count = count;
            best_perim = perim;
        }
    }

    best_perim.to_string()
}

pub fn p040() -> String {
    let mut prod = 1;

    for goal_digit in vec![1, 10, 100, 1_000, 10_000, 100_000, 1_000_000] {
        let mut seen_so_far = 0;
        for n in itertools::unfold(0, |state| { *state += 1; Some(*state) }) {
            let s = n.to_string();

            let num_chars = s.chars().count();

            if seen_so_far + num_chars >= goal_digit {
                let c = s.chars().nth(goal_digit - seen_so_far - 1).unwrap();
                let int: i64 = c.to_string().parse().unwrap();
                prod *= int;
                break;
            }

            seen_so_far += num_chars;
        }
    }

    prod.to_string()
}

pub fn p041() -> String {
    // can't be 1 digit (only option is 1)
    // can't be 2, 3, 5, 6, 8, or 9 digit pandigital, since
    // digit sum is 3, 6, 15, 21, 36, 45 (respectively), all div by 3

    let digit_options: Vec<usize> = vec![7, 4]; // descending so we can stop when we find one

    fn fact(mut n: usize) -> usize {
        let mut f = 1;
        while n > 1 {
            f *= n;
            n -= 1;
        }
        f
    }

    for num_digits in digit_options {
        let digits: Vec<usize> = (1 .. num_digits+1).collect();
        let num_perms = fact(num_digits);
        for n in 0 .. num_perms {
            // start with the biggest permutation, so the first prime found is the biggest
            let perm = toys::nth_permutation(&digits, num_perms - n - 1);
            let val = perm.iter().fold(0, |acc, digit| acc * 10 + *digit);

            if numerics::is_prime(&BigInt::from(val)) {
                return val.to_string();
            }
        }
    }

    panic!("Didn't find any pandigital primes of any number of digits!");
}

pub fn p042() -> String {
    fn is_tri(n: u64) -> bool {
        let mut i = 0;
        let mut tri = 0;

        if n == 0 {
            return true;
        }

        while tri < n {
            i += 1;
            tri = (i*(i+1))/2;

            if tri == n {
                return true;
            }
        }

        return false;
    }

    let letter_scores: HashMap<char, u64> = {
        let letter_enum = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().enumerate();

        letter_enum.fold(HashMap::new(), |mut map, tup| {
            // A -> 1, B -> 2, etc.
            map.insert(tup.1, (tup.0 + 1) as u64);
            map
        })
    };

    let filename = "resources/p042.txt";
    let mut f = File::open(filename)
        .expect(&format!("File '{}' not found!", filename));

    let mut contents = String::new();

    f.read_to_string(&mut contents)
        .expect("IO issue of some kind");

    let mut count = 0;
    for token in contents.split(',') {
        let score = token.chars()
            .filter(|c| *c != '"') // strip quotes
            .map(|c| letter_scores.get(&c).unwrap())
            .sum();

        if is_tri(score) {
            count += 1;
        }
    }

    count.to_string()
}

pub fn p043() -> String {
    fn fact(mut n: usize) -> usize {
        let mut out = 1;
        while n > 1 {
            out *= n;
            n -= 1;
        }
        out
    }

    let digits: Vec<_> = (0 .. 10).collect();
    let div_reqs = vec![2, 3, 5, 7, 11, 13, 17];

    let mut total: u64 = 0;
    // skip the ones with leading zeros
    for n in fact(9) .. fact(10) {
        // shoddy iteration through pandigitals
        let perm = toys::nth_permutation(&digits, n);

        let success = div_reqs.iter().enumerate()
            .all(|(ind, div_req)| {
                let subsum = perm[ind+1 .. ind+4].iter().fold(0, |acc, dig| acc * 10 + **dig);
                subsum % div_req == 0
            });

        if success {
            let val = perm.iter().fold(0, |acc, dig| acc * 10 + **dig);
            total += val;
        }
    }

    total.to_string()
}

pub fn p044() -> String {
    use std::cmp::Ordering;

    let mut pents_seen: HashSet<i64> = HashSet::new();
    let mut pent_ind = 0;
    let mut last_pent = 0;

    let mut is_pent = |p| {
        while last_pent < p {
            pent_ind += 1;
            let next_pent = pent_ind * (3*pent_ind - 1) / 2;
            if next_pent < last_pent {
                panic!("Fuck! Overflow!");
            }
            last_pent = next_pent;
            pents_seen.insert(last_pent);
        }

        pents_seen.contains(&p)
    };

    #[derive(Eq, PartialEq, Debug)]
    struct PentState {
        pub top_ind: i64,
        pub top_pent: i64,
        pub bot_ind: i64,
        pub bot_pent: i64,
        pub diff: i64,
    }

    impl PentState {
        /// Adds the children (0 to 2) of this state to the heap.
        pub fn add_next(&self, to_visit: &mut BinaryHeap<PentState>) {
            if self.top_ind == self.bot_ind + 1 {
                // then do the special level up
                let next_top = PentState::pent(self.top_ind + 1);

                to_visit.push(PentState {
                    top_ind: self.top_ind + 1,
                    top_pent: next_top,
                    bot_ind: self.top_ind,
                    bot_pent: self.top_pent,
                    diff: next_top - self.top_pent
                });
            }

            if self.bot_ind > 1 {
                // either way, do the regular level down
                let next_bot = PentState::pent(self.bot_ind - 1);

                to_visit.push(PentState {
                    top_ind: self.top_ind,
                    top_pent: self.top_pent,
                    bot_ind: self.bot_ind - 1,
                    bot_pent: next_bot,
                    diff: self.top_pent - next_bot
                });
            }
        }

        pub fn pent(n: i64) -> i64 {
            (n * (3*n - 1)) / 2
        }

        pub fn new() -> PentState {
            PentState {
                top_ind: 2,
                top_pent: 5,
                bot_ind: 1,
                bot_pent: 1,
                diff: 4
            }
        }
    }

    impl PartialOrd<PentState> for PentState {
        fn partial_cmp(&self, other: &PentState) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for PentState {
        fn cmp(&self, other: &PentState) -> Ordering {
            if self.diff != other.diff {
                other.diff.cmp(&self.diff)
            } else if self.top_ind != other.top_ind {
                other.top_ind.cmp(&self.top_ind)
            } else if self.bot_ind != other.bot_ind {
                other.bot_ind.cmp(&self.bot_ind)
            } else {
                Ordering::Equal
            }
        }
    }

    // This is an "infinite min heap" (because of the reverse ordering above)
    // so that if X is popped off, then the immediate successor of X in the ordered space
    // is already in the heap. This is guaranteed because the children of X are always strictly
    // worse than X, and everything is the eventual descendent of PentState::new(). So literally
    // the first thing we see is the right answer :)
    let mut to_visit = BinaryHeap::new();
    to_visit.push(PentState::new());

    loop {
        let next = to_visit.pop().unwrap();

        next.add_next(&mut to_visit);

        if is_pent(next.diff) && is_pent(next.top_pent + next.bot_pent) {
            return next.diff.to_string();
        }
    }
}

pub fn p045() -> String {
    let one = BigInt::from(1);
    let two = BigInt::from(2);
    let three = BigInt::from(3);

    let mut tri_ind = BigInt::from(1);
    let mut tri = BigInt::from(1);

    let mut pen_ind = BigInt::from(1);
    let mut pen = BigInt::from(1);

    let mut hex_ind = BigInt::from(1);
    let mut hex = BigInt::from(1);

    let mut successes = 0;
    let desired_successes = 3;
    loop {
        if tri < pen || tri < hex {
            tri_ind = &tri_ind + &one;
            tri = &tri_ind * (&tri_ind + &one) / &two;
        } else if pen < tri || pen < hex {
            pen_ind = &pen_ind + &one;
            pen = &pen_ind * (&three * &pen_ind - &one) / &two;
        } else if hex < tri {
            // at this point, already established hex <= tri and tri == pen
            hex_ind = &hex_ind + &one;
            hex = &hex_ind * (&two * &hex_ind - &one);
        } else {
            successes += 1;
            if successes >= desired_successes {
                return hex.to_string()
            } else {
                tri_ind = &tri_ind + &one;
                tri = &tri_ind * (&tri_ind + &one) / &two;
            }
        }
    }
}

pub fn p046() -> String {
    let cap = 1_000_000; // picked out of the air; this function will panic if it wasn't high enough

    let primes = &numerics::all_primes(cap as usize).into_iter()
        .map(|n| n as i64).collect::<Vec<i64>>()[1..]; // skip 2 since we only want odds

    let squares: HashSet<i64> = itertools::unfold(-1, |n| { *n += 1; Some(2 * (*n) * (*n) )})
        .take_while(|n| *n <= cap)
        .collect();

    for n in itertools::unfold(1, |state| { *state += 2; Some(*state) }) {
        let mut found = false;
        for p in primes {
            if squares.contains(&(n - p)) {
                found = true;
                break;
            } else if p > &n {
                break;
            }
        }

        if !found {
            return n.to_string()
        }
    }

    panic!("Cap {} wasn't big enough!");
}

pub fn p047() -> String {
    let cap = 200_000; // picked out of the air; panics if it was too low

    let req_primes = 4; // number of required distinct prime factors for a "success"
    let req_successes = 4;

    let mut consecutive = 0;

    let sieve = numerics::num_prime_div_sieve(cap);

    for n in 2 .. cap {
        if *sieve.get(n).unwrap() >= req_primes {
            consecutive += 1;
            if consecutive >= req_successes {
                return (n - req_successes + 1).to_string()
            }
        } else {
            consecutive = 0;
        }
    }

    panic!("Cap of {} was too low! Did not find a solution!");
}

pub fn p048() -> String {
    let total = (1 .. 1_000 +1)
        .map(|n| num::pow(BigInt::from(n), n)) // n^n
        .fold(BigInt::from(0), |total, n| &total + &n); // sum

    let modulus = num::pow(BigInt::from(10), 10); // get last ten digits
    (&total % &modulus).to_string()
}

pub fn p049() -> String {
    let digit_vec = |mut n| {
        let mut out = Vec::new();
        while n > 0 {
            out.push(n % 10);
            n /= 10;
        }
        out
    };

    let is_perm = |a, b| {
        let mut a = digit_vec(a);
        let mut b = digit_vec(b);

        a.sort();
        b.sort();

        a == b
    };

    let mut seen = HashSet::new();

    let fact_4 = 24; // 4!

    // four digit primes
    let all_primes: Vec<usize> = numerics::all_primes(10_000)
        .into_iter().skip_while(|&n| n < 1_000).collect();

    let prime_set: HashSet<usize> = all_primes.iter().map(|n| *n).collect();

    for &first_prime in all_primes.iter() {
        let dv = digit_vec(first_prime);
        for perm in (0 .. fact_4).map(|n| toys::nth_permutation(&dv, n).into_iter().fold(0, |acc, &d| acc * 10 + d)) {
            if perm > first_prime && prime_set.contains(&perm) {
                let third = perm + (perm - first_prime);
                if third < 10_000 && prime_set.contains(&third) && is_perm(first_prime, third) {
                    seen.insert((first_prime, perm, third));
                }
            }
        }
    }

    seen.remove(&(1487, 4817, 8147));

    assert!(seen.len() == 1, "Should only see one option");

    for (a, b, c) in seen {
        return format!("{}{}{}", a, b, c);
    }

    cannot_happen();
}

pub fn p050() -> String {
    let cap = 1_000_000;

    let primes_vec = numerics::all_primes(cap as usize);
    let primes_set: HashSet<_> = primes_vec.iter().collect();

    let mut best_prime = 0;
    let mut best_length = 0;

    for start_ind in 0 .. primes_vec.len() {
        let mut total = *primes_vec.get(start_ind).unwrap();
        let mut length = 1;

        for end_ind in start_ind+1 .. primes_vec.len() {
            length += 1;
            total += *primes_vec.get(end_ind).unwrap();

            if total >= cap {
                break;
            } else if length > best_length && primes_set.contains(&total) {
                best_prime = total;
                best_length = length;
            }
        }
    }

    best_prime.to_string()
}
