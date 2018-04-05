use std::ops::{Add, Sub, Mul, Div};
use std::cmp::{max, min, Ordering};
use std::fs::File;
use std::io::prelude::*;
use std::collections::{HashMap, HashSet, BinaryHeap};
use std::str::FromStr;

use itertools;

use num;
use num::{pow};
use num::bigint::BigInt;

use euler_lib::prelude::*;
use euler_lib::numerics;
use euler_lib::data::{RevSortBy};
use euler_lib::toys;
use euler_lib::numerics::Partial;



pub fn problem(problem_number: i32) -> String {
    match problem_number {
        51 => p051(),
        52 => p052(),
        53 => p053(),
        54 => p054(),
        55 => p055(),
        56 => p056(),
        57 => p057(),
        58 => p058(),
        59 => p059(),
        60 => p060(),
        61 => p061(),
        62 => p062(),
        63 => p063(),
        64 => p064(),
        65 => p065(),
        66 => p066(),
        67 => p067(),
        68 => p068(),
        69 => p069(),
        70 => p070(),
        71 => p071(),
        72 => p072(),
        73 => p073(),
        74 => p074(),
        75 => p075(),
        76 => p076(),
        77 => p077(),
        78 => p078(),
        79 => p079(),
        80 => p080(),
        81 => p081(),
        82 => p082(),
        83 => p083(),
        84 => p084(),
        85 => p085(),
        86 => p086(),
        87 => p087(),
        88 => p088(),
        89 => p089(),
        90 => p090(),
        91 => p091(),
        92 => p092(),
        93 => p093(),
        94 => p094(),
        95 => p095(),
        96 => p096(),
        97 => p097(),
        98 => p098(),
        99 => p099(),
        100 => p100(),

        _ => {
            panic!("Problem {} should not be passed to this module!", problem_number);
        },
    }
}

pub fn p051() -> String {
    let cap_digits = 6;
    let cap = pow(10, cap_digits);
    let primes: HashSet<usize> = numerics::all_primes(cap).into_iter().collect();

    type Digit = Option<usize>;

    #[derive(Debug, Eq, PartialEq)]
    struct Poss {
        digits: Vec<Digit>,
    }

    impl Poss {
        fn count(&self, primes: &HashSet<usize>) -> usize {
            let mut total = 0;
            for d in 0 .. 10 {
                let p = self.replace_with(d);
                if primes.contains(&p) {
                    total += 1;
                }
            }
            total
        }

        fn wins(&self, primes: &HashSet<usize>) -> Vec<usize> {
            let mut out = Vec::new();
            for d in 0 .. 10 {
                let p = self.replace_with(d);
                if primes.contains(&p) {
                    out.push(p);
                }
            }
            out
        }

        fn replace_with(&self, d: usize) -> usize {
            let mut total = 0;
            for &i in &self.digits {
                if let Some(k) = i {
                    total = 10 * total + k;
                } else if d == 0 && total == 0 {
                    return 0; // not prime; this handles the leading zero problem
                } else {
                    total = 10 * total + d;
                }
            }
            total
        }

        fn next(mut self) -> Poss {
            for i in 0 .. self.digits.len() {
                if self.digits[i] == None {
                    self.digits[i] = Some(0);
                } else if self.digits[i] == Some(9) {
                    self.digits[i] = None;
                    return self;
                } else {
                    self.digits[i] = Some(self.digits[i].unwrap() + 1);
                    return self;
                }
            }

            let mut new_digits = Vec::with_capacity(self.digits.len() + 1);
            for _ in 0 .. self.digits.len() + 1 {
                new_digits.push(Some(0));
            }
            Poss { digits: new_digits }
        }

        fn valid(&self) -> bool {
            for i in self.digits.iter() {
                if i.is_none() {
                    return true;
                }
            }
            false
        }

        fn true_next(mut self, cap_digits: usize) -> Poss {
            loop {
                self = self.next();
                if self.digits.len() > cap_digits {
                    panic!("Broke through the cap!");
                } else if self.valid() {
                    return self;
                }
            }
        }

        fn start() -> Poss {
            let digits = vec![ None ];
            Poss { digits }
        }
    }

    let mut poss = Poss::start();

    loop {
        if poss.count(&primes) >= 8 {
            return poss.wins(&primes)[0].to_string();
        }
        poss = poss.true_next(cap_digits);
    }
}

pub fn p052() -> String {
    fn digit_counts(mut n: u64) -> HashMap<u64, u64> {
        let mut out = HashMap::new();

        while n > 0 {
            *out.entry(n % 10).or_insert(0) += 1;
            n /= 10;
        }

        out
    }

    for n in itertools::unfold(0, |state| { *state += 1; Some(*state) }) {
        let dc = digit_counts(n);

        if (2 .. 7).all(|k| dc == digit_counts(k * n)) {
            return n.to_string()
        }
    }

    cannot_happen()
}

pub fn p053() -> String {
    #[derive(Eq, PartialEq, Debug, Hash, Clone, Copy)]
    enum Fact {
        Big,
        Small(u64)
    }

    impl Fact {
        fn is_big(&self) -> bool {
            match self {
                &Fact::Big => true,
                &Fact::Small(_) => false,
            }
        }

        fn small_value(&self) -> u64 {
            match self {
                &Fact::Small(n) => n,
                &Fact::Big => panic!("Can't unwrap BIG")
            }
        }
    }

    impl Add<Fact> for Fact {
        type Output = Fact;

        fn add(self, other: Fact) -> Fact {
            if self.is_big() || other.is_big() {
                Fact::Big
            } else {
                let total = self.small_value() + other.small_value();
                if total > 1_000_000 {
                    Fact::Big
                } else {
                    Fact::Small(total)
                }
            }
        }
    }

    fn fact(n: u64, k: u64, cache: &mut HashMap<(u64, u64), Fact>) -> Fact {
        let key = (n, k);
        if cache.contains_key(&key) {
            *cache.get(&key).unwrap()
        } else {
            let val =
                if n == 0 || k == 0 || k == n {
                    Fact::Small(1)
                } else {
                    fact(n-1, k, cache) + fact(n-1, k-1, cache)
                };

            cache.insert(key, val);
            *cache.get(&key).unwrap()
        }
    }

    let mut bigs = 0;
    let mut cache = HashMap::new();

    for n in 0 .. 100+1 {
        for k in 0 .. n+1 {
            if fact(n, k, &mut cache).is_big() {
                bigs += 1;
            }
        }
    }

    bigs.to_string()
}

pub fn p054() -> String {
    use std::fmt;
    use std::fmt::Display;

    #[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Copy, Clone)]
    struct Rank {
        rank: u32
    }

    impl Rank {
        fn new(c: char) -> Rank {
            let rank = match c {
                'T' => 10,
                'J' => 11,
                'Q' => 12,
                'K' => 13,
                'A' => 14,
                num => num.to_digit(10).unwrap()
            };
            Rank { rank }
        }
    }

    #[derive(Eq, PartialEq, Debug, Hash, Copy, Clone)]
    struct Suit {
        suit: char
    }

    #[derive(Eq, PartialEq, Debug, Hash, Copy, Clone)]
    struct Card {
        rank: Rank,
        suit: Suit
    }

    impl Card {
        fn new(chars: Vec<char>) -> Card {
            Card {
                rank: Rank::new(chars[0]),
                suit: Suit { suit: chars[1] }
            }
        }
    }

    impl Display for Card {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{}{}", self.rank.rank, self.suit.suit)
        }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct Hand {
        cards: Vec<Card>,
        ranks: HashMap<Rank, usize>,
        suits: HashMap<Suit, usize>,
    }

    impl Display for Hand {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{} {} {} {} {}", self.cards[0], self.cards[1], self.cards[2], self.cards[3], self.cards[4])
        }
    }

    impl Hand {
        fn new(cards: &[Card]) -> Hand {
            let mut ranks = HashMap::new();
            let mut suits = HashMap::new();

            for ref card in cards.iter() {
                *ranks.entry(card.rank).or_insert(0) += 1;
                *suits.entry(card.suit).or_insert(0) += 1;
            }

            Hand { cards: cards.to_owned(), ranks, suits }
        }

        fn str_flush(hand: &Hand) -> Option<Rank> {
            let f = Hand::flush(hand);
            let s = Hand::straight(hand);

            if f.is_some() {
                s
            } else {
                None
            }
        }

        fn four_kind(hand: &Hand) -> Option<Rank> {
            hand.ranks.iter()
                .filter_map(|(&k, &v)| if v >= 4 { Some(k) } else { None })
                .max()
        }

        fn full_house(hand: &Hand) -> Option<Rank> {
            let three_kind = Hand::three_kind(hand);
            if three_kind.is_none() {
                return None;
            }

            let pair = hand.ranks.iter()
                .filter_map(|(&k, &v)| if k != three_kind.unwrap() && v >= 2 { Some(k) } else { None })
                .nth(0);

            if pair.is_some() {
                three_kind
            } else {
                None
            }
        }

        fn flush(hand: &Hand) -> Option<Rank> {
            let flush_suit = hand.suits.iter()
                .filter_map(|(&k, &v)| if v >= 5 { Some(k) } else { None })
                .nth(0);

            if flush_suit.is_some() {
                hand.ranks.keys().map(|&r| r).max()
            } else {
                None
            }
        }

        fn straight(hand: &Hand) -> Option<Rank> {
            let mut ranks = hand.cards.iter()
                .map(|&card| card.rank)
                .collect::<HashSet<Rank>>().into_iter() // de-duping
                .collect::<Vec<Rank>>();

            if ranks.len() < 5 {
                None
            } else {
                ranks.sort();
                if ranks[4].rank - ranks[0].rank == 4 {
                    Some(ranks[4])
                } else {
                    None
                }
            }
        }

        fn three_kind(hand: &Hand) -> Option<Rank> {
            hand.ranks.iter()
                .filter_map(|(&k, &v)| if v >= 3 { Some(k) } else { None })
                .max()
        }

        fn two_pair(hand: &Hand) -> Option<Rank> {
            let pairs: Vec<Rank> = hand.ranks.iter()
                .filter_map(|(&k, &v)| if v >= 2 { Some(k) } else { None })
                .collect();

            if pairs.len() >= 2 {
                pairs.into_iter().max()
            } else {
                None
            }
        }

        fn one_pair(hand: &Hand) -> Option<Rank> {
            hand.ranks.iter()
                .filter_map(|(&k, &v)| if v >= 2 { Some(k) } else { None })
                .max()
        }

        fn high_card(hand: &Hand) -> Option<Rank> {
            hand.ranks.iter()
                .filter_map(|(&k, &v)| if v >= 1 { Some(k) } else { None })
                .max()
        }
    }

    enum WinStatus {
        One, Two, Both, Neither
    }

    #[derive(Eq, PartialEq, Debug)]
    struct Play {
        one: Hand,
        two: Hand
    }

    impl Play {
        fn new(line: &str) -> Play {
            let tokens: Vec<Card> = line.split(" ")
                .map(|token| Card::new(token.chars().collect::<Vec<char>>()))
                .collect();

            let one = Hand::new(&tokens[0..5]);
            let two = Hand::new(&tokens[5..10]);

            Play { one, two }
        }

        fn one_wins(&self) -> bool {
            let mut arr: Vec<fn(&Hand) -> Option<Rank>> = Vec::new();
            arr.push(Hand::str_flush);
            arr.push(Hand::four_kind);
            arr.push(Hand::full_house);
            arr.push(Hand::flush);
            arr.push(Hand::straight);
            arr.push(Hand::three_kind);
            arr.push(Hand::two_pair);
            arr.push(Hand::one_pair);
            arr.push(Hand::high_card);

            for &checker in &arr {
                match self.check_by(checker) {
                    WinStatus::Both => {
                        println!("One: {}", self.one);
                        println!("Two: {}", self.two);
                        panic!("Ambiguous?!")
                    },
                    WinStatus::One => return true,
                    WinStatus::Two => return false,
                    WinStatus::Neither => continue,
                }
            }

            panic!("Did not resolve things!");
        }

        fn check_by(&self, checker: fn(&Hand) -> Option<Rank>) -> WinStatus {
            let a = checker(&self.one);
            let b = checker(&self.two);

            if let Some(a_rank) = a {
                if let Some(b_rank) = b {
                    if a_rank > b_rank {
                        WinStatus::One
                    } else if a_rank == b_rank {
                        let a_hc = Hand::high_card(&self.one).unwrap();
                        let b_hc = Hand::high_card(&self.two).unwrap();

                        if a_hc == b_hc {
                            WinStatus::Both
                        } else if a_hc < b_hc {
                            WinStatus::Two
                        } else {
                            WinStatus::One
                        }
                    } else {
                        WinStatus::Two
                    }
                } else {
                    WinStatus::One
                }
            } else {
                if let Some(_) = b {
                    WinStatus::Two
                } else {
                    WinStatus::Neither
                }
            }
        }
    }

    let mut text = String::new();
    File::open("resources/p054.txt").expect("IO Error?")
        .read_to_string(&mut text).expect("IO Error?");

    text.lines().map(Play::new)
        .filter(Play::one_wins)
        .count().to_string()
}

pub fn p055() -> String {
    fn rev(mut n: BigInt) -> BigInt {
        let zero = BigInt::from(0);
        let ten = BigInt::from(10);

        let mut out = BigInt::from(0);
        while n > zero {
            out = &(&out * &ten) + &(&n % &ten);
            n = &n / &ten;
        }

        out
    }

    fn is_lychrel(mut n: BigInt) -> bool {
        // have to do at least one iteration
        n = &n + &rev(n.clone());
        for _ in 1 .. 50 {
            let r = rev(n.clone());
            if &n == &r {
                return false;
            } else {
                n = &n + &r;
            }
        }

        true
    }

    (1 .. 10_000)
        .filter(|n| is_lychrel(BigInt::from(*n)))
        .count().to_string()
}

pub fn p056() -> String {
    fn power(a: u64, b: usize) -> u64 {
        pow(BigInt::from(a), b)
            .to_str_radix(10).chars()
            .map(|c| c.to_digit(10).unwrap() as u64)
            .sum::<u64>()
    }

    let mut best = 0;

    for a in 1 .. 101 {
        for b in 1 .. 101 {
            best = max(best, power(a, b));
        }
    }

    best.to_string()
}

pub fn p057() -> String {
    let mut num = BigInt::from(1);
    let mut den = BigInt::from(1);

    let mut it = 0;
    let mut count = 0;

    while it < 1_000 {
        let old_num = num;
        let old_den = den;

        // this is a tricky way of computing convergents of sqrt(2), doesn't generalize well
        den = &old_num + &old_den;
        num = &den + &old_den;

        it += 1;

        if num.to_str_radix(10).chars().count() > den.to_str_radix(10).chars().count() {
            count += 1;
        }
    }

    count.to_string()
}

pub fn p058() -> String {
    fn is_prime(n: u64, primes: &Vec<u64>) -> bool {
        if n < 2 {
            false
        } else {
            for p in primes {
                if p*p > n {
                    return true
                } else if n % p == 0 {
                    return false
                }
            }
            panic!("Prime set too small!");
        }
    }

    let prime_cap = 500_000; // picked arbitrarily; good up to cap**2
    let primes = &numerics::all_primes(prime_cap).into_iter().map(|p| p as u64).collect();

    // skipping the first diagonal to avoid early stopping
    let mut side_length: u64 = 3;
    let mut n: u64 = 9;
    let mut prime_count = 3;
    let mut count = 5;

    while prime_count * 10 >= count {
        side_length += 2;

        for _ in 0 .. 4 {
            n = n.checked_add(side_length - 1).unwrap();
            count += 1;
            if is_prime(n, primes) {
                prime_count += 1;
            }
        }
    }

    side_length.to_string()
}

pub fn p059() -> String {
    let mut text = String::new();

    let chars = {
        let start = 'a' as u8;
        let end = 'z' as u8;

        (start .. end+1).collect::<Vec<u8>>()
    };

    fn decrypt(encrypted: &[u8], password: &[u8; 3]) -> Vec<char> {
        let mut out = Vec::with_capacity(encrypted.len());

        let mut i = 0;
        for &c in encrypted {
            out.push((c ^ password[i]) as char);
            i = (i+1) % 3;
        }

        out
    }

    fn very_plausible(text: &[char], tests: &[Vec<char>]) -> bool {
        tests.iter().all(|slice| plausible(text, slice))
    }

    fn plausible(text: &[char], test: &[char]) -> bool {
        if text.len() < test.len() {
            return false;
        }

        for i in 0 .. (text.len() - test.len() + 1) {
            if &text[i .. i+test.len()] == test {
                return true;
            }
        }
        false
    }

    File::open("resources/p059.txt").unwrap()
        .read_to_string(&mut text).unwrap();

    let encrypted: Vec<u8> = text.trim().split(",")
        .map(|s| s.trim().parse::<u8>().unwrap())
        .collect();

    let tests = vec![ vec!['t', 'h', 'e'], vec!['i', 's'], vec!['o', 'f'],
                      vec!['e'], vec!['a'], vec!['a', 'm'], vec!['i', 'n'], vec!['i', 'n', 'g']];

    let mut wins = Vec::new();
    let mut password = [0; 3];
    for &a in &chars {
        password[0] = a;
        for &b in &chars {
            password[1] = b;
            for &c in &chars {
                password[2] = c;

                let d = decrypt(&encrypted, &password);
                if very_plausible(&d, &tests) {
                    wins.push(d);
                }
            }
        }
    }

    if wins.len() < 1 {
        panic!("No solutions!");
    } else if wins.len() > 1 {
        panic!(format!("'{}' > 1 solutions!", wins.len()));
    }

    let win: Vec<char> = wins.pop().unwrap();
    win.into_iter()
        .map(|c| c as u32)
        .sum::<u32>().to_string()
}

pub fn p060() -> String {
    let prime_cap = 10_000_000 as usize;
    let primes = numerics::all_primes(prime_cap);
    let prime_set: HashSet<_> = primes.iter().map(|&p| p).collect();

    let is_prime = |n| {
        if n < prime_cap {
            return prime_set.contains(&n);
        }

        for &p in primes.iter() {
            if p * p >= n {
                return true;
            } else if n % p == 0 {
                return false;
            }
        }

        panic!("Prime cap too low!");
    };

    let concat = |a, b| {
        let mut tp = 1 as usize;
        while tp <= b {
            tp *= 10;
        }
        a * tp + b
    };

    let good_pair = |a, b| {
        is_prime(concat(a, b)) && is_prime(concat(b, a))
    };

    let mut best_sum: Option<usize> = None;
    let mut saved = HashMap::new();

    for i1 in 0 .. primes.len() as usize {
        let p1 = primes[i1];
        let s1 = p1;
        if let Some(n) = best_sum {
            if n <= s1 {
                break;
            }
        }

        saved.insert(p1, Vec::new());

        for i2 in 0 .. i1 {
            let p2 = primes[i2];
            let s2 = s1 + p2;

            if let Some(n) = best_sum {
                if n <= s2 {
                    break;
                }
            }

            if !good_pair(p1, p2) {
                continue;
            }

            saved.get_mut(&p1).unwrap().push(p2);
            let useful = saved.get(&p1).unwrap();

            for i3 in 0 .. useful.len() {
                let p3 = useful[i3];
                let s3 = s2 + p3;

                if let Some(n) = best_sum {
                    if n <= s3 {
                        break;
                    }
                }

                if !good_pair(p2, p3) {
                    continue;
                }

                for i4 in 0 .. i3 {
                    let p4 = useful[i4];
                    let s4 = s3 + p4;

                    if let Some(n) = best_sum {
                        if n <= s4 {
                            break;
                        }
                    }

                    if !good_pair(p2, p4) || !good_pair(p3, p4) {
                        continue;
                    }

                    for i5 in 0 .. i4 {
                        let p5 = useful[i5];
                        let s5 = s4 + p5;

                        if let Some(n) = best_sum {
                            if n <= s5 {
                                break;
                            }
                        }

                        if !good_pair(p2, p5) || !good_pair(p3, p5) || !good_pair(p4, p5) {
                            continue;
                        }

                        println!("Found: {} + {} + {} + {} + {} = {}", p1, p2, p3, p4, p5, s5);
                        best_sum = Some(s5);
                    }

                }
            }
        }
    }

    if let Some(n) = best_sum {
        n.to_string()
    } else {
        panic!("No solutions found!");
    }
}

pub fn p061() -> String {
    fn mapper<F>(f: F) -> HashSet<usize>
        where F: Fn(usize) -> usize
    {
        let mut out = HashSet::new();
        let mut n: usize = 1;
        let mut v: usize = f(n);

        while v < 1_000 {
            n += 1;
            v = f(n);
        }
        while v < 10_000 {
            out.insert(v);
            n += 1;
            v = f(n);
        }
        out
    }

    let tri_vec = mapper(|n| (n*(n+1))/2);
    let squ_vec = mapper(|n| n*n);
    let pen_vec = mapper(|n| (n*(3*n-1))/2);
    let hex_vec = mapper(|n| n*(2*n-1));
    let sep_vec = mapper(|n| (n*(5*n-3))/2);
    let oct_vec = mapper(|n| n*(3*n-2));

    let mut all = HashSet::new();
    for ref s in vec![&tri_vec, &squ_vec, &pen_vec, &hex_vec, &sep_vec, &oct_vec] {
        for &i in s.iter() {
            all.insert(i);
        }
    }

    let chain = |a, b| {
        a % 100 == b / 100
    };

    let works = |a, b, c, d, e, f| {
        // super inefficient but works for us today ...
        let start = vec![a, b, c, d, e, f];
        for p in 0 .. 720 {
            let perm = toys::nth_permutation(&start, p);
            if tri_vec.contains(&perm[0]) && squ_vec.contains(&perm[1])
                && pen_vec.contains(&perm[2]) && hex_vec.contains(&perm[3])
                && sep_vec.contains(&perm[4]) && oct_vec.contains(&perm[5])
            {
                return true;
            }
        }
        false
    };

    for &a in all.iter() {
        for &b in all.iter() {
            if !chain(a, b) || a == b {
                continue;
            }

            for &c in all.iter() {
                if !chain(b, c) || a == c || b == c {
                    continue;
                }

                for &d in all.iter() {
                    if !chain(c, d) || a == d || b == d || c == d {
                        continue;
                    }

                    for &e in all.iter() {
                        if !chain(d, e) || a == e || b == e || c == e || d == e {
                            continue;
                        }

                        let f = a / 100 + (e % 100) * 100;
                        if !all.contains(&f) {
                            continue;
                        }

                        if works(a, b, c, d, e, f) {
                            return (a + b + c + d + e + f).to_string();
                        }
                    }
                }
            }
        }
    }

    panic!("No solution found!");
}

pub fn p062() -> String {
    fn sorted_digits(mut n: u64) -> Vec<u64> {
        let mut out = Vec::new();
        while n > 0 {
            out.push(n % 10);
            n /= 10;
        }
        out.sort();
        out
    }

    let goal_orbit = 5;

    let mut num_digits = 0;
    let mut orbits: HashMap<Vec<u64>, Vec<u64>> = HashMap::new();

    let mut n = 1;
    loop {
        let digits = sorted_digits(pow(n, 3));

        // when we enter a new number of digits, we may be done ...
        if digits.len() > num_digits {
            let mut min_orbit = None;
            for (_, val) in orbits.iter() {
                if val.len() == goal_orbit {
                    if min_orbit.is_none() || min_orbit.unwrap() > *val.get(0).unwrap() {
                        min_orbit = Some(*val.get(0).unwrap());
                    }
                }
            }
            if min_orbit.is_some() {
                return pow(min_orbit.unwrap(), 3).to_string();
            }

            orbits.clear();
            num_digits = digits.len();
        }

        if orbits.contains_key(&digits) {
            orbits.get_mut(&digits).unwrap().push(n);
        } else {
            let mut orbit = Vec::new();
            orbit.push(n);
            orbits.insert(digits, orbit);
        }

        n += 1;
    }
}

pub fn p063() -> String {
    // fact: 10^n is always an n+1-digit number so the base is {1, ..., 9}
    let mut total = 0;

    for base in (1 .. 10).map(|b| BigInt::from(b)) {
        println!("Base: {}", base);
        let mut power = base.clone();
        let mut low_bd = BigInt::from(1);

        while power >= low_bd {
            println!("   {}", power);

            power = &power * &base;
            low_bd = &low_bd * &BigInt::from(10);

            total += 1;
        }
    }

    total.to_string()
}

pub fn p064() -> String {
    fn cycle_length(d: i64) -> usize {
        let mut seen_set = HashSet::new();
        let mut seen_vec = Vec::new();

        let mut start = Partial { d: d, a: 0, b: 1 };

        while !seen_set.contains(&start) {
            seen_set.insert(start);
            seen_vec.push(start);
            start = start.next().1;
            if start.is_dead() {
                return 0;
            }
        }

        for i in 0 .. seen_vec.len() {
            if seen_vec[i] == start {
                return seen_vec.len() - i;
            }
        }

        cannot_happen()
    }

    (1 .. 10_001).map(|d| cycle_length(d))
        .filter(|&len| len % 2 == 1)
        .count().to_string()
}

pub fn p065() -> String {
    #[derive(Debug)]
    struct Rational {
        num: BigInt,
        den: BigInt
    }

    impl Rational {
        fn from(n: u64) -> Rational {
            Rational { num: BigInt::from(n), den: BigInt::from(1) }
        }

        fn flip_add(&self, n: u64) -> Rational {
            let new_num = &self.den + &self.num * BigInt::from(n);
            let new_den = self.num.clone();

            Rational::reduce(new_num, new_den)
        }

        fn reduce(num: BigInt, den: BigInt) -> Rational {
            let g = num::integer::gcd(num.clone(), den.clone());
            Rational {
                num: &num / &g,
                den: &den / &g
            }
        }
    }

    let cap = 100;

    let mut conv: Vec<u64> = itertools::unfold(
        0,
        |state| {
            let val =
                if *state == 0 { 2 } else if *state % 3 == 2 { ((*state + 1) / 3) * 2 } else { 1 };
            *state += 1;
            Some(val)
        }).take(cap).collect();

    let mut frac = Rational::from(conv.pop().unwrap());

    while !conv.is_empty() {
        frac = frac.flip_add(conv.pop().unwrap());
    }

    frac.num.to_string().chars().map(|c| c.to_digit(10).unwrap() as u64).sum::<u64>().to_string()
}

pub fn p066() -> String {
    fn min_sol(d: BigInt) -> BigInt {
        let one = BigInt::from(1);
        for (x, y) in numerics::RootConvergentIter::new(&d) {
            if &x*&x - (&d)*&(&y*&y) == one {
                return x;
            }
        }

        cannot_happen()
    }

    let cap = 1_000;
    let squares: HashSet<_> = (1 .. cap+1).map(|n| n*n).collect();

    let mut max_x = BigInt::from(0);
    let mut max_d = 0;

    for d in (1 .. cap+1).filter(|&n| !squares.contains(&n)) {
        let x = min_sol(BigInt::from(d));
        if x > max_x {
            max_x = x;
            max_d = d;
        }
    }

    max_d.to_string()
}

pub fn p067() -> String {
    let mut text = String::new();

    File::open("resources/p067.txt").expect("Error reading file?")
        .read_to_string(&mut text).expect("Error reading file?");

    let grid = text.lines()
        .map(|line| line.split_whitespace().map(|token| token.parse::<u64>().unwrap()).collect::<Vec<u64>>())
        .collect::<Vec<Vec<u64>>>();

    let mut cache = HashMap::new();

    fn best_path(x: usize, y: usize, grid: &Vec<Vec<u64>>, cache: &mut HashMap<(usize, usize), u64>) -> u64 {
        if y == grid.len() - 1 {
            *grid.get(y).unwrap().get(x).unwrap()
        } else if cache.contains_key(&(x, y)) {
            *cache.get(&(x, y)).unwrap()
        } else {
            let left = best_path(x, y+1, grid, cache);
            let right = best_path(x+1, y+1, grid, cache);

            let val = max(left, right) + grid.get(y).unwrap().get(x).unwrap();

            cache.insert((x, y), val);
            val
        }
    };

    best_path(0, 0, &grid, &mut cache).to_string()
}

pub fn p068() -> String {
    let digits = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    #[derive(Eq, PartialEq, Debug)]
    struct NGon {
        items: [[i32; 3]; 5]
    }

    impl NGon {
        fn new(v: &[i32]) -> NGon {
            if v.len() != 10 {
                panic!("Only full n-gons!");
            }

            // 0 1 2 ; 3 2 4 ; 5 4 6 ; 7 6 8 ; 9 8 1
            NGon {
                items: [
                    [v[0], v[1], v[2]],
                    [v[3], v[2], v[4]],
                    [v[5], v[4], v[6]],
                    [v[7], v[6], v[8]],
                    [v[9], v[8], v[1]]
                ]
            }.canonical()
        }

        fn canonical(self) -> NGon {
            let mut least_ind = 0;
            for i in 1 .. self.items.len() {
                if self.items[i][0] < self.items[least_ind][0] {
                    least_ind = i;
                }
            }

            let items = self.items;
            let new_items = [
                items[(least_ind + 0) % 5],
                items[(least_ind + 1) % 5],
                items[(least_ind + 2) % 5],
                items[(least_ind + 3) % 5],
                items[(least_ind + 4) % 5],
            ];

            NGon { items: new_items }
        }

        fn to_string(&self) -> String {
            let mut out = String::new();
            for arr in self.items.iter() {
                for i in arr.iter() {
                    out.push_str(&i.to_string());
                }
            }
            out
        }
    }

    fn consistent(built: &[i32]) -> bool {
        if built.len() < 3 {
            return true;
        }

        let sum = built[0] + built[1] + built[2];

        // then try the next two n-gons
        if built.len() < 5 {
            return true;
        }

        let second = built[3] + built[2] + built[4];

        if second != sum {
            return false;
        }

        if built.len() < 7 {
            return true;
        }

        // 0 1 2 ; 3 2 4 ; 5 4 6 ; 7 6 8 ; 9 8 1
        let third = built[5] + built[4] + built[6];

        if third != sum {
            return false;
        }

        if built.len() < 9 {
            return true;
        }

        let fourth = built[7] + built[6] + built[8];

        if fourth != sum {
            return false;
        }

        if built.len() < 10 {
            return true;
        }

        let last = built[9] + built[8] + built[1];

        if last != sum {
            return false;
        }

        true
    }

    fn permutations(digits: &[i32], seen: &mut HashSet<i32>, built: &mut Vec<i32>, all: &mut Vec<NGon>) {
        if built.len() == digits.len() {
            let next = NGon::new(built);
            if all.iter().all(|ref ngon| next != **ngon) {
                all.push(next);
            }

            return;
        }

        for &d in digits.iter() {
            if seen.contains(&d) {
                continue;
            }

            seen.insert(d);
            built.push(d);

            if consistent(built) {
                permutations(digits, seen, built, all);
            }

            seen.remove(&d);
            built.pop();
        }
    }

    let mut options = Vec::new();
    permutations(&digits, &mut HashSet::new(), &mut Vec::new(), &mut options);

    for ref x in &options {
        println!("{:?}, {}", x, x.to_string());
    }

    let mut strings: Vec<String> = options.into_iter()
        .map(|ngon| ngon.to_string())
        .filter(|&ref s| s.chars().count() == 16)
        .collect();

    strings.sort();

    strings.pop().unwrap()
}

pub fn p069() -> String {
    let cap = 1_000_000;

    fn frac_max(a: (usize, usize), b: (usize, usize)) -> (usize, usize) {
        if a.0 * b.1 >= a.1 * b.0 {
            a
        } else {
            b
        }
    }

    let best = numerics::all_totient(cap +1).into_iter()
        .enumerate() // (i, phi(i))
        .fold((1, 1), frac_max);

    best.0.to_string()
}

pub fn p070() -> String {
    let cap = 10_000_000;

    fn is_perm(a: usize, b: usize) -> bool {
        digits(a) == digits(b)
    }

    fn digits(mut n: usize) -> HashMap<usize, usize> {
        let mut out = HashMap::new();

        while n > 0 {
            let d = n % 10;

            *out.entry(d).or_insert(0) += 1;

            n /= 10;
        }
        out
    }

    let phis = numerics::all_totient(cap);

    let mut best_frac = (1, 0); // acts like infinity in our comparator

    for (n, phi) in phis.into_iter().enumerate().skip(2) {
        if is_perm(n, phi) {
            let new_frac = (n, phi);
            if new_frac.0 * best_frac.1 < new_frac.1 * best_frac.0 {
                best_frac = new_frac;
            }
        }
    }

    best_frac.0.to_string()
}

pub fn p071() -> String {
    let d_cap: u64 = 1_000_000;

    let mut left = (0, 1);
    let mut right = (1, 1);

    let goal = (3, 7); // get just to the left of this

    // Farey sequence traversal, binary search
    loop {
        let mid = (left.0 + right.0, left.1 + right.1);

        if mid.1 > d_cap {
            return left.0.to_string();
        } else if mid.0 * goal.1 >= goal.0 * mid.1 { // mid >= goal
            right = mid;
        } else {
            left = mid;
        }
    }
}

pub fn p072() -> String {
    let d_cap = 1_000_000;

    numerics::all_totient(d_cap+1).into_iter()
        .skip(2) // skip 0 (irrelevant) and 1 (corresponds to 0 and 1, not "proper")
        .map(|n| n as u64) // overflow safe because it's at most d_cap^2
        .sum::<u64>().to_string()
}

pub fn p073() -> String {
    fn lt(a: &(u64, u64), b: &(u64, u64)) -> bool {
        a.0 * b.1 < b.0 * a.1
    }

    let lower_bound = (1, 3);
    let upper_bound = (1, 2);
    let d_cap = 12_000;

    let mut stack = Vec::new();
    let mut count = 0;

    // Farey sequence traversal; prune parts of the tree which are not helpful
    stack.push(((0, 1), (1, 1)));

    while let Some((left, right)) = stack.pop() {
        let mid = (left.0 + right.0, left.1 + right.1);

        if mid.1 > d_cap {
            continue;
        }

        if lt(&lower_bound, &mid) && lt(&mid, &upper_bound) {
            count += 1;
        }

        if lt(&lower_bound, &mid) {
            stack.push((left, mid));
        }

        if lt(&mid, &upper_bound) {
            stack.push((mid, right));
        }
    }

    count.to_string()
}

pub fn p074() -> String {
    let mut fact = HashMap::new();
    fact.insert(0, 1);
    for n in 1 .. 10 {
        let prev = *fact.get(&(n-1)).unwrap();
        fact.insert(n, n * prev);
    }
    let fact = fact; // immutable now

    fn df_sum(mut n: u64, fact: &HashMap<u64, u64>) -> u64 {
        if n == 0 {
            return 1;
        }

        let mut total = 0;
        while n > 0 {
            total += *fact.get(&(n % 10)).unwrap();
            n /= 10;
        }
        total
    }

    fn chain_length(n: u64, cache: &mut HashMap<u64, usize>, fact: &HashMap<u64, u64>) -> usize {
        // magic, given by the problem; these are all the cycles that exist
        // it wouldn't be hard to find them but I'm not going to bother if they just tell you
        if n == 169 || n == 363601 || n == 1454 {
            3
        } else if n == 871 || n == 45361 || n == 872 || n == 45362 {
            3
        } else if cache.contains_key(&n) {
            *cache.get(&n).unwrap()
        } else {
            let dfs = df_sum(n, fact);
            let val =
                if dfs == n {
                    1
                } else {
                    1 + chain_length(dfs, cache, fact)
                };
            cache.insert(n, val);
            val
        }
    }

    let mut cache = HashMap::new();
    let mut total = 0;
    let goal_num = 60; // for whatever reason
    for n in 1 .. 1_000_000 {
        if chain_length(n, &mut cache, &fact) == goal_num {
            total += 1;
        }
    }
    total.to_string()
}

pub fn p075() -> String {
    let l_max = 1_500_000;

    let mut peri_counts = HashMap::new();

    let mut m = 1;
    while 2*m*(m+1) <= l_max {

        let start = 1 + (m%2);
        let mut n = start;

        while 2*m*(m+n) <= l_max && n < m {
            if numerics::gcd(m, n) == 1 {
                let mut k = 1;
                loop {
                    let a = k * (m * m - n * n);
                    let b = k * 2 * m * n;
                    let c = k * (m * m + n * n);

                    let peri = a + b + c;
                    if peri > l_max {
                        break;
                    }

                    *peri_counts.entry(peri).or_insert(0) += 1;

                    k += 1;
                }
            }

            n += 2;
        }

        m += 1;
    }

    peri_counts.into_iter()
        .filter_map(|(p, count)| if count == 1 { Some(p) } else { None })
        .count()
        .to_string()
}

pub fn p076() -> String {
    fn partitions<'a>(n: u64, part_cap: u64, cache: &'a mut HashMap<(u64, u64), BigInt>) -> &'a BigInt {
        if n < part_cap {
            return partitions(n, n, cache);
        }

        let key = (n, part_cap);
        if cache.contains_key(&key) {
            cache.get(&key).unwrap()
        } else {
            let val =
                if n == 0 || n == 1 || part_cap == 1 {
                    BigInt::from(1)
                } else {
                    let mut acc = BigInt::from(0);
                    for piece in 1 .. part_cap+1 {
                        acc = &acc + partitions(n - piece, piece, cache);
                    }
                    acc
                };

            cache.insert(key, val);
            cache.get(&key).unwrap()
        }
    }

    let mut cache = HashMap::new();
    // we subtract 1 because the question asks for "proper partitions"
    (partitions(100, 100, &mut cache) - &BigInt::from(1)).to_string()
}

pub fn p077() -> String {
    let prime_cap = 1_000_000;

    let primes = numerics::all_primes(prime_cap);

    fn prime_summations(n: usize, ind_cap: usize, primes: &[usize], cache: &mut HashMap<(usize, usize), usize>) -> usize {
        if ind_cap == 0 {
            0
        } else if n == 0 {
            1
        } else if ind_cap == 1 {
            if n % primes[0] == 0 {
                1
            } else {
                0
            }
        } else {
            let key = (n, ind_cap);

            if cache.contains_key(&key) {
                *cache.get(&key).unwrap()
            } else {
                let mut val = 0;

                for i in 0 .. ind_cap {
                    if n >= primes[i] {
                        val += prime_summations(n - primes[i], i+1, primes, cache);
                    }
                }

                cache.insert(key, val);
                val
            }
        }
    }

    let mut cache = HashMap::new();
    let target = 5_000;

    for n in 2 .. prime_cap {
        let ps = prime_summations(n, primes.len(), &primes, &mut cache);
        if ps >= target {
            return n.to_string()
        }
    }

    panic!("Cap was too low!");
}

pub fn p078() -> String {
    fn parts(n: i64, modulus: i64, cache: &mut HashMap<(i64, i64), i64>) -> i64 {
        let key = (n, modulus);
        if n == 1 || n == 0 {
            1
        } else if cache.contains_key(&key) {
            *cache.get(&key).unwrap()
        } else {
            let mut total = 0;

            let mut k = 1;

            let mut sign = 1;
            let mut sign_counter = 0;

            loop {
                let pent = k * (3*k - 1) / 2;
                if pent > n {
                    break;
                }

                total = (total + sign * parts(n - pent, modulus, cache)) % modulus;

                // 1, -1, 2, -2, 3, -3, ...
                if k > 0 { k = -k; } else { k = 1-k; }

                if sign_counter == 0 {
                    sign_counter = 1;
                } else {
                    sign_counter = 0;
                    sign = -sign;
                }
            }

            cache.insert(key, total);
            total
        }
    }

    let modulus = 1_000_000;

    let mut cache = HashMap::new();

    itertools::unfold(0, |state| { *state += 1; Some(*state) })
        .filter_map(|n| if parts(n, modulus, &mut cache) == 0 { Some(n) } else { None })
        .next().unwrap().to_string()
}

pub fn p079() -> String {
    let mut text = String::new();
    File::open("resources/p079.txt").expect("IO Error?")
        .read_to_string(&mut text).expect("IO Error?");

    fn next(password: &mut Vec<u8>) {
        let mut i = 0;
        while i < password.len() {
            password[i] += 1;
            if password[i] < 10 {
                return;
            } else {
                password[i] = 0;
                i += 1;
            }
        }
        password.push(0);
    }

    let tests = text.lines()
        .map(|line| line.chars().map(|c| c.to_digit(10).unwrap() as u8).collect::<Vec<u8>>())
        .collect::<Vec<Vec<u8>>>();

    fn fits(password: &[u8], test: &[u8]) -> bool {
        if password.len() < test.len() {
            return false;
        } else if test.len() == 0 {
            return true;
        }

        let mut j = 0;
        for i in 0 .. password.len() {
            if password[i] == test[j] {
                j += 1;
                if j >= test.len() {
                    return true;
                }
            }
        }

        false
    }

    let mut password = vec![];

    loop {
        next(&mut password);

        if tests.iter().all(|test| fits(&password, test)) {
            return password.iter()
                .fold(String::new(), |mut s, &d| { s.push_str(&d.to_string()); s });
        }
    }
}

pub fn p080() -> String {
    fn sqrt_ish(n: usize, digits: usize) -> BigInt {
        let square = &BigInt::from(n) * &pow(BigInt::from(10), digits * 2);

        let one = BigInt::from(1);
        let two = BigInt::from(2);

        let mut low = BigInt::from(1);
        let mut high = square.clone();

        while &low + &one < high {
            let mid = &(&low + &high) / &two;

            if &mid * &mid > square {
                high = mid;
            } else {
                low = mid;
            }
        }

        low
    }

    fn digit_sum(n: BigInt) -> usize {
        n.to_str_radix(10).chars()
            .map(|c| c.to_digit(10).unwrap() as usize)
            .sum::<usize>()
    }

    let squares = (1 .. 11).map(|n| n*n).collect::<HashSet<usize>>();

    (1 .. 101).filter(|&n| !squares.contains(&n))
        .map(|n| sqrt_ish(n, 99))
        .map(|s| digit_sum(s))
        .sum::<usize>()
        .to_string()
}

pub fn p081() -> String {
    fn best_path(x: usize, y: usize, cache: &mut HashMap<(usize, usize), u64>, grid: &Vec<Vec<u64>>) -> u64 {
        let key = (x, y);
        if cache.contains_key(&key) {
            *cache.get(&key).unwrap()
        } else {
            let curr = *grid.get(y).unwrap().get(x).unwrap();
            let val =
                if y == grid.len() - 1 {
                    if x == grid.get(y).unwrap().len() - 1 {
                        curr
                    } else {
                        curr + best_path(x+1, y, cache, grid)
                    }
                } else {
                    if x == grid.get(y).unwrap().len() - 1 {
                        curr + best_path(x, y+1, cache, grid)
                    } else {
                        curr + min(best_path(x+1, y, cache, grid), best_path(x, y+1, cache, grid))
                    }
                };

            cache.insert(key, val);
            val
        }
    }

    let mut text = String::new();
    File::open("resources/p081.txt").expect("IO Error?")
        .read_to_string(&mut text).expect("IO Error?");

    let grid: Vec<Vec<u64>> = text.lines()
        .map(|line|
            line.split(",")
                .map(|token| token.parse::<u64>().expect("Parse error?"))
                .collect())
        .collect();

    let mut cache = HashMap::new();

    best_path(0, 0, &mut cache, &grid).to_string()
}

pub fn p082() -> String {
    fn new_better(a: &BigInt, b: &Option<BigInt>) -> bool {
        b.is_none() || a < b.as_ref().unwrap()
    }

    // TODO -- invent a "rectangular array" class!

    let mut text = String::new();
    File::open("resources/p082.txt").expect("IO Error!")
        .read_to_string(&mut text).expect("IO Error!");

    let grid: Vec<Vec<BigInt>> =
        text.lines()
            .map(|line|
                line.split(",")
                    .map(|token| BigInt::from_str(token).expect("Parse error?"))
                    .collect())
            .collect();

    let width = grid.get(0).unwrap().len();
    let height = grid.len();

    let mut to_process = BinaryHeap::<RevSortBy<BigInt, (usize, usize)>>::new();

    // populate a grid of None for best-per-cell
    let mut best_by_pos: Vec<Vec<Option<BigInt>>> =
        grid.iter().map(|row| row.iter().map(|_| None).collect()).collect();

    // set up our starting squares
    for y in 0 .. grid.len() {
        to_process.push(RevSortBy { cost: (&grid[y][0]).clone(), data: (0, y) });
    }

    loop {
        let path = to_process.pop().expect("Right side is unreachable???");
        let x = path.data.0;
        let y = path.data.1;

        if x == width - 1 {
            // then we're done; by the min-heap property we win :)
            return path.cost.to_string();
        } else {
            // could move right
            let right_pos = (x+1, y);
            let right_cost = &path.cost + &grid[y][x+1];

            if new_better(&right_cost, &best_by_pos[y][x+1]) {
                best_by_pos[y][x+1] = Some(right_cost.clone());
                to_process.push(RevSortBy{ cost: right_cost, data: right_pos });
            }
        }

        if y > 0 {
            // then up is possible
            let up_pos = (x, y-1);
            let up_cost = &path.cost + &grid[y-1][x];

            if new_better(&up_cost, &best_by_pos[y-1][x]) {
                best_by_pos[y-1][x] = Some(up_cost.clone());
                to_process.push(RevSortBy{ cost: up_cost, data: up_pos });
            }
        }

        if y < height - 1 {
            // then down is possible
            let down_pos = (x, y+1);
            let down_cost = &path.cost + &grid[y+1][x];

            if new_better(&down_cost, &best_by_pos[y+1][x]) {
                best_by_pos[y+1][x] = Some(down_cost.clone());
                to_process.push(RevSortBy{ cost: down_cost, data: down_pos });
            }
        }
    }
}

pub fn p083() -> String {
    fn new_better(a: &BigInt, b: &Option<BigInt>) -> bool {
        b.is_none() || a < b.as_ref().unwrap()
    }

    // TODO -- invent a "rectangular array" class!

    let mut text = String::new();
    File::open("resources/p083.txt").expect("IO Error!")
        .read_to_string(&mut text).expect("IO Error!");

    let grid: Vec<Vec<BigInt>> =
        text.lines()
            .map(|line|
                line.split(",")
                    .map(|token| BigInt::from_str(token).expect("Parse error?"))
                    .collect())
            .collect();

    let width = grid.get(0).unwrap().len();
    let height = grid.len();

    let mut to_process = BinaryHeap::<RevSortBy<BigInt, (usize, usize)>>::new();

    // populate a grid of None for best-per-cell
    let mut best_by_pos: Vec<Vec<Option<BigInt>>> =
        grid.iter().map(|row| row.iter().map(|_| None).collect()).collect();

    // set up our starting squares
    to_process.push(RevSortBy{ cost: grid[0][0].clone(), data: (0, 0) });

    loop {
        let path = to_process.pop().expect("Right side is unreachable???");
        let x = path.data.0;
        let y = path.data.1;

        if (x, y) == (width - 1, height - 1) {
            return path.cost.to_string();
        }

        if x < width - 1 {
            // could move right
            let right_pos = (x+1, y);
            let right_cost = &path.cost + &grid[y][x+1];

            if new_better(&right_cost, &best_by_pos[y][x+1]) {
                best_by_pos[y][x+1] = Some(right_cost.clone());
                to_process.push(RevSortBy{ cost: right_cost, data: right_pos });
            }
        }

        if x > 0 {
            // could move left
            let left_pos = (x-1, y);
            let left_cost = &path.cost + &grid[y][x-1];

            if new_better(&left_cost, &best_by_pos[y][x-1]) {
                best_by_pos[y][x-1] = Some(left_cost.clone());
                to_process.push(RevSortBy{ cost: left_cost, data: left_pos });
            }
        }

        if y > 0 {
            // then up is possible
            let up_pos = (x, y-1);
            let up_cost = &path.cost + &grid[y-1][x];

            if new_better(&up_cost, &best_by_pos[y-1][x]) {
                best_by_pos[y-1][x] = Some(up_cost.clone());
                to_process.push(RevSortBy{ cost: up_cost, data: up_pos });
            }
        }

        if y < height - 1 {
            // then down is possible
            let down_pos = (x, y+1);
            let down_cost = &path.cost + &grid[y+1][x];

            if new_better(&down_cost, &best_by_pos[y+1][x]) {
                best_by_pos[y+1][x] = Some(down_cost.clone());
                to_process.push(RevSortBy{ cost: down_cost, data: down_pos });
            }
        }
    }
}

pub fn p084() -> String {
    use std::ops::Deref;
    use std;

    struct State([f64; 40]);

    impl Deref for State {
        type Target = [f64; 40];

        fn deref(&self) -> &[f64; 40] {
            let &State(ref arr) = self;
            arr
        }
    }

    fn add_board(start: usize, distance: usize) -> usize {
        (start + distance) % 40
    }

    fn dice_move(mut start_state: [f64; 40], dice_size: usize) -> [f64; 40] {
        // TODO -- cleanup the repeated code (I guess?)
        let roll_prob = 1.0 / ((dice_size * dice_size) as f64); // probability of any particular roll

        // first time
        let mut next_state = start_state.clone();
        let mut first_double_probs = [0.0; 40];
        for start in 0 .. 40 {
            let to_prob = start_state[start] * roll_prob;

            for a in 1 .. dice_size+1 {
                for b in 1 .. dice_size+1 {
                    let to_ind = add_board(start, a + b);

                    if a == b {
                        first_double_probs[to_ind] += to_prob;
                    }

                    move_prob(&mut next_state, start, to_ind, to_prob);
                }
            }
        }

        // second time
        start_state = next_state;
        next_state = start_state.clone();

        let mut second_double_probs = [0.0; 40];
        for start in 0 .. 40 {
            let to_prob = first_double_probs[start] * roll_prob;

            for a in 1 .. dice_size+1 {
                for b in 1 .. dice_size+1 {
                    let to_ind = add_board(start, a + b);

                    if a == b {
                        second_double_probs[to_ind] += to_prob;
                    }

                    move_prob(&mut next_state, start, to_ind, to_prob);
                }
            }
        }

        // third time
        start_state = next_state;
        next_state = start_state.clone();

        for start in 0 .. 40 {
            let to_prob = second_double_probs[start] * roll_prob;

            for a in 1 .. dice_size+1 {
                for b in 1 .. dice_size+1 {
                    let to_ind = add_board(start, a + b);

                    if a == b {
                        move_prob(&mut next_state, start, 30, to_prob); // go to JAIL on 3rd double
                    } else {
                        move_prob(&mut next_state, start, to_ind, to_prob);
                    }
                }
            }
        }

        next_state
    }

    fn move_prob(state: &mut [f64; 40], from_ind: usize, to_ind: usize, mut amt: f64) {
        if state[from_ind] < amt - 0.000000001 {
            panic!("Cannot move {} from index {}, since it only has {} in it!", amt, from_ind, state[from_ind])
        }

        if amt > state[from_ind] {
            amt = state[from_ind];
        }

        state[from_ind] -= amt;
        state[to_ind] += amt;
    }

    fn next(start_state: State, dice_size: usize) -> State {
        let State(mut state) = start_state;

        // first handle rolling, including doubles
        state = dice_move(state, dice_size);

        // then handle chance (based on previous stopping point)
        let ch_inds = [7, 22, 36];
        for &i in ch_inds.iter() {
            let self_prob = state[i];
            let card_prob = self_prob / 16.0; // the raw probability of any particular card here

            move_prob(&mut state, i, 0, card_prob); // move to "GO"
            move_prob(&mut state, i, 10, card_prob); // go to JAIL
            move_prob(&mut state, i, 11, card_prob); // go to C1
            move_prob(&mut state, i, 24, card_prob); // go to E3
            move_prob(&mut state, i, 39, card_prob); // go to H2
            move_prob(&mut state, i, 5, card_prob); // go to R1

            let next_r = match i {
                7 => 15,
                22 => 25,
                36 => 5,
                _ => cannot_happen()
            };
            move_prob(&mut state, i, next_r, 2.0 * card_prob); // go to next railroad

            let next_u = match i {
                7 => 12,
                22 => 28,
                36 => 12,
                _ => cannot_happen()
            };
            move_prob(&mut state, i, next_u, card_prob); // go to next utility

            let back_3 = add_board(i, 37); // really (i - 3) % 40 except usize doesn't do negative well
            move_prob(&mut state, i, back_3, card_prob); // go back 3 spaces
        }

        // then handle CC (based on previous stopping point; chance can lead to CC but not vv)
        let cc_inds = [2, 17, 33];
        for &i in cc_inds.iter() {
            let self_prob = state[i];
            let card_prob = self_prob / 16.0; // the raw probability of any particular card here

            move_prob(&mut state, i, 0, card_prob); // go to GO
            move_prob(&mut state, i, 10, card_prob); // go to JAIL
        }

        // then handle go to jail
        let jail_prob = state[30];
        move_prob(&mut state, 30, 10, jail_prob);

        State(state)
    }

    let mut arr = [0.0; 40];
    arr[0] = 1.0; // start at GO
    let mut state = State(arr);

    for _ in 0 .. 40_000 {
        state = next(state, 4);
    }

    let mut inds: Vec<_> = (0 .. 40).collect();
    inds.sort_by(|&a, &b| {
        // note: order reversed
        (state[b]).partial_cmp(&state[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    inds.into_iter()
        .take(3)
        .fold(String::new(), |mut acc, i| {
            if i < 10 {
                acc.push_str("0");
            }
            acc.push_str(&i.to_string());
            acc
        })
}

pub fn p085() -> String {
    fn num_rects(width: i64, height: i64, cache: &mut HashMap<(i64, i64), i64>) -> i64 {
        if width < height {
            return num_rects(height, width, cache);
        }

        // so now width >= height
        if height == 1 {
            width * (width + 1) / 2
        } else if cache.contains_key(&(width, height)) {
            *cache.get(&(width, height)).unwrap()
        } else {
            let val = num_rects(width, height - 1, cache) + num_rects(width, 1, cache) * height;
            cache.insert((width, height), val);
            val
        }
    }

    let cap: i64 = 2_000_000;

    let mut cache = HashMap::new();

    let mut best_nr: i64 = 0;
    let mut best_area: i64 = 0;

    for width in itertools::unfold(0, |state| { *state += 1; Some(*state) }) {
        if num_rects(width, 1, &mut cache) > cap * 2 {
            break;
        }

        for height in 1 .. width+1 {
            let nr = num_rects(width, height, &mut cache);

            if nr > cap * 2 {
                break;
            } else if (nr - cap).abs() < (best_nr - cap).abs() {
                best_nr = nr;
                best_area = width * height;
            }
        }
    }

    best_area.to_string()
}

pub fn p086() -> String {
    fn shortest_path(a: u64, b: u64, c: u64) -> u64 {
        // really the square of the shortest path
        a*a + b*b + c*c + 2*(min(a*b, min(a*c, b*c)))
    }

    let m_cap = 10_000; // just needs to be big enough (but think about overflow ~3*m_cap**2)
    let goal = 1_000_000;

    let squares: HashSet<u64> = (1 .. 3*m_cap+1).map(|k| k*k).collect();

    let mut total = 0;

    for a in 1 .. m_cap+1 {
        for b in 1 .. a+1 {
            for c in 1 .. b+1 {
                if squares.contains(&shortest_path(a, b, c)) {
                    total += 1;
                }
            }
        }

        if total >= goal {
            return a.to_string();
        }
    }

    panic!("m_cap was too low!");
}

pub fn p087() -> String {
    let cap = 50_000_000;
    let cap_root = 10_000; // must be >= sqrt(cap)

    let primes = numerics::all_primes(cap_root);

    let mut found = HashSet::new();

    for asq in primes.iter().map(|&a| a*a).take_while(|&asq| asq < cap) {
        for cub in primes.iter().map(|&b| pow(b, 3) + asq).take_while(|&cub| cub < cap) {
            for total in primes.iter().map(|&c| pow(c, 4) + cub).take_while(|&total| total < cap) {
                found.insert(total);
            }
        }
    }

    found.len().to_string()
}

pub fn p088() -> String {
    #[derive(Debug)]
    struct PartialAnswer {
        sum: u64,
        prod: u64,
        k: u64,
        max_next: u64
    }

    impl PartialAnswer {
        fn from_start(start: u64) -> PartialAnswer {
            PartialAnswer { sum: start, prod: start, k: 1, max_next: start }
        }

        fn next(&self, next: u64) -> PartialAnswer {
            if next > self.max_next {
                panic!("Error in creation!");
            }
            PartialAnswer {
                sum: self.sum + next,
                prod: self.prod * next,
                k: self.k + 1,
                max_next: next
            }
        }

        fn all_ones(&self) -> PartialAnswer {
            if self.sum > self.prod {
                panic!("Can't solve this!");
            }

            PartialAnswer {
                sum: self.prod,
                prod: self.prod,
                k: self.k + self.prod - self.sum,
                max_next: 1
            }
        }

        // every number that isn't a 1 makes prod and sum farther apart (assuming start >= 3)
        // so the quickest way to bring this to a close is to pad out with one
        fn least_finishing_k(&self) -> u64 {
            self.k + self.prod - self.sum
        }
    }

    let cap = 12_000;

    let mut queue = Vec::new();
    for n in 2 .. cap+1 {
        let next = PartialAnswer::from_start(n);
        if next.least_finishing_k() < cap {
            queue.push(next);
        }
    }

    let mut seen: u64 = 0;
    let mut best = HashMap::new();
    while !queue.is_empty() {
        seen += 1;
        let to_process = queue.pop().unwrap();

        if to_process.k > 1 && to_process.prod == to_process.sum {
            let record = best.entry(to_process.k).or_insert(to_process.prod);
            *record = min(*record, to_process.prod);
        } else {
            let ones = to_process.all_ones();
            if ones.k > 1 {
                let record = best.entry(ones.k).or_insert(ones.prod);
                *record = min(*record, ones.prod);
            }
        }

        for i in 2 .. to_process.max_next+1 {
            let next = to_process.next(i);
            if next.least_finishing_k() <= cap {
                queue.push(next);
            }
        }
    }

    println!("Saw a total of {} queue elements.", seen);

    best.values()
        .fold(HashSet::new(), |mut acc, &n| { acc.insert(n); acc })
        .iter().fold(0, |acc, &n| acc + n)
        .to_string()
}

pub fn p089() -> String {
    let mut text = String::new();
    File::open("resources/p089.txt").expect("IO Error?")
        .read_to_string(&mut text).expect("IO Error?");
    let old_romans: Vec<String> = text.lines().map(|s| s.to_string()).collect();

    let min_denoms = vec![
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
        (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
        (5, "V"), (4, "IV"), (1, "I")];

    let min_den_chars: Vec<_> = min_denoms.iter()
        .map(|&(ref c, ref s)| (*c, s.chars().collect::<Vec<char>>())).collect();

    fn min_roman(mut n: u64, denoms: &[(u64, &str)]) -> String {
        let mut out = String::new();

        for &(ref cost, ref string) in denoms {
            while n >= *cost {
                n -= *cost;
                out.push_str(string);
            }
        }

        out
    }

    fn starts_with(s: &[char], c: &[char]) -> bool {
        s.len() >= c.len() && (0 .. c.len()).all(|i| s[i] == c[i])
    }

    fn to_num(mut roman: &[char], min_den_chars: &Vec<(u64, Vec<char>)>) -> u64 {
        let mut total = 0;

        for &(ref cost, ref chars) in min_den_chars {
            while starts_with(roman, chars) {
                total += *cost;
                roman = &roman[chars.len()..];
            }
        }

        total
    }

    fn improvement(old_roman: &str, denoms: &[(u64, &str)], min_den_chars: &Vec<(u64, Vec<char>)>) -> usize {
        let num = to_num(&old_roman.chars().collect::<Vec<char>>(), min_den_chars);
        let fixed = min_roman(num, denoms);
        let savings = old_roman.chars().count() - fixed.chars().count();

        savings
    }

    old_romans.iter().map(|s| improvement(s, &min_denoms, &min_den_chars)).sum::<usize>().to_string()
}

pub fn p090() -> String {
    // probably overkill on all the objects but whatever, sometimes it's easier to write more code
    #[derive(Eq, PartialEq, Debug)]
    struct Cube {
        digits: HashSet<u32>,
    }

    impl Cube {
        fn can_make(&self, x: u32) -> bool {
            if x == 9 || x == 6 {
                self.digits.contains(&6) || self.digits.contains(&9)
            } else {
                self.digits.contains(&x)
            }
        }

        fn new(a: u32, b: u32, c: u32, d: u32, e: u32, f: u32) -> Cube {
            let mut digits = HashSet::with_capacity(6);
            digits.insert(a);
            digits.insert(b);
            digits.insert(c);
            digits.insert(d);
            digits.insert(e);
            digits.insert(f);
            Cube { digits }
        }
    }

    impl Ord for Cube {
        fn cmp(&self, other: &Cube) -> Ordering {
            for i in 0 .. 10 {
                if self.digits.contains(&i) {
                    if !other.digits.contains(&i) {
                        return Ordering::Less;
                    }
                } else if other.digits.contains(&i) {
                    return Ordering::Greater;
                }
            }

            Ordering::Equal
        }
    }

    impl PartialOrd for Cube {
        fn partial_cmp(&self, other: &Cube) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    #[derive(Debug)]
    struct Pair<'a> {
        left: &'a Cube,
        right: &'a Cube,
    }

    impl <'a> Pair<'a> {
        fn works(&self) -> bool {
            fn rev((x, y): (u32, u32)) -> (u32, u32) {
                (y, x)
            }

            for pair in vec![(0, 1), (0, 4), (0, 9), (1, 6), (2, 5), (3, 6), (4, 9), (6, 4), (8, 1)] {
                if !self.can_fit(pair) && !self.can_fit(rev(pair)) {
                    return false;
                }
            }

            return true;
        }

        fn can_fit(&self, (x, y): (u32, u32)) -> bool {
            self.left.can_make(x) && self.right.can_make(y)
        }
    }

    let all_cubes = {
        let mut cubes = Vec::with_capacity(210);

        for a in 0 .. 10 {
            for b in a+1 .. 10 {
                for c in b+1 .. 10 {
                    for d in c+1 .. 10 {
                        for e in d+1 .. 10 {
                            for f in e+1 .. 10 {
                                cubes.push(Cube::new(a, b, c, d, e, f));
                            }
                        }
                    }
                }
            }
        }

        cubes
    };

    let mut total = 0;
    for left in all_cubes.iter() {
        for right in all_cubes.iter() {
            if left > right {
                continue;
            }

            let pair = Pair { left, right };
            if pair.works() {
                total += 1;
            }
        }
    }

    total.to_string()
}

pub fn p091() -> String {
    let cap = 50;

    // just the ones that have the major vertex on the origin; pretty simple
    let mut total = cap * cap;

    // we could be way smarter but cap is really small, so we can be dumb
    // assume (x1,y1) is the major vertex
    for x1 in 0 .. cap+1 {
        let start1 =
            if x1 == 0 { 1 } else { 0 };
        for y1 in start1 .. cap+1 {

            for x2 in 0 .. cap+1 {
                let start2 =
                    if x2 == 0 { 1 } else { 0 };
                for y2 in start2 .. cap+1 {

                    if x1 == x2 && y1 == y2 {
                        continue;
                    }

                    if y1 * (y2 - y1) == -x1 * (x2 - x1) {
                        total += 1;
                    }
                }
            }
        }
    }

    total.to_string()
}

pub fn p092() -> String {
    fn sds(mut n: u64) -> u64 {
        let mut total = 0;
        while n > 0 {
            total += pow(n % 10, 2);
            n /= 10;
        }
        total
    }

    fn to_89(n: u64, cache: &mut HashMap<u64, bool>) -> bool {
        if n == 1 {
            false
        } else if n == 89 {
            true
        } else if cache.contains_key(&n) {
            *cache.get(&n).unwrap()
        } else {
            let val = to_89(sds(n), cache);
            cache.insert(n, val);
            val
        }
    }

    let mut cache = HashMap::new();

    (1 .. 10_000_000).filter(|&n| to_89(n, &mut cache)).count().to_string()
}

pub fn p093() -> String {
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct Rational {
        num: i64,
        den: i64,
    }

    impl Rational {
        fn from(n: i64) -> Rational {
            Rational { num: n, den: 1 }
        }

        fn from_nd(mut n: i64, mut d: i64) -> Rational {
            if d < 0 {
                n = -n;
                d = -d;
            }
            let g = numerics::gcd(n, d);
            Rational { num: n/g, den: d/g }
        }

        fn ok(n: i64) -> MaybeFrac {
            Ok(Rational::from(n))
        }
    }

    impl Add<Rational> for Rational {
        type Output = Rational;

        fn add(self, rhs: Rational) -> Rational {
            let num = self.num * rhs.den + self.den * rhs.num;
            let den = self.den * rhs.den;

            Rational::from_nd(num, den)
        }
    }

    impl Sub<Rational> for Rational {
        type Output = Rational;

        fn sub(self, rhs: Rational) -> Rational {
            let num = self.num * rhs.den - self.den * rhs.num;
            let den = self.den * rhs.den;

            Rational::from_nd(num, den)
        }
    }

    impl Mul<Rational> for Rational {
        type Output = Rational;

        fn mul(self, rhs: Rational) -> Rational {
            let num = self.num * rhs.num;
            let den = self.den * rhs.den;

            Rational::from_nd(num, den)
        }
    }

    impl Div<Rational> for Rational {
        type Output = MaybeFrac;

        fn div(self, rhs: Rational) -> MaybeFrac {
            if rhs.num == 0 {
                Err(())
            } else {
                let num = self.num * rhs.den;
                let den = self.den * rhs.num;

                Ok(Rational::from_nd(num, den))
            }
        }
    }

    type MaybeFrac = Result<Rational, ()>;
    type OpType = Fn(MaybeFrac, MaybeFrac) -> MaybeFrac;
    type Op = Box<OpType>;

    // forms: a * (b * (c * d))
    //        a * ((b * c) * d)
    //        (a * c) * (b * d)
    //        (a * (b * c)) * d
    //        ((a * b) * c) * d
    enum Form { A, B, C, D, E }

    impl Form {
        fn forms() -> Vec<Form> {
            vec![Form::A, Form::B, Form::C, Form::D, Form::E]
        }
    }

    fn apply(nums: &Vec<MaybeFrac>, (ref f, ref g, ref h): (&Op, &Op, &Op), form: Form) -> MaybeFrac {
        match form {
            Form::A => f(nums[0], g(nums[1], h(nums[2], nums[3]))),
            Form::B => f(nums[0], g(h(nums[1], nums[2]), nums[3])),
            Form::C => f(g(nums[0], nums[1]), h(nums[2], nums[3])),
            Form::D => f(g(nums[0], h(nums[1], nums[2])), nums[3]),
            Form::E => f(g(h(nums[0], nums[1]), nums[2]), nums[3]),
        }
    }

    fn run(nums: &HashSet<i64>) -> i64 {
        let mut i = 1;
        while nums.contains(&i) {
            i += 1;
        }
        i - 1
    }

    let ops: Vec<Op> = vec![
        Box::new(|a, b| Ok(a? + b?)),
        Box::new(|a, b| Ok(a? - b?)),
        Box::new(|a, b| Ok(a? * b?)),
        Box::new(|a, b| a? / b?),
    ];

    let triples = &{
        let mut out = Vec::with_capacity(pow(ops.len(), 3));
        for a in &ops {
            for b in &ops {
                for c in &ops {
                    out.push((a, b, c));
                }
            }
        }
        out
    };

    fn all_results(v: &Vec<MaybeFrac>, triples: &Vec<(&Op, &Op, &Op)>) -> i64 {
        let mut set = HashSet::new();
        for i in 0 .. 24 {
            let perm = toys::nth_permutation(v, i).iter().map(|&&n| n).collect();
            for triple in triples {
                for form in Form::forms() {
                    if let Ok(frac) = apply(&perm, *triple, form) {
                        if frac.den == 1 && frac.num >= 0 {
                            set.insert(frac.num);
                        }
                    }
                }
            }
        }
        run(&set)
    };

    let mut best_str = 0;
    let mut best = -1;
    for a in 0 .. 10 {
        for b in a+1 .. 10 {
            for c in b+1 .. 10 {
                for d in c+1 .. 10 {
                    let digits = vec![Rational::ok(a), Rational::ok(b), Rational::ok(c), Rational::ok(d)];
                    let res = all_results(&digits, triples);
                    if res > best {
                        best = res;
                        best_str = 1000 * a + 100 * b + 10 * c + d;
                    }
                }
            }
        }
    }

    best_str.to_string()
}

pub fn p094() -> String {
    let cap = 1_000_000_000;

    let mut total_perimeter: i64 = 0;

    let mut m: i64 = 1;

    while 2*m*m + 2*m <= cap {
        let mut n = 1 + (m%2);

        while n < m && 2*m*m + 2*m*n <= cap {
            // no need to check gcd because if it's not primitive it'll just fail
            let a = m*m - n*n;
            let b = 2*m*n;
            let c = m*m + n*n;

            let side = min(a, b);
            let diff = c - 2 * side;

            if diff == 1 || diff == -1 {
                println!("Victory: {} {} {}", a, b, c);
                total_perimeter += 2*(side + c);
            }

            n += 2;
        }

        m += 1;
    }

    total_perimeter.to_string()
}

pub fn p095() -> String {
    // Note: I tried a bunch of elegant caching logic but ultimately it didn't help -- the cache hit
    // rate was so low that all the time maintaining and loading the cache didn't help. Lame!
    fn cycle_length(n: usize, cap: usize, all_sd: &Vec<usize>) -> Option<usize> {
        let mut seen = HashSet::with_capacity(50);
        let mut path = Vec::new();

        let mut curr = n;

        while !seen.contains(&curr) {
            seen.insert(curr);
            path.push(curr);

            match all_sd.get(curr) {
                None => return None,
                Some(&next) => {
                    if next >= cap {
                        return None;
                    } else {
                        curr = next;
                    }
                }
            }
        }

        if curr == n {
            Some(path.len())
        } else {
            None
        }
    }

    let cap = 1_000_000;

    let all_sd = numerics::all_sum_divisors(cap).into_iter()
        .enumerate().map(|(n, t)| t-n).collect::<Vec<usize>>();

    let mut min = cap + 1;
    let mut max_cycle = 0;

    for n in 0 .. cap {
        if let Some(length) = cycle_length(n, cap, &all_sd) {
            if length > max_cycle {
                max_cycle = length;
                min = n;
            } else if length == max_cycle && n < min {
                min = n;
            }
        }
    }

    min.to_string()
}

pub fn p096() -> String {
    #[derive(Debug, Clone)]
    struct Board {
        data: [[Option<u32>; 9]; 9],
        poss: Vec<Vec<HashSet<u32>>>
    }

    impl Board {
        fn solve(mut self) -> Result<Board, ()> {
            while self.help_step() {}

            if self.is_done() {
                if self.has_troubles() {
                    return Err(());
                } else {
                    return Ok(self);
                }
            } else if self.is_inconsistent() {
                return Err(());
            } else {
                for y in 0 .. 9 {
                    for x in 0 .. 9 {
                        if self.data[y][x].is_none() {
                            for &p in self.poss[y][x].iter() {
                                let mut maybe = self.clone();
                                maybe.data[y][x] = Some(p);
                                maybe.poss[y][x].retain(|&n| n == p);
                                if let Ok(board) = maybe.solve() {
                                    return Ok(board)
                                }
                            }
                            // If we got this far then no option was correct, which means we were
                            // trying to solve an impossible configuration (which is probably the
                            // result of an earlier wrong guess)
                            return Err(());
                        }
                    }
                }
                panic!("Cannot happen!");
            }
        }

        fn has_troubles(&self) -> bool {
            for cell in &Board::cells() {
                let mut known = HashSet::new();
                for &(x, y) in cell {
                    if let Some(n) = self.data[y][x] {
                        if known.contains(&n) {
                            return true;
                        } else {
                            known.insert(n);
                        }
                    }
                }
            }
            return false;
        }

        fn from(lines: Vec<&&str>) -> Option<Board> {
            if lines.len() != 9 {
                return None;
            } else if !lines.iter().all(|&v| v.chars().count() == 9) {
                return None;
            } else {
                let mut data = [[None; 9]; 9];
                for row in 0 .. 9 {
                    for (col, c) in lines[row].chars().enumerate() {
                        data[row][col] = Some(c.to_digit(10).unwrap());
                        if data[row][col] == Some(0) {
                            data[row][col] = None;
                        }
                    }
                }

                let poss: Vec<Vec<HashSet<u32>>> = (0..9)
                    .map(|_| (0..9).map(|_| HashSet::new()).collect())
                    .collect();

                let mut out = Board { data, poss };
                out.init_known();

                Some(out)
            }
        }

        fn init_known(&mut self) {
            for row in 0 .. 9 {
                for col in 0 .. 9 {
                    if let Some(n) = self.data[row][col] {
                        self.poss[row][col].insert(n);
                    } else {
                        for i in 1 .. 10 {
                            self.poss[row][col].insert(i);
                        }
                    }
                }
            }
        }

        fn rows() -> Vec<Vec<(usize, usize)>> {
            let mut out = Vec::with_capacity(9);
            for row in 0 .. 9 {
                let mut r = Vec::with_capacity(9);
                for col in 0 .. 9 {
                    r.push((row, col));
                }
                out.push(r);
            }
            out
        }

        fn cols() -> Vec<Vec<(usize, usize)>> {
            let mut out = Vec::with_capacity(9);
            for col in 0 .. 9 {
                let mut c = Vec::with_capacity(9);
                for row in 0 .. 9 {
                    c.push((row, col));
                }
                out.push(c);
            }
            out
        }

        fn grid() -> Vec<Vec<(usize, usize)>> {
            let mut out = Vec::with_capacity(9);
            for cx in vec![0, 3, 6] {
                for cy in vec![0, 3, 6] {
                    let mut g = Vec::with_capacity(9);
                    for row in cy .. cy+3 {
                        for col in cx .. cx+3 {
                            g.push((row, col));
                        }
                    }
                    out.push(g);
                }
            }
            out
        }

        fn cells() -> Vec<Vec<(usize, usize)>> {
            let mut out = Vec::with_capacity(27);

            for cell in Board::rows() {
                out.push(cell);
            }

            for cell in Board::cols() {
                out.push(cell);
            }

            for cell in Board::grid() {
                out.push(cell);
            }

            out
        }

        fn block_singles(&mut self) -> usize {
            let mut changes = 0;

            for cell in &Board::cells() {
                let mut known = HashSet::new();
                for &(y, x) in cell.iter() {
                    if let Some(n) = self.data[y][x] {
                        known.insert(n);
                    }
                }
                for &(y, x) in cell.iter() {
                    if self.data[y][x].is_none() {
                        let old_len = self.poss[y][x].len();
                        self.poss[y][x].retain(|&n| !known.contains(&n));
                        changes += old_len - self.poss[y][x].len();
                    }
                }
            }

            changes
        }

        fn affirm_singles(&mut self) -> usize {
            let mut changes = 0;

            for cell in &Board::cells() {
                let mut counts = HashMap::new();
                for &(y, x) in cell.iter() {
                    for &p in self.poss[y][x].iter() {
                        *counts.entry(p).or_insert(0) += 1;
                    }
                }
                for &(y, x) in cell.iter() {
                    let onlys: Vec<u32> = self.poss[y][x].iter().filter(|n| counts.get(n) == Some(&1)).map(|&n| n).collect();
                    if onlys.len() > 1 {
                        changes += self.poss[y][x].len();
                        self.poss[y][x].clear();
                    } else if onlys.len() == 1 {
                        changes += self.poss[y][x].len() - 1;
                        self.poss[y][x].retain(|&n| n == onlys[0]);
                    }
                }
            }

            changes
        }

        fn singletons(&mut self) -> usize {
            let mut changes = 0;

            for col in 0 .. 9 {
                for row in 0 .. 9 {
                    if self.data[row][col].is_none() && self.poss[row][col].len() == 1 {
                        self.data[row][col] = Some(*self.poss[row][col].iter().next().unwrap());
                        changes += 1;
                    }
                }
            }

            changes
        }

        fn help_step(&mut self) -> bool {
            let mut changed = false;
            changed |= self.block_singles() > 0;
            changed |= self.affirm_singles() > 0;
            changed |= self.singletons() > 0;

            changed
        }

        fn is_done(&self) -> bool {
            for row in 0 .. 9 {
                for col in 0 .. 9 {
                    if self.data[row][col].is_none() {
                        return false;
                    }
                }
            }
            true
        }

        fn is_inconsistent(&self) -> bool {
            for row in 0 .. 9 {
                for col in 0 .. 9 {
                    if self.poss[row][col].len() == 0 {
                        return true;
                    }
                }
            }
            false
        }
    }

    let mut text = String::new();
    File::open("resources/p096.txt").expect("IO Error?")
        .read_to_string(&mut text).expect("IO Error?");

    let boards: Vec<Board> = text.lines().collect::<Vec<&str>>()
        .chunks(10)
        .map(|chunk| chunk.iter().skip(1).collect::<Vec<&&str>>())
        .filter_map(Board::from)
        .collect::<Vec<Board>>();

    let mut total = 0;
    for board in boards.iter() {
        let board = board.clone().solve().unwrap();
        total += board.data[0][0].unwrap() * 100 + board.data[0][1].unwrap() * 10 + board.data[0][2].unwrap();
    }
    total.to_string()
}

pub fn p097() -> String {
    //28433×2^7830457+1 % 10^10

    let modulo = pow(BigInt::from(10), 10);
    let power = numerics::powmod(BigInt::from(2), 7830457, modulo.clone());
    let value = BigInt::from(28433) * power + BigInt::from(1);

    (value % modulo).to_string()
}

pub fn p098() -> String {
    #[derive(Debug)]
    struct Word {
        chars: Vec<char>,
        counts: HashMap<char, usize>
    }

    impl Word {
        pub fn from(chars: Vec<char>) -> Word {
            let mut counts = HashMap::with_capacity(chars.len());

            for &c in chars.iter() {
                let entry = counts.entry(c).or_insert(0);
                *entry += 1;
            }

            Word { chars, counts }
        }
    }

    fn possible_squares(w: &Vec<char>) -> Vec<u64> {
        if w.len() == 0 {
            return Vec::new();
        } else if w.len() > 18 {
            panic!("Concerned about overflow!");
        }

        let mut out = Vec::new();
        let mut n = 1;
        let mut nsq = 1;

        let nsq_min = pow(10, w.len() - 1);
        let nsq_max = 10 * nsq_min - 1;

        while nsq < nsq_min {
            n += 1;
            nsq = n*n;
        }
        while nsq <= nsq_max {
            if maps_to(w, nsq) {
                out.push(nsq);
            }
            n += 1;
            nsq = n*n;
        }
        out
    }

    fn maps_to(w: &Vec<char>, mut n: u64) -> bool {
        // PRE: w.len() == n.to_string().chars().count()
        let mut i = w.len() - 1;
        let mut let_dig: HashMap<char, u64> = HashMap::new();
        let mut dig_let: HashMap<u64, char> = HashMap::new();

        loop {
            let d = n % 10;
            let l = w[i];

            if let_dig.contains_key(&l) {
                if *let_dig.get(&l).unwrap() != d {
                    return false;
                }
            } else {
                let_dig.insert(l, d);
            }

            if dig_let.contains_key(&d) {
                if *dig_let.get(&d).unwrap() != l {
                    return false;
                }
            } else {
                dig_let.insert(d, l);
            }

            if i == 0 {
                return true;
            } else {
                n /= 10;
                i -= 1;
            }
        }
    }

    fn is_square(n: u64) -> bool {
        if n > pow(10, 15) {
            panic!("I hate overflow considerations");
        }
        // not fast
        let mut i = 0;
        loop {
            let isq = i*i;
            if isq < n {
                i += 1;
            } else if isq == n {
                return true;
            } else {
                return false;
            }
        }
    }

    fn perm(w1: &Vec<char>, w2: &Vec<char>, mut n: u64) -> Option<u64> {
        let mut i = w1.len() - 1;
        let mut let_dig = HashMap::new();

        loop {
            let_dig.insert(w1[i], n % 10);
            n /= 10;

            if i == 0 {
                break;
            } else {
                i -= 1;
            }
        }

        if let_dig.get(&w2[0]).unwrap() == &0 {
            return None;
        }

        let mut n = 0;
        for i in 0 .. w2.len() {
            n = 10 * n + *let_dig.get(&w2[i]).unwrap();
        }

        Some(n)
    }

    let mut text = String::new();

    File::open("resources/p098.txt").expect("IO Error?")
        .read_to_string(&mut text).expect("IO Error?");

    let words: Vec<Word> = text.split(",")
        .map(|s| s.chars().filter(|&c| c != '"').collect::<Vec<char>>())
        .map(|v| Word::from(v))
        .collect();

    let mut best = 0;
    for (i, ref w1) in words.iter().enumerate() {
        for w2 in &(words[i+1..]) {
            if w1.counts != w2.counts {
                continue;
            }

            for square1 in possible_squares(&w1.chars) {
                if let Some(square2) = perm(&w1.chars, &w2.chars, square1) {
                    if is_square(square2) {
                        best = max(best, max(square1, square2));
                    }
                }
            }
        }
    }

    best.to_string()
}

pub fn p099() -> String {
    fn better(base_a: f64, pow_a: f64, base_b: f64, pow_b: f64) -> bool {
        pow_a * (base_a.ln()) > pow_b * (base_b.ln())
    }

    let mut text = String::new();

    File::open("resources/p099.txt").expect("IO Error")
        .read_to_string(&mut text).expect("IO Error");

    let mut best: Option<(f64, f64)> = None;
    let mut best_ind = None;

    for (ind, line) in text.lines().enumerate() {
        let tokens: Vec<_> = line.split(",").collect();
        assert_eq!(tokens.len(), 2);
        let base = tokens.get(0).unwrap().parse::<f64>().unwrap();
        let power = tokens.get(1).unwrap().parse::<f64>().unwrap();

        if best.is_none() || better(base, power, best.unwrap().0, best.unwrap().1) {
            best_ind = Some(ind);
            best = Some((base, power));
        }
    }

    (best_ind.unwrap()+1).to_string()
}

pub fn p100() -> String {
    let cap = BigInt::from(1_000_000_000_000 as u64);

    let one = BigInt::from(1);
    let two = BigInt::from(2);
    let three = BigInt::from(3);
    let four = BigInt::from(4);

    let mut x = BigInt::from(1);
    let mut y = BigInt::from(1);

    loop {
        let (old_x, old_y) = (x, y);
        x = &three * &old_x + &four * &old_y;
        y = &two * &old_x + &three * &old_y;

        let (a, b) = (&(&y+&one)/&two, &(&x+&one)/&two);

        if b > cap {
            return a.to_string();
        }
    }
}
