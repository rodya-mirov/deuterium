use std::collections::{HashMap};

use num::pow::{pow};

use euler_lib::numerics::{powmod};


pub fn p205() -> String {
    fn get_distr(faces: usize, num_dice: usize) -> (HashMap<usize, usize>, usize) {
        if num_dice == 0 {
            let mut out = HashMap::new();
            out.insert(0, 1);
            return (out, 1);
        }

        let (prev_distr, prev_total) = get_distr(faces, num_dice-1);
        let mut out: HashMap<usize, usize> = HashMap::new();

        for face in 1 .. (faces+1) {
            for (roll, mult) in &prev_distr {
                *out.entry(face+roll).or_insert(0) += mult;
            }
        }

        (out, prev_total * faces)
    }

    let (p_dist, p_total) = get_distr(4, 9);
    let (c_dist, c_total) = get_distr(6, 6);

    let total = p_total * c_total;
    let mut wins = 0;

    for (p_score, p_mult) in &p_dist {
        for (c_score, c_mult) in &c_dist {
            if p_score > c_score {
                wins += p_mult * c_mult;
            }
        }
    }

    format!("{:.7}", ((wins as f64) / (total as f64)))
}

pub fn p206() -> String {
    fn works(mut y: u64) -> bool {
        let mut k = 9;
        while k >= 1 {
            if y % 10 != k {
                return false;
            }
            y /= 100;
            k -= 1;
        }
        true
    }
    
    // x^2 ends in 0 so x ends in 0 so x = 10y for some y
    // y^2 ends in 9 so y ends in 3 or 7, so y=10z+3 or 10z+7 for some z
    // x^2 has 19 digits so x has 10 digits so y has 9 digits so z has 8 digits
    // and since x^2 starts with 1, so does y^2, so y^2 < 2 * 10^16 so y < 2 * 10^8
    // so z < 2 * 10^7 and that makes <10,000,000 choices for z
    // which means the rest is computationally trivial and further analysis is pointless

    let z_min = pow(10, 7);
    let z_max = 2 * pow(10, 7);

    for z in z_min .. z_max {
        let y = z*10 + 3;

        if works(y*y) {
            return (y*10).to_string();
        }

        let y = z*10 + 7;

        if works(y*y) {
            return (y*10).to_string();
        }
    }

    panic!("Did not find solution");
}



pub fn p250() -> String {
    let mult_mod = pow(10, 16);
    let cap = 250250;
    let modulus = 250;

    let mut possibilities = HashMap::new();
    possibilities.insert(0 as u64, 1 as usize);

    for i in 1 .. cap+1 {
        let mut new_poss = possibilities.clone();
        let addl = powmod(i, i, modulus);

        for (key, count) in possibilities.iter() {
            let new_key = (key+addl) % modulus;
            let new_count = new_poss.entry(new_key).or_insert(0);
            *new_count = (*new_count + count) % mult_mod;
        }

        possibilities = new_poss;
    }

    // subtract 1 because they want nonempty subsets
    (possibilities.get(&0).unwrap()-1).to_string()
}