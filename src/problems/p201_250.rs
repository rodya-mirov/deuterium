use num::pow::{pow};

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