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

    let y_min = pow(10, 8);
    let y_max = y_min * 2; // actually we'll hit the winner before then

    let z_min = y_min / 10;
    let z_max = y_max / 10;

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