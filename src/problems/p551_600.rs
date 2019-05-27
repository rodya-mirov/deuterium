pub fn p587() -> String {
    use std::f64::consts::PI;

    let max_ratio: f64 = 0.001;

    let full_area: f64 = 1.0 - PI / 4.0;

    let mut n_int: u32 = 1;
    let mut n: f64 = 1.0; // as a float, so we don't have to constantly cast it

    loop {
        let x = (n * n + n - n * (2.0 * n).sqrt()) / (n * n + 1.0);
        let y = x / n;

        let left_area = x * y / 2.0;

        let para_area = 0.5 * (y + 1.0) * (1.0 - x);

        let theta = (1.0 - 0.5 * y * y - 0.5 * (1.0 - x) * (1.0 - x)).acos();
        let circ_area = 0.5 * theta;
        let right_area = para_area - circ_area;

        let tri_area = left_area + right_area;

        if tri_area / full_area <= max_ratio {
            return n_int.to_string();
        }

        n_int += 1;
        n += 1.0;
    }
}
