pub fn p173() -> String {
    // tbh i'm a bit disappointed this worked (1.5 ms with --release)
    // there are a lot more optimizations that could/should be made, like
    // figuring out COUNT directly from outer_width (this is technically quadratic)
    let max_squares = 1_000_000;

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