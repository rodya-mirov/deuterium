extern crate time;
extern crate euler_lib;
extern crate num;
extern crate itertools;

use std::env;
use std::process;

mod problems;

fn main() {
    match parse_num(env::args()) {
        Err(msg) => {
            println!("{}", msg);
            process::exit(1);
        },
        Ok(num) => {
            do_problem(num);
        }
    }
}

fn parse_num(mut args: env::Args) -> Result<i32, String> {
    let cmd = match args.next() {
        None => return Err("Usage: [cmd] [problem_number]".to_string()),
        Some(c) => c
    };

    let num_str = match args.next() {
        None => return Err(format!("Usage: {} [problem_number]", cmd)),
        Some(n) => n
    };

    if let Some(_) = args.next() {
        // too many args
        return Err(format!("Usage: {} [problem_number]", cmd));
    }

    match num_str.parse::<i32>() {
        Err(_) => return Err(format!("Usage: {} [problem_number]", cmd)),
        Ok(n) => Ok(n)
    }
}

fn do_problem(problem_num: i32) {
    let start_time = time::now();

    let answer: String = match problem_num {
        n if n <= 50 => problems::p001_050::problem(n),
        n if n <= 100 => problems::p051_100::problem(n),

        102 => problems::p101_150::p102(),
        103 => problems::p101_150::p103(),
        104 => problems::p101_150::p104(),
        105 => problems::p101_150::p105(),
        106 => problems::p101_150::p106(),
        107 => problems::p101_150::p107(),
        108 => problems::p101_150::p108(),
        109 => problems::p101_150::p109(),
        110 => problems::p101_150::p110(),

        112 => problems::p101_150::p112(),
        113 => problems::p101_150::p113(),
        114 => problems::p101_150::p114(),
        115 => problems::p101_150::p115(),
        116 => problems::p101_150::p116(),
        117 => problems::p101_150::p117(),
        118 => problems::p101_150::p118(),
        119 => problems::p101_150::p119(),
        120 => problems::p101_150::p120(),
        121 => problems::p101_150::p121(),

        123 => problems::p101_150::p123(),
        124 => problems::p101_150::p124(),
        125 => problems::p101_150::p125(),

        129 => problems::p101_150::p129(),
        130 => problems::p101_150::p130(),

        132 => problems::p101_150::p132(),
        133 => problems::p101_150::p133(),
        134 => problems::p101_150::p134(),

        137 => problems::p101_150::p137(),

        139 => problems::p101_150::p139(),
        140 => problems::p101_150::p140(),

        145 => problems::p101_150::p145(),
        146 => problems::p101_150::p146(),
        147 => problems::p101_150::p147(),

        149 => problems::p101_150::p149(),



        164 => problems::p151_200::p164(),
        165 => problems::p151_200::p165(),

        169 => problems::p151_200::p169(),
        
        173 => problems::p151_200::p173(),
        174 => problems::p151_200::p174(),

        181 => problems::p151_200::p181(),

        188 => problems::p151_200::p188(),

        191 => problems::p151_200::p191(),



        204 => problems::p201_250::p204(),
        205 => problems::p201_250::p205(),
        206 => problems::p201_250::p206(),

        231 => problems::p201_250::p231(),

        250 => problems::p201_250::p250(),



        504 => problems::p501_550::p504(),

        523 => problems::p501_550::p523(),



        587 => problems::p551_600::p587(),

        _ => {
            println!("Problem {} is not yet implemented!", problem_num);
            process::exit(1);
        },
    };

    let end_time = time::now();

    println!("Answer: {}", answer);
    println!("Process took {:.3} ms.", micros_to_ms((end_time - start_time).num_microseconds().unwrap()));

}

fn micros_to_ms(micros: i64) -> f64 {
    (micros as f64) / 1000.0
}
