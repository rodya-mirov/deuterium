pub struct ForeverRange<T, F>
    where F: Fn(T) -> Option<T>
{
    start: Option<T>,
    step: F,
}

impl <T, F> ForeverRange<T, F>
    where F: Fn(T) -> Option<T>
{
    pub fn new(start: T, step: F) -> ForeverRange<T, F> {
        ForeverRange { start: Some(start), step }
    }
}

impl <T, F> Iterator for ForeverRange<T, F>
    where F: Fn(T) -> Option<T>,
          T: Clone
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let out = self.start.clone();

        self.start = match self.start.clone() {
            None => None,
            Some(val) => (self.step)(val),
        };

        out
    }
}
