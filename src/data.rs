use std::cmp::{Ordering};

#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone)]
pub struct SortBy<C, T> {
    pub cost: C, // cost MUST go first!
    pub data: T,
}

#[derive(Eq, PartialEq, Copy, Clone)]
pub struct RevSortBy<C, T> {
    pub cost: C, // don't reorder these fields
    pub data: T,
}

impl <C: Ord, T: Ord + Eq> Ord for RevSortBy<C, T> {
    fn cmp(&self, other: &RevSortBy<C, T>) -> Ordering {
        match other.cost.cmp(&self.cost) {
            Ordering::Less => Ordering::Less,
            Ordering::Greater => Ordering::Greater,
            Ordering::Equal => self.data.cmp(&other.data),
        }
    }
}

impl <C: PartialOrd, T: PartialEq> PartialOrd for RevSortBy<C, T> {
    fn partial_cmp(&self, other: &RevSortBy<C, T>) -> Option<Ordering> {
        other.cost.partial_cmp(&self.cost)
    }
}
