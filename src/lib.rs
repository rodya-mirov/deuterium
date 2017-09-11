pub extern crate num;
extern crate itertools;

pub mod numerics;
pub mod iterators;
pub mod prelude;
pub mod data;

///
/// Module for the sort of "toy computations" with no real mathematical signficance, but
/// nevertheless seem to crop up repeatedly.
///
pub mod toys {
    use num::bigint::BigInt;

    ///
    /// Determines if the supplied number is a palindrome.
    ///
    /// Does so by stringifying and reversing. This is never true for negatives.
    ///
    /// # Examples
    /// ```
    /// extern crate num;
    /// use num::BigInt;
    ///
    /// # extern crate euler_lib; use euler_lib::toys::is_palindrome;
    /// # fn main() {
    /// assert!(is_palindrome(&BigInt::from(121)));
    /// assert!(!is_palindrome(&BigInt::from(123)));
    ///
    /// assert!(!is_palindrome(&BigInt::from(-121)));
    /// # }
    /// ```
    ///
    pub fn is_palindrome(n: &BigInt) -> bool {
        let forward: Vec<char> = n.to_str_radix(10).chars().collect();
        is_symmetric(&forward)
    }

    ///
    /// Determines if a given slice is symmetric (first == last, and so on).
    ///
    /// # Examples
    /// ```
    /// # extern crate euler_lib; use euler_lib::toys::is_symmetric;
    /// # fn main() {
    /// assert!(is_symmetric(&vec![1, 2, 3, 2, 1]));
    /// assert!(is_symmetric(&vec![1, 2, 2, 1]));
    /// assert!(!is_symmetric(&vec![1, 2, 3, 3, 1]));
    /// assert!(!is_symmetric(&vec![1, 2, 1, 1]));
    /// assert!(!is_symmetric(&vec![1, 2]));
    /// # }
    /// ```
    pub fn is_symmetric<T: PartialEq>(v: &[T]) -> bool {
        let length = v.len();
        for i in 0 .. length / 2 {
            if v.get(i) != v.get(length - i - 1) {
                return false;
            }
        }
        return true;
    }

    ///
    /// Compute the nth permutation of a given list.
    ///
    /// # Examples
    /// ```
    /// # extern crate euler_lib; use euler_lib::toys::nth_permutation;
    /// # fn main() {
    /// let vals = vec![0, 1, 2, 3];
    /// assert_eq!(nth_permutation(&vals, 0), vec![&0, &1, &2, &3]);
    /// assert_eq!(nth_permutation(&vals, 1), vec![&0, &1, &3, &2]);
    /// assert_eq!(nth_permutation(&vals, 3), vec![&0, &2, &3, &1]);
    /// assert_eq!(nth_permutation(&vals, 8), vec![&1, &2, &0, &3]);
    /// // and so on
    /// # }
    /// ```
    pub fn nth_permutation<'a, T>(orig: &'a Vec<T>, mut n: usize) -> Vec<&'a T> {
        fn fact(mut k: usize) -> usize {
            let mut out = 1;
            while k > 1 {
                out *= k;
                k -= 1;
            }
            out
        }

        n %= fact(orig.len());

        let mut ref_copy: Vec<&T> = orig.iter().collect();
        let mut out = Vec::new();

        while ref_copy.len() > 0 {
            let modulus = fact(ref_copy.len() - 1);
            let remove_ind = n / modulus;
            n %= modulus;

            out.push(ref_copy.remove(remove_ind));
        }

        out
    }
}
