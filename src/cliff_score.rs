#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
pub enum CliffScore {
    // The derived implementation of `PartialOrd` will use the order of the
    // variants, with the top variants "smaller" than the lower variants.
    // By placing `Overloaded` before `Score`, we ensure that `Overloaded`
    // will always be less (i.e., worse) than any `Score`, which is what
    // we want.
    #[default]
    Overloaded,
    Score(u64),
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use test_case::test_case;

    use super::CliffScore as CS;

    #[test_case(CS::Score(3), CS::Score(5), Ordering::Less; "Score 3 should be less than score 5")]
    #[test_case(CS::Score(8), CS::Score(5), Ordering::Greater; "Score 8 should be greater than score 5")]
    #[test_case(CS::Score(3), CS::Score(3), Ordering::Equal; "Score 3 should equal score 3")]
    #[test_case(CS::Score(3), CS::Overloaded, Ordering::Greater; "Score should be greater than Overloaded")]
    #[test_case(CS::Overloaded, CS::Score(0), Ordering::Less; "Overloaded should be less than Score")]
    #[test_case(CS::Overloaded, CS::Overloaded, Ordering::Equal; "Overloaded should equal Overloaded")]
    fn scores_compare_correctly(x: CS, y: CS, expected_ordering: Ordering) {
        assert_eq!(x.cmp(&y), expected_ordering);
    }
}
