// use ec_core::individual::scorer::Scorer;
// use ec_linear::genome::bitstring::Bitstring;

// use crate::knapsack::{self, Knapsack};

// pub struct CliffScorer {
//     knapsack: Knapsack,
// }

// impl CliffScorer {
//     pub fn new(knapsack: Knapsack) -> Self {
//         Self { knapsack }
//     }
// }

// impl Scorer<Bitstring> for CliffScorer {
//     type Score = u64;

//     fn score(&self, genome: &Bitstring) -> Self::Score {
//         todo!()
//     }
// }

use ec_core::individual::scorer::Scorer;
use ec_linear::genome::bitstring::Bitstring;

use crate::{cliff_score::CliffScore, knapsack::Knapsack};

pub struct CliffScorer {
    knapsack: Knapsack,
}

impl CliffScorer {
    #[must_use]
    pub const fn new(knapsack: Knapsack) -> Self {
        Self { knapsack }
    }
}

impl Scorer<Bitstring> for CliffScorer {
    type Score = CliffScore;

    fn score(&self, genome: &Bitstring) -> Self::Score {
        let value = self.knapsack.value(genome);
        let weight = self.knapsack.weight(genome);
        if weight > self.knapsack.capacity() {
            CliffScore::Overloaded
        } else {
            CliffScore::Score(value)
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use test_case::test_case;

//     use ec_core::individual::scorer::Scorer;
//     use ec_linear::genome::bitstring::Bitstring;

//     use crate::{cliff::score::CliffScore, item::Item, knapsack::Knapsack};

//     use super::CliffScorer;

//     #[test_case([false, false, false], 0; "choose no items")]
//     #[test_case([false, true, false], 9; "choose one item")]
//     #[test_case([true, false, true], 7; "choose two items")]
//     fn test_choices(choices: [bool; 3], expected_value: u64) {
//         let knapsack = Knapsack::new(
//             vec![Item::new(1, 5, 8), Item::new(2, 9, 6), Item::new(3, 2, 7)],
//             100,
//         );
//         let scorer = CliffScorer::new(knapsack);

//         let choices = Bitstring::from_iter(choices);
//         assert_eq!(scorer.score(&choices), CliffScore::Score(expected_value));
//     }

//     #[test]
//     fn test_overloading() {
//         let knapsack = Knapsack::new(
//             vec![Item::new(1, 5, 8), Item::new(2, 9, 6), Item::new(3, 2, 7)],
//             10,
//         );
//         let scorer = CliffScorer::new(knapsack);

//         let choices = Bitstring::from_iter([true, true, false]);
//         assert_eq!(scorer.score(&choices), CliffScore::Overloaded);
//     }
// }
