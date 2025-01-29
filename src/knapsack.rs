use anyhow::{anyhow, Context};
use ec_linear::genome::bitstring::Bitstring;
use std::{
    fs::File,
    io::{self, BufRead},
    path::Path,
    str::FromStr,
};

use crate::item::Item;

/// Representation of a given knapsack problem.
///
/// A knapsack problem is a `capacity` along with a collection `items``,
/// each of which has a value and weight.
// We need to derive `Debug` so we can print out instances of `Knapsack`.
#[derive(Debug)]
pub struct Knapsack {
    /// The collection of items to choose from in this instance
    items: Vec<Item>,
    /// The capacity of the knapsack, i.e., the maximum total weight it can hold
    capacity: u64,
}

impl Knapsack {
    /// Construct a knapsack instance from a collection of items and a capacity.
    #[must_use]
    pub const fn new(items: Vec<Item>, capacity: u64) -> Self {
        Self { items, capacity }
    }

    /// Get the items available in this knapsack as a _slice_. This is essentially like
    /// returning a reference to the `Vec<Item>`, but doesn't expose the implementation details.
    #[must_use]
    pub fn items(&self) -> &[Item] {
        &self.items
    }

    /// Get the number of items in this knapsack instance.
    #[must_use]
    pub fn num_items(&self) -> usize {
        self.items.len()
    }

    /// Get a reference to a particular item in the knapsack. This returns `None` if the
    /// given `index` is outside the range of legal indices; it returns `Some(item)` if
    /// the `index` is legal.
    #[must_use]
    pub fn get_item(&self, index: usize) -> Option<&Item> {
        self.items.get(index)
    }

    /// Get an iterator over the items in the knapsack.
    pub fn iter(&self) -> impl Iterator<Item = &Item> {
        self.items.iter()
    }

    /// Get the capacity of the knapsack, i.e., the maximum total weight it can hold.
    #[must_use]
    pub const fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Get the value of a current set of choices for this knapsack. `choices` is a `Bitstring`
    /// indicating which `Item`s to include (1s in `choices`) and which to leave out (0s in `choices`).
    /// This is the sum of the value of all the chosen items as specified in `choices`.
    #[must_use]
    pub fn value(&self, choices: &Bitstring) -> u64 {
        self.items
            .iter()
            .zip(choices.iter())
            .filter_map(|(item, included)| included.then_some(item.value()))
            .sum()
    }

    /// Get the total weight of a current set of choices for this knapsack. `choices` is a `Bitstring`
    /// indicating which `Item`s to include (1s in `choices`) and which to leave out (0s in `choices`).
    /// This is the sum of the weights of all the chosen items as specified in `choices`.
    #[must_use]
    pub fn weight(&self, choices: &Bitstring) -> u64 {
        self.items
            .iter()
            .zip(choices.iter())
            .filter_map(|(item, included)| included.then_some(item.weight()))
            .sum()
    }

    /// Parse a knapsack instance from a text file.
    ///
    /// There are numerous different formats for instances of the knapsack problem.
    /// This follows that used by https://github.com/JorikJooken/knapsackProblemInstances/
    /// which has a format like this example:
    ///
    /// ```text
    /// 3
    /// 1 3 8
    /// 2 2 8
    /// 3 9 1
    /// 10
    /// ```
    ///
    /// - The first line is an integer `N` indicating how many items are available to choose from.
    /// - The next `N` lines are the `N` items, specified by three integers:
    ///    - The first integer in the line is the item number (1, 2, 3, etc.).
    ///    - The second integer is the value of the item.
    ///    - The third integer is the weight of the item.
    /// - The last line in the file is an integer `C` that is the capacity of the knapsack.
    ///
    /// # Errors
    ///
    /// This can fail if:
    ///    - We fail to open the file, or
    ///    - The file contents have the wrong format
    pub fn from_file_path(file_path: impl AsRef<Path>) -> anyhow::Result<Self> {
        // Open the file, returning (via the `?` operator) an error if there's a problem opening the file.
        let file = File::open(file_path.as_ref())?;
        // Create a buffered reader for this file.
        let reader = io::BufReader::new(file);
        // Create an iterator over all the lines in the file.
        let mut line_iter = reader.lines();

        // Get the first line and parse it into a `usize` for the number of items.
        let num_items = line_iter
            // Get the first item (line) from the iterator.
            // The result of `.next()` is an `Option`, with the `None` variant indicating
            // that there _was_ no next value (i.e., no line). The `Some` variant wraps
            // a `Result<String, std::io::Error>`. This will be `Some(s)` for some `String`
            // `s` if it was able to successfully read the line; it will be `Err(e)` for some
            // I/O error if there was an error reading the line.
            .next()
            // If the file was empty, `.next()` would return the `None` variant.
            // We'll turn that into a `Result::Err` variant and return it with `?`. The second
            // `?` is for the `Result` inside the `Option`, and will return that inner error
            // if there is one, leaving us with the string for that line if everything was OK.
            .ok_or_else(|| anyhow!("The input file {:?} was empty", file_path.as_ref()))??
            // Parse that string into a `usize`, returning any error with the `?` operator.
            .parse::<usize>()?;

        let mut items: Vec<Item> = Vec::with_capacity(num_items);
        for n in 0..num_items {
            // Get the next item (line) from the iterator. The error handling is essentially
            // the same as in reading `num_items` above.
            let line = line_iter.next().ok_or_else(|| anyhow!("Failed to read line {n} from the file; is the number of items on the first line correct?"))??;
            // Parse `line` into an `Item`, returning any parse error with the `?` operator.
            let item = Item::from_str(&line)
                .with_context(|| "Failed to parse line '{line}' into an `Item`.")?;
            // Add the successfully parsed `Item` to the vector of `items`.
            items.push(item);
        }

        // Ensure that we got the right number of `Item`s. This could fail if, for example, the
        // file didn't have enough lines.
        anyhow::ensure!(
            items.len() == num_items,
            "We weren't able to read {num_items} from the file, and only got {}.",
            items.len()
        );

        // Parse the knapsack capacity from the last line, similar to how we parsed the number
        // of items from the first line.
        let capacity = line_iter
            .next()
            .ok_or_else(|| anyhow!(
                "There was no capacity line in the input file {:?}\nThis might be because the number of items was set incorrectly.",
                file_path.as_ref()
            ))??
            .parse()?;

        Ok(Self { items, capacity })
    }
}

#[expect(clippy::unwrap_used, reason = ".unwrap() is reasonable in tests")]
#[cfg(test)]
mod tests {
    use super::Knapsack;
    use crate::item::Item;
    use ec_linear::genome::bitstring::Bitstring;
    use test_case::test_case;

    #[test]
    fn parse_from_file_path() {
        let knapsack = Knapsack::from_file_path("knapsacks/tiny.txt").unwrap();
        assert_eq!(knapsack.num_items(), 3);
        assert_eq!(knapsack.get_item(0), Some(&Item::new(1, 3, 8)));
        assert_eq!(knapsack.get_item(1), Some(&Item::new(2, 2, 8)));
        assert_eq!(knapsack.get_item(2), Some(&Item::new(3, 9, 1)));
        assert_eq!(knapsack.capacity(), 10);
    }

    #[test_case([false, false, false], 0; "choose no items")]
    #[test_case([false, true, false], 9; "choose one item")]
    #[test_case([true, false, true], 7; "choose two items")]
    fn test_values(choices: [bool; 3], expected_value: u64) {
        let knapsack = Knapsack::new(
            vec![Item::new(1, 5, 8), Item::new(2, 9, 6), Item::new(3, 2, 7)],
            100,
        );

        let choices = Bitstring::from_iter(choices);
        assert_eq!(knapsack.value(&choices), expected_value);
    }

    #[test_case([false, false, false], 0; "choose no items")]
    #[test_case([false, true, false], 6; "choose one item")]
    #[test_case([true, false, true], 15; "choose two items")]
    fn test_weights(choices: [bool; 3], expected_weight: u64) {
        let knapsack = Knapsack::new(
            vec![Item::new(1, 5, 8), Item::new(2, 9, 6), Item::new(3, 2, 7)],
            100,
        );

        let choices = Bitstring::from_iter(choices);
        assert_eq!(knapsack.weight(&choices), expected_weight);
    }
}
