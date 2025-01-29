use std::str::FromStr;

/// Represents an item with an id, value, and weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Item {
    id: u64,
    value: u64,
    weight: u64,
}

impl Item {
    /// Creates a new `Item`.
    ///
    /// # Parameters
    /// - `id`: The unique identifier for the item.
    /// - `value`: The value of the item.
    /// - `weight`: The weight of the item.
    ///
    /// # Returns
    /// A new `Item` instance.
    #[must_use]
    pub const fn new(id: u64, value: u64, weight: u64) -> Self {
        Self { id, value, weight }
    }

    /// Returns the id of the item.
    ///
    /// # Returns
    /// The id of the item.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Returns the value of the item.
    ///
    /// # Returns
    /// The value of the item.
    #[must_use]
    pub const fn value(&self) -> u64 {
        self.value
    }

    /// Returns the weight of the item.
    ///
    /// # Returns
    /// The weight of the item.
    #[must_use]
    pub const fn weight(&self) -> u64 {
        self.weight
    }
}

impl FromStr for Item {
    type Err = anyhow::Error;

    /// Parses a string slice to create an `Item`.
    ///
    /// # Parameters
    /// - `s`: The string slice to parse.
    ///
    /// # Returns
    /// A `Result` containing the `Item` if parsing was successful, or an error if it was not.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let values = s
            .split_ascii_whitespace()
            .map(FromStr::from_str)
            .collect::<Result<Vec<_>, _>>()?;
        anyhow::ensure!(
            values.len() == 3,
            "The item specification line '{s}' should have had 3 whitespace separated fields"
        );
        Ok(Self::new(values[0], values[1], values[2]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item_new() {
        let item = Item::new(1, 100, 50);
        assert_eq!(item.id(), 1);
        assert_eq!(item.value(), 100);
        assert_eq!(item.weight(), 50);
    }

    #[test]
    fn test_item_from_str() {
        let item_str = "1 100 50";
        let item = Item::from_str(item_str).unwrap();
        assert_eq!(item.id(), 1);
        assert_eq!(item.value(), 100);
        assert_eq!(item.weight(), 50);
    }

    #[test]
    fn test_item_from_str_invalid() {
        let item_str = "1 100";
        let result = Item::from_str(item_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_item_from_str_non_numeric() {
        let item_str = "1 abc 50";
        let result = item_str.parse::<Item>();
        assert!(result.is_err());
    }

    #[test]
    fn test_item_from_str_insufficient_fields() {
        let item_str = "1 100";
        let result = item_str.parse::<Item>();
        assert!(result.is_err());
    }

    #[test]
    fn test_item_from_str_extra_fields() {
        let item_str = "1 100 50 200";
        let result = item_str.parse::<Item>();
        assert!(result.is_err());
    }
}
