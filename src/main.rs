fn main() -> anyhow::Result<()> {
    // The returns the unit type `()` wrapped in the `Ok` variant of
    // `Result`. The lack of a semicolon (`;`) at the end of the line
    // makes this the last value in the function, which is what Rust
    // will return in the absence of an explicit `return` statement.
    Ok(())
}