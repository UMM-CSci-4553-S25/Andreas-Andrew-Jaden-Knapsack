mod cliff_score;
mod cliff_scorer;
mod item;
mod knapsack;

use cliff_score::CliffScore;
use cliff_scorer::CliffScorer;
use course_helpers::{ec_run::Run, statistics::entropy};
use ec_core::{
    individual::ec::EcIndividual,
    operator::selector::{best::Best, tournament::Tournament, Selector},
};
use ec_linear::{
    genome::bitstring::Bitstring, mutator::with_one_over_length::WithOneOverLength,
    recombinator::uniform_xo::UniformXo,
};
use knapsack::Knapsack;
use rand::Rng;

fn report_on_generation(
    generation_number: usize,
    population: &Vec<EcIndividual<Bitstring, CliffScore>>,
    best_in_run: &mut Option<EcIndividual<Bitstring, CliffScore>>,
    rng: &mut impl Rng,
) {
    // Get the best individual in the population and print out its score.
    let best = Best.select(population, rng).unwrap();
    println!(
        "Best score in generation {generation_number} was {:?}",
        best.test_results
    );
    // Calculate the entropy of the population and print it out.
    println!("\tEntropy of the population was {}", entropy(population));
    // If the best individual in this generation is better than the best in the run so far,
    // update the best in the run.
    match best_in_run {
        // If there is no best in the run so far, set it to a clone of the best in this generation.
        None => *best_in_run = Some(best.clone()),
        // If there is a best in the run so far, and the best in this generation is better, update it.
        Some(b) if best.test_results > b.test_results => *b = best.clone(),
        // If there is a best in the run so far, and the best in this generation is not better, do nothing.
        _ => (),
    }
}

fn main() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    let file_path = "knapsacks/big.txt";
    let knapsack = Knapsack::from_file_path(file_path)?;

    let mut best_in_run = None;

    let run = Run::builder()
        // The number of bits should equal the number of items.
        .bit_length(knapsack.num_items())
        // The maximum number of generations to run; this is somewhat arbitrary
        .max_generations(1_000)
        // The population size, which is also somewhat arbitrary, but larger is better
        // until it's so big that memory management becomes a problem.
        .population_size(1_000)
        // How do we want to select parent individuals? This takes two individuals at
        // random from the population, and then chooses the better of the two from this
        // tournament. You can change this to larger tournaments by changing `2` to your
        // desired tournament size.
        .selector(Tournament::of_size::<2>())
        // How do we want to mutate individual knapsack solutions? This flips
        // on average one bit, thereby adding or removing one item from the solution.
        .mutator(WithOneOverLength)
        // How do we want to recombine parent solutions? This randomly chooses for
        // each bit whether to take it from the first or the second parent, giving
        // use a "shuffled" set of choices from both parents.
        .recombinator(UniformXo)
        // Do we want to use parallel evaluation? If this is `true`, the run will use
        // all the available cores to evaluate the population in parallel. This can speed
        // up the process considerably, at the cost of heating up your CPU.
        .parallel_evaluation(true)
        // How do we want to score different knapsack "solutions"? This is the only
        // problem dependent part of building the run. We'll start with a simple scorer
        // that returns a `CliffScore`. This is an `enum` with two variants: `Score(v)`
        // where `v` is the value of the items if they fit in the knapsack
        // and `Overloaded` otherwise.  This is implemented so that `Overloaded` is
        // always worse than any `Score(v)` value.
        .scorer(CliffScorer::new(knapsack))
        // Add an inspector. This is a function that is called after each generation
        // and can be used to collect and/or print out information about the run. We'll use this to
        // print out the best score in each generation, and to keep track of the best score in the run.
        .inspector(|generation_number, population| {
            report_on_generation(generation_number, population, &mut best_in_run, &mut rng);
        })
        // Now that we've specified all the elements, we can build the run.
        .build();

    let final_population = run.execute()?;

    let best = Best.select(&final_population, &mut rng)?;
    println!("Best in final generation {best:?}");
    println!("Best in overall run: {best_in_run:?}");

    // The returns the unit type `()` wrapped in the `Ok` variant of
    // `Result`. The lack of a semicolon (`;`) at the end of the line
    // makes this the last value in the function, which is what Rust
    // will return in the absence of an explicit `return` statement.
    Ok(())
}
