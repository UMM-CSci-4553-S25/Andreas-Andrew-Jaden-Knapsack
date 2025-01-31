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
    let rng = std::cell::RefCell::new(rand::rng());
    let file_path1 = "knapsacks/test.txt";
    let file_path2 = "knapsacks/big.txt";

    let knapsack1 = Knapsack::from_file_path(file_path1)?;
    let knapsack2 = Knapsack::from_file_path(file_path2)?;

    let mut best_in_run1 = None;
    let mut best_in_run2 = None;

    let run1 = Run::builder()
        .bit_length(knapsack1.num_items())
        .max_generations(1_000)
        .population_size(1_000)
        .selector(Tournament::of_size::<2>())
        .mutator(WithOneOverLength)
        .recombinator(UniformXo)
        .parallel_evaluation(true)
        .scorer(CliffScorer::new(knapsack1))
        .inspector(|generation_number, population| {
            report_on_generation(generation_number, population, &mut best_in_run1, &mut rng.borrow_mut());
        })
        .build();

    let run2 = Run::builder()
        .bit_length(knapsack2.num_items())
        .max_generations(1_000)
        .population_size(1_000)
        .selector(Tournament::of_size::<2>())
        .mutator(WithOneOverLength)
        .recombinator(UniformXo)
        .parallel_evaluation(true)
        .scorer(CliffScorer::new(knapsack2))
        .inspector(|generation_number, population| {
            report_on_generation(generation_number, population, &mut best_in_run2, &mut rng.borrow_mut());
        })
        .build();

    let final_population1 = run1.execute()?;
    let final_population2 = run2.execute()?;

    let best1 = Best.select(&final_population1, &mut rng.borrow_mut())?;
    let best2 = Best.select(&final_population2, &mut rng.borrow_mut())?;

    println!("Best in final generation for dataset 1: {best1:?}");
    println!("Best in overall run for dataset 1: {best_in_run1:?}");
    println!("Best in final generation for dataset 2: {best2:?}");
    println!("Best in overall run for dataset 2: {best_in_run2:?}");

    // Compare the results
    if let (Some(b1), Some(b2)) = (best_in_run1, best_in_run2) {
        if b1.test_results > b2.test_results {
            println!("Dataset 1 has the better result.");
        } else if b1.test_results < b2.test_results {
            println!("Dataset 2 has the better result.");
        } else {
            println!("Both datasets have equal results.");
        }
    }

    Ok(())
}
