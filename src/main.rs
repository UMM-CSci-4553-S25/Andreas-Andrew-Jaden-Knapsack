mod cliff_score;
mod cliff_scorer;
mod item;
mod knapsack;

use std::fs::File;
use std::io::Write;
use rand::{SeedableRng, rngs::StdRng};
use cliff_score::CliffScore;
use cliff_scorer::CliffScorer;
use course_helpers::{ec_run::Run, statistics::entropy};
use ec_core::{
    individual::ec::EcIndividual,
    operator::selector::{best::Best, tournament::Tournament, Selector},
};
use ec_linear::{
    genome::bitstring::Bitstring, 
    mutator::with_one_over_length::WithOneOverLength,
    recombinator::uniform_xo::UniformXo,
};
use knapsack::Knapsack;
use rand::Rng;

// Structure to hold run results
#[derive(Debug)]
struct RunResult {
    generation_count: usize,
    run_number: usize,
    best_score: CliffScore,
    generation_found: usize,
    run_time_ms: u128,
}

fn execute_single_run(
    generations: usize,
    run_number: usize,
    knapsack: &Knapsack,
    base_seed: u64,
) -> anyhow::Result<RunResult> {
    let start_time = std::time::Instant::now();
    // Create a shared container for the best result
    let best_in_run = std::sync::Arc::new(std::sync::Mutex::new(None));
    let best_generation = std::sync::Arc::new(std::sync::Mutex::new(0));
    
    // Clone the Arc for the closure
    let best_in_run_clone = best_in_run.clone();
    let best_generation_clone = best_generation.clone();

    let run = Run::builder()
        .bit_length(knapsack.num_items())
        .max_generations(generations)
        .population_size(1_000)
        .selector(Tournament::of_size::<2>())
        .mutator(WithOneOverLength)
        .recombinator(UniformXo)
        .parallel_evaluation(true)
        .scorer(CliffScorer::new(knapsack.clone()))
        .inspector(move |generation_number, population| {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(generation_number as u64));
            let best = Best.select(population, &mut rng).unwrap();
            
            let mut current_best = best_in_run_clone.lock().unwrap();
            let mut current_gen = best_generation_clone.lock().unwrap();
            
            match &*current_best {
                None => {
                    *current_best = Some(best.clone());
                    *current_gen = generation_number;
                }
                Some(b) if best.test_results > b.test_results => {
                    *current_best = Some(best.clone());
                    *current_gen = generation_number;
                }
                _ => (),
            }
        })
        .build();

    let _final_population = run.execute()?;
    let runtime = start_time.elapsed().as_millis();

    // Get the final results from the shared containers
    let best = best_in_run.lock().unwrap();
    let gen = *best_generation.lock().unwrap();
    
    Ok(RunResult {
        generation_count: generations,
        run_number,
        best_score: best.as_ref().unwrap().test_results,
        generation_found: gen,
        run_time_ms: runtime,
    })
}

fn main() -> anyhow::Result<()> {
    let mut results_file = File::create("knapsack_results.csv")?;
    writeln!(
        results_file, 
        "experiment_type,run_number,generations,best_score,generation_found,run_time_ms"
    )?;

    let knapsack = Knapsack::from_file_path("knapsacks/big.txt")?;
    const NUM_RUNS: usize = 50;
    const BASELINE_GENERATIONS: usize = 100;
    const GENERATION_START: usize = 5;
    const INCREMENT_SIZE: usize = 5;

    // First, run the baseline experiments (250 generations, 50 times)
    println!("Running baseline experiments ({} generations)...", BASELINE_GENERATIONS);
    for run in 0..NUM_RUNS {
        let base_seed = run as u64;
        let result = execute_single_run(BASELINE_GENERATIONS, run, &knapsack, base_seed)?;
        
        writeln!(
            results_file,
            "baseline,{},{},{:?},{},{}",
            result.run_number,
            result.generation_count,
            result.best_score.to_int(),
            result.generation_found,
            result.run_time_ms
        )?;
        
        println!("Completed baseline run {} of {}", run + 1, NUM_RUNS);
    }

    // Then run the incremental experiments (10 to 230 generations, 50 times each)
    for generations in (GENERATION_START..BASELINE_GENERATIONS).step_by(INCREMENT_SIZE) {
        println!("Running experiments with {} generations...", generations);
        
        for run in 0..NUM_RUNS {
            let base_seed = (generations + run) as u64;
            let result = execute_single_run(generations, run, &knapsack, base_seed)?;
            
            writeln!(
                results_file,
                "incremental,{},{},{:?},{},{}",
                result.run_number,
                result.generation_count,
                result.best_score.to_int(),
                result.generation_found,
                result.run_time_ms
            )?;
            
            println!("Completed run {} of {} for {} generations", 
                run + 1, NUM_RUNS, generations);
        }
    }

    println!("Experiment complete! Results written to knapsack_results.csv");
    Ok(())
}