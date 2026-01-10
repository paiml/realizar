//! Test generation with Qwen2 to see actual output
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("=== Qwen2-0.5B Generation Test ===\n");

    // Test 1: Simple "2+2="
    let tokens_2plus2 = vec![17u32, 10, 17, 28]; // "2+2="
    print!("Input: 2+2= (tokens {:?})\n", tokens_2plus2);
    print!("Generating with greedy decoding...\n");

    let config = QuantizedGenerateConfig {
        max_tokens: 10,
        temperature: 0.0, // Greedy
        top_k: 1,
        stop_tokens: vec![],
    };

    let output = model.generate(&tokens_2plus2, &config)?;
    print!("Output tokens: {:?}\n", output);
    print!("Decoded: ");
    for tok in &output {
        let s = vocab.get(*tok as usize).map(|s| s.as_str()).unwrap_or("?");
        print!("{}", s);
    }
    println!("\n");

    // Test 2: Try with chat format - maybe model needs instruction format
    // Qwen2-Instruct uses ChatML format:
    // <|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n

    // Find special tokens
    let im_start = vocab
        .iter()
        .enumerate()
        .find(|(_, s)| s.contains("im_start"))
        .map(|(i, _)| i as u32);
    let im_end = vocab
        .iter()
        .enumerate()
        .find(|(_, s)| s.contains("im_end"))
        .map(|(i, _)| i as u32);

    println!("Special tokens:");
    println!("  <|im_start|>: {:?}", im_start);
    println!("  <|im_end|>: {:?}", im_end);

    // Test 3: Just "1+1="
    let tokens_1plus1 = vec![16u32, 10, 16, 28]; // "1+1="
    print!("\nInput: 1+1= (tokens {:?})\n", tokens_1plus1);

    let output2 = model.generate(&tokens_1plus1, &config)?;
    print!("Output tokens: {:?}\n", output2);
    print!("Decoded: ");
    for tok in &output2 {
        let s = vocab.get(*tok as usize).map(|s| s.as_str()).unwrap_or("?");
        print!("{}", s);
    }
    println!("\n");

    // Test 4: Test with a simple continuation prompt
    // "The answer is"
    // First find these tokens
    let the_tok = vocab
        .iter()
        .enumerate()
        .find(|(_, s)| s.as_str() == "The")
        .map(|(i, _)| i as u32);
    let answer_tok = vocab
        .iter()
        .enumerate()
        .find(|(_, s)| s.as_str() == "Ġanswer")
        .map(|(i, _)| i as u32);
    let is_tok = vocab
        .iter()
        .enumerate()
        .find(|(_, s)| s.as_str() == "Ġis")
        .map(|(i, _)| i as u32);

    println!(
        "Token lookup: The={:?}, answer={:?}, is={:?}",
        the_tok, answer_tok, is_tok
    );

    if let (Some(t1), Some(t2), Some(t3)) = (the_tok, answer_tok, is_tok) {
        let tokens_answer = vec![t1, t2, t3];
        print!("\nInput: 'The answer is' (tokens {:?})\n", tokens_answer);

        let output3 = model.generate(&tokens_answer, &config)?;
        print!("Output tokens: {:?}\n", output3);
        print!("Decoded: ");
        for tok in &output3 {
            let s = vocab.get(*tok as usize).map(|s| s.as_str()).unwrap_or("?");
            print!("{}", s);
        }
        println!();
    }

    Ok(())
}
