use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("Token ID -> String mapping:");
    for id in [15, 16, 17, 18, 19, 20, 21] {
        let s = vocab.get(id).map(|s| s.as_str()).unwrap_or("?");
        println!("  {} = {:?}", id, s);
    }

    println!("\nDigit tokens:");
    for ch in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] {
        // Search for the digit token
        for (i, tok) in vocab.iter().enumerate().take(100) {
            if tok == &ch.to_string() || tok == &format!("Ġ{}", ch) {
                println!("  '{}' = token {}", ch, i);
                break;
            }
        }
    }
}
