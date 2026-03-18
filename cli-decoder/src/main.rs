use clap::Parser;
use ndarray::Array3;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::inputs;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Decode .dac tokens back to 16kHz audio using ONNX Runtime.", long_about = None)]
struct Args {
    /// Path to the input .dac file
    input: PathBuf,

    /// Path to the decoder ONNX model
    #[arg(short, long, default_value = "../ml-model-workshop/dac_decoder_16k.onnx")]
    model: PathBuf,

    /// Path to the output .wav file
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // 1. Determine output path
    let output_path = args.output.unwrap_or_else(|| {
        let mut path = args.input.clone();
        path.set_extension("wav");
        path
    });

    println!("Loading decoder model: {:?}", args.model);
    if !args.model.exists() {
        return Err(format!("Model file not found: {:?}", args.model).into());
    }

    // 2. Initialize ONNX Runtime Session
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_file(args.model)?;

    println!("Reading input tokens: {:?}", args.input);
    let mut file = File::open(&args.input)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // 3. Convert bytes to i64 tokens
    // We expect tokens to be sent from the browser/backend as raw BigInt64Array buffers
    // In Rust, we'll cast the buffer directly to i64.
    let token_size = std::mem::size_of::<i64>();
    let token_count = buffer.len() / token_size; 
    if buffer.len() % token_size != 0 {
        println!("Warning: Buffer length is not a multiple of 8 bytes. Truncating.");
    }

    let tokens_raw: Vec<i64> = buffer
        .chunks_exact(token_size)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    // The DAC model expects tokens in shape [batch, codes, length]
    // For 16k model, it's strictly 12 codebooks.
    let codes = 12;
    let sequence_length = token_count / codes;

    if token_count % codes != 0 {
        return Err(format!(
            "Token count ({}) is not divisible by the number of codebooks ({}). \
             The stream may have been interrupted or use a different model configuration.",
            token_count, codes
        ).into());
    }

    println!("Reshaping tokens to [1, {}, {}]", codes, sequence_length);
    let tokens_array = Array3::from_shape_vec((1, codes, sequence_length), tokens_raw)?;

    // 4. Run Inference
    println!("Running inference...");
    let outputs = session.run(inputs![tokens_array]?)?;
    let reconstructed_audio = outputs[0].try_extract_tensor::<f32>()?;

    // 5. Save as WAV
    println!("Saving reconstructed audio to: {:?}", output_path);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, spec)?;
    for &sample in reconstructed_audio.iter() {
        let amplitude = i16::MAX as f32;
        writer.write_sample((sample.clamp(-1.0, 1.0) * amplitude) as i16)?;
    }
    writer.finalize()?;

    println!("Done!");
    Ok(())
}
