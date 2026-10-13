use std::process::ExitCode;

fn main() -> ExitCode {
    match raster_inference_cli::run_from_env() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("raster-inference: {error:#}");
            ExitCode::from(1)
        }
    }
}
