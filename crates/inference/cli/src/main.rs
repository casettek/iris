use std::process::ExitCode;

fn main() -> ExitCode {
    match iris_cli::run_from_env() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("iris: {error:#}");
            ExitCode::from(1)
        }
    }
}
